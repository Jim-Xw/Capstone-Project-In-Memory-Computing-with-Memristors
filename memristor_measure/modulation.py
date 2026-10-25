"""LTP/LTD pulse-modulation experiment for memristors.

Purpose: alternately applies potentiation/depression write pulses and, after each pulse,
reads the resistance and conductance as the average of several 0.1 V small-signal reads.
Wiring: the device must be connected between FORCE HI and FORCE LO on the Keithley front
panel, with the FRONT terminals selected; SENSE must not be used in place of FORCE.
Software mode is timed by the host, with a minimum pulse width of 20 ms; hardware mode
applies to the 2461 only and is triggered by the instrument, which gives more accurate
timing. Example:
``python memristor_measure/modulation.py --pulse-mode software --dev DEV-1``.
The CSV keeps the original columns and appends the write-pulse set/readback voltage,
current, compliance, mode, source status and stop reason. Safety warning: the current
is hard-limited to 50 uA and no voltage may exceed 2 V in magnitude; compliance or an
incomplete pulse ends the run immediately, and every exit path returns the output to
zero and switches it off.
Series resistor: with ``--r-series <Ohm>``, R and G are corrected for the series
resistor, a device-voltage column is appended and the output name carries an RS tag.
"""

import os
import sys
import time
import argparse
from datetime import datetime

import platform
import matplotlib
_fonts = {"Windows": "Microsoft YaHei", "Darwin": "PingFang SC",
          "Linux": "WenQuanYi Micro Hei"}
matplotlib.rcParams["font.family"]        = _fonts.get(platform.system(), "DejaVu Sans")
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams.update({
    "axes.labelsize": 12, "axes.titlesize": 12, "axes.linewidth": 1.2,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "xtick.minor.width": 0.9, "ytick.minor.width": 0.9,
})
matplotlib.use("TkAgg")   # use "Agg" without a display

import matplotlib.pyplot as plt
import numpy as np

try:
    from .keithley_smu import (
        AUTO_ADDRESS,
        MAX_SAFE_CURRENT_A,
        InstrumentConnectionError,
        add_series_argument,
        apply_voltage_pulse,
        configure_voltage_source,
        connect_smu,
        device_voltage,
        read_vi,
        safe_shutdown,
        series_tag,
        validate_safety_limits,
        validate_series,
    )
except ImportError:
    from keithley_smu import (
        AUTO_ADDRESS,
        MAX_SAFE_CURRENT_A,
        InstrumentConnectionError,
        add_series_argument,
        apply_voltage_pulse,
        configure_voltage_source,
        connect_smu,
        device_voltage,
        read_vi,
        safe_shutdown,
        series_tag,
        validate_safety_limits,
        validate_series,
    )

# ══════════════════════════════════════════════════════
#  ★ Editable parameters ★
# ══════════════════════════════════════════════════════
DEFAULT_VISA_ADDRESS = AUTO_ADDRESS

# ── Pulse amplitudes (signed, configurable polarity) ────────
#   Default: positive pulses potentiate (LTP) and negative pulses depress (LTD).
#   If the device polarity is reversed, swap the signs.
V_LTP      = 0.5      # potentiation pulse amplitude (V)
V_LTD      = -0.5     # depression pulse amplitude (V)

# ── Pulse timing ─────────────────────────────────
T_PULSE    = 20e-3      # pulse width (s), minimum 20 ms in software mode
T_RELAX    = 50e-3      # relaxation wait between the write pulse and the read (s)

# ── Pulse counts ─────────────────────────────────
N_LTP      = 5       # number of potentiation pulses
N_LTD      = 5       # number of depression pulses
N_CYCLES   = 1        # number of LTP-LTD cycle repeats (1 for v1)

# ── Initial RESET (diagnostic) ──────────────────────
#   Purpose: drive the device to HRS before the LTP segment, so that LTP has room to rise
#   Note: use a stronger single voltage than the LTD pulses (the pulse amplitude is
#   often not enough for a full reset)
#   Diagnostic logic:
#     LTP rises monotonically after RESET  → the start point was "stuck"; the device is usable
#     still ~2% random scatter after RESET → dynamic range too small; improve the device first
ENABLE_INIT_RESET = False  # the 20 initial RESET pulses are disabled by default
V_INIT_RESET   = -0.6      # initial RESET voltage (V)
T_INIT_RESET   = 50e-3     # initial RESET pulse width (s), longer to ensure a full reset
N_INIT_RESET   = 20         # number of initial RESET pulses
RESET_MIN_DROP = 0.1       # criterion: if G falls by less than this fraction after RESET,
                           # the device is flagged as not resetting

# ── Read parameters ──────────────────────────────
V_READ     = 0.1      # read voltage (V), far below the switching voltage, non-destructive
N_READ_AVG = 10      # samples averaged per read
T_READ_SETTLE = 5e-3  # read settling time (s)

# ── Protection ───────────────────────────────────
I_CC       = MAX_SAFE_CURRENT_A  # compliance current (A), must not exceed 50 uA
NPLC       = 0.1      # integration time (short, for faster reads between pulses)

# ── Series resistor ──────────────────────────────
R_SERIES   = 0.0      # series resistor on the HI lead (Ohm), set by --r-series; 0 = none
# ══════════════════════════════════════════════════════

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "Pulse_LTP_LTD")


# ──────────────────────────────────────────────────────
#  Connection
# ──────────────────────────────────────────────────────
def fmt_series(r_series: float) -> str:
    """Series-resistor label, e.g. '10 kOhm'."""
    return f"{r_series / 1e3:g} kOhm" if r_series >= 1e3 else f"{r_series:g} Ohm"


def connect(address: str):
    try:
        return connect_smu(address)
    except InstrumentConnectionError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


# ──────────────────────────────────────────────────────
#  SMU initialisation
# ──────────────────────────────────────────────────────
def init_smu(inst, i_cc: float, nplc: float):
    actual_icc = configure_voltage_source(inst, i_cc, nplc)
    print(f"[INIT] I_cc = {actual_icc*1e6:.0f} uA, NPLC = {nplc}, done")
    return actual_icc


# ──────────────────────────────────────────────────────
#  Read conductance (small signal, average of N): non-destructive
#  Returns (G in siemens, R in ohms, I in amperes)
# ──────────────────────────────────────────────────────
def read_conductance(inst, v_read: float, n_avg: int,
                     settle: float) -> tuple:
    validate_safety_limits(v_read, I_CC)
    currents = []
    compliance = False
    try:
        inst.write(f":SOUR:VOLT:LEV {v_read:.9g}")
        inst.write(":OUTP ON")
        time.sleep(settle)
        for _ in range(n_avg):
            reading = read_vi(inst, v_read, I_CC)
            compliance = compliance or reading.compliance
            if np.isfinite(reading.current_a):
                currents.append(reading.current_a)
            time.sleep(0.001)
    finally:
        safe_shutdown(inst)

    if compliance or not currents:
        return float("nan"), float("nan"), float(np.mean(currents)) if currents else float("nan")
    i_mean = float(np.mean(currents))
    if abs(i_mean) > 1e-13:
        resistance = abs(v_read / i_mean) - R_SERIES   # device resistance (series R subtracted)
        if resistance <= 0:
            return float("nan"), resistance, i_mean
        return 1.0 / resistance, resistance, i_mean
    return 0.0, float("inf"), i_mean


# ──────────────────────────────────────────────────────
#  Apply one write pulse → return to zero → relax
# ──────────────────────────────────────────────────────
def apply_pulse(inst, mode: str, v_pulse: float, t_pulse: float,
                t_relax: float):
    return apply_voltage_pulse(
        inst, mode=mode, voltage_v=v_pulse, width_s=t_pulse,
        relax_s=t_relax, current_limit_a=I_CC,
    )


# ──────────────────────────────────────────────────────
#  Initial RESET (diagnostic): drive the device to HRS and quantify the fall in conductance
#  Returns (G_before, G_after, diagnostic string)
# ──────────────────────────────────────────────────────
def initial_reset(inst, pulse_mode: str) -> tuple:
    print(f"\n{'─'*56}")
    print(f"  Initial RESET (diagnostic)")
    print(f"  V = {V_INIT_RESET:+.2f} V × {N_INIT_RESET} pulses, "
          f"width {T_INIT_RESET*1e3:.0f} ms")
    print(f"{'─'*56}")

    G_before, R_before, _ = read_conductance(inst, V_READ,
                                             N_READ_AVG, T_READ_SETTLE)
    print(f"  Before RESET: G = {G_before*1e6:.3f} uS   R = {R_before/1e3:.2f} kΩ")

    reset_records = []
    for k in range(N_INIT_RESET):
        pulse = apply_pulse(inst, pulse_mode, V_INIT_RESET, T_INIT_RESET, T_RELAX)
        if pulse.compliance or not pulse.completed:
            G, R = float("nan"), float("nan")
        else:
            G, R, _ = read_conductance(inst, V_READ, N_READ_AVG, T_READ_SETTLE)
        reset_records.append((
            -1, "INIT_RESET", k, V_INIT_RESET, G, R, float("nan"),
            pulse.voltage_set_v, pulse.voltage_actual_v, pulse.current_a,
            pulse.compliance, pulse.mode, pulse.source_status,
            pulse.stopped_reason, pulse.completed, pulse.relative_time_s,
        ))
        print(f"  reset #{k}: G = {G*1e6:.3f} uS   R = {R/1e3:.2f} kΩ")
        if pulse.compliance or not pulse.completed:
            safe_shutdown(inst)
            print(f"  [STOP] Initial RESET fault: {pulse.stopped_reason or 'pulse_incomplete'}")
            return G_before, G, "Initial RESET stopped by a safety condition", reset_records, True

    G_after, R_after, _ = read_conductance(inst, V_READ,
                                           N_READ_AVG, T_READ_SETTLE)
    print(f"  After RESET:  G = {G_after*1e6:.3f} uS   R = {R_after/1e3:.2f} kΩ")

    # ── Diagnostic criterion ─────────────────────────
    if G_before > 0:
        drop = (G_before - G_after) / G_before
    else:
        drop = 0.0

    if drop >= RESET_MIN_DROP:
        diag = (f"[OK] Conductance fell by {drop*100:.1f}%; RESET effective, "
                f"LTP has room to rise")
    else:
        diag = (f"(!) Conductance changed by only {drop*100:.1f}% (<{RESET_MIN_DROP*100:.0f}%); "
                f"this voltage may not reset the device. Check V_INIT_RESET / device window")
    print(f"  Diagnosis: {diag}")

    return G_before, G_after, diag, reset_records, False


# ──────────────────────────────────────────────────────
#  One complete LTP-LTD cycle
#  Returns a dict of phase/pulse_idx/G/R/I lists
# ──────────────────────────────────────────────────────
def run_ltp_ltd_cycle(inst, cycle_idx: int, pulse_mode: str) -> tuple:
    records = []   # each entry: (cycle, phase, pulse_idx, v_pulse, G, R, I)

    print(f"\n{'─'*56}")
    print(f"  Cycle {cycle_idx+1} / {N_CYCLES}")
    print(f"{'─'*56}")

    # ── LTP segment ──────────────────────────────────
    print(f"  [LTP] V_pulse = {V_LTP:+.2f} V × {N_LTP} pulses")
    print(f"  {'#':>4}  {'G (uS)':>10}  {'R (kΩ)':>10}")
    for k in range(N_LTP):
        pulse = apply_pulse(inst, pulse_mode, V_LTP, T_PULSE, T_RELAX)
        if pulse.compliance or not pulse.completed:
            G, R, I = float("nan"), float("nan"), float("nan")
        else:
            G, R, I = read_conductance(inst, V_READ, N_READ_AVG, T_READ_SETTLE)
        records.append((
            cycle_idx, "LTP", k, V_LTP, G, R, I,
            pulse.voltage_set_v, pulse.voltage_actual_v, pulse.current_a,
            pulse.compliance, pulse.mode, pulse.source_status,
            pulse.stopped_reason, pulse.completed, pulse.relative_time_s,
        ))
        if k % 4 == 0 or k == N_LTP - 1:
            print(f"  {k:>4}  {G*1e6:>10.3f}  {R/1e3:>10.2f}")
        if pulse.compliance or not pulse.completed:
            safe_shutdown(inst)
            return records, True

    # ── LTD segment ──────────────────────────────────
    print(f"  [LTD] V_pulse = {V_LTD:+.2f} V × {N_LTD} pulses")
    print(f"  {'#':>4}  {'G (uS)':>10}  {'R (kΩ)':>10}")
    for k in range(N_LTD):
        pulse = apply_pulse(inst, pulse_mode, V_LTD, T_PULSE, T_RELAX)
        if pulse.compliance or not pulse.completed:
            G, R, I = float("nan"), float("nan"), float("nan")
        else:
            G, R, I = read_conductance(inst, V_READ, N_READ_AVG, T_READ_SETTLE)
        records.append((
            cycle_idx, "LTD", k, V_LTD, G, R, I,
            pulse.voltage_set_v, pulse.voltage_actual_v, pulse.current_a,
            pulse.compliance, pulse.mode, pulse.source_status,
            pulse.stopped_reason, pulse.completed, pulse.relative_time_s,
        ))
        if k % 4 == 0 or k == N_LTD - 1:
            print(f"  {k:>4}  {G*1e6:>10.3f}  {R/1e3:>10.2f}")
        if pulse.compliance or not pulse.completed:
            safe_shutdown(inst)
            return records, True

    return records, False





# ──────────────────────────────────────────────────────
#  Save CSV
# ──────────────────────────────────────────────────────
def save_csv(all_records: list, path: str, r_series: float = None):
    if r_series is None:
        r_series = R_SERIES
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cycle", "phase", "pulse_idx", "v_pulse_V",
                    "G_S", "G_uS", "R_Ohm", "I_A",
                    "pulse_voltage_set_V", "pulse_voltage_actual_V",
                    "pulse_current_A", "compliance", "pulse_mode",
                    "source_status", "stopped_reason", "completed",
                    "pulse_relative_time_s",
                    "device_voltage_V", "pulse_device_voltage_V",
                    "r_series_Ohm"])
        for record in all_records:
            (cyc, phase, idx, vp, G, R, I, pv_set, pv_actual,
             pulse_i, compliance, pulse_mode, source_status,
             stopped_reason, completed, pulse_time) = record
            v_dev = device_voltage(V_READ, I, r_series)
            pv_src = pv_actual if np.isfinite(pv_actual) else pv_set
            pv_dev = device_voltage(pv_src, pulse_i, r_series)
            w.writerow([
                cyc, phase, idx, f"{vp:.4f}",
                f"{G:.6e}", f"{G*1e6:.4f}",
                f"{R:.2f}" if np.isfinite(R) else ("nan" if np.isnan(R) else "inf"),
                f"{I:.6e}",
                f"{pv_set:.6e}", f"{pv_actual:.6e}",
                f"{pulse_i:.6e}", compliance, pulse_mode,
                "" if source_status is None else source_status,
                stopped_reason, completed,
                "" if pulse_time is None else f"{pulse_time:.6e}",
                f"{v_dev:.6e}", f"{pv_dev:.6e}",
                f"{r_series:g}",
            ])
    print(f"\n[CSV] Saved: {path}")


# ──────────────────────────────────────────────────────
#  Plot: G vs pulse number (LTP rising + LTD falling)
# ──────────────────────────────────────────────────────
def plot_ltp_ltd(all_records: list, img_path: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    ax.tick_params(which="both", colors="#FFFFFF", labelsize=11)
    for sp in ax.spines.values():
        sp.set_color("#FFFFFF")

    cycles = sorted(set(r[0] for r in all_records))
    cmap_ltp = ["#2196F3", "#42A5F5", "#64B5F6", "#90CAF9"]
    cmap_ltd = ["#FF5252", "#FF7043", "#FF8A65", "#FFAB91"]

    global_x = 0
    for cyc in cycles:
        ltp = [r for r in all_records if r[0] == cyc and r[1] == "LTP"]
        ltd = [r for r in all_records if r[0] == cyc and r[1] == "LTD"]

        x_ltp = list(range(global_x, global_x + len(ltp)))
        g_ltp = [r[4]*1e6 for r in ltp]
        global_x += len(ltp)

        x_ltd = list(range(global_x, global_x + len(ltd)))
        g_ltd = [r[4]*1e6 for r in ltd]
        global_x += len(ltd)

        c_ltp = cmap_ltp[cyc % len(cmap_ltp)]
        c_ltd = cmap_ltd[cyc % len(cmap_ltd)]

        ax.plot(x_ltp, g_ltp, "-o", color=c_ltp, ms=4, lw=1.5,
                label=f"LTP cyc{cyc+1}" if len(cycles) > 1 else "LTP (potentiation)")
        ax.plot(x_ltd, g_ltd, "-s", color=c_ltd, ms=4, lw=1.5,
                label=f"LTD cyc{cyc+1}" if len(cycles) > 1 else "LTD (depression)")

    ax.set_xlabel("Pulse number (#)", color="#FFFFFF", fontsize=12)
    ax.set_ylabel("Conductance G (uS)", color="#FFFFFF", fontsize=12)
    ax.set_title(
        f"LTP / LTD pulse modulation   "
        f"V_LTP={V_LTP:+.1f}V  V_LTD={V_LTD:+.1f}V  "
        f"t_pulse={T_PULSE*1e3:.0f}ms  V_read={V_READ}V"
        + (f"  Rs={fmt_series(R_SERIES)}" if R_SERIES > 0 else ""),
        color="#FFFFFF", fontsize=10, pad=10)
    ax.legend(facecolor="#000000", edgecolor="#FFFFFF",
              labelcolor="#FFFFFF", fontsize=9)
    ax.grid(True, color="#3A3A3A", lw=0.5)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.99, 0.01, f"Ni/AlOx/TiOx/Ti   {ts}   (ms-pulse proxy)",
             color="#888888", fontsize=7, ha="right")

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[PNG] Saved: {img_path}")
    plt.show()


# ──────────────────────────────────────────────────────
#  Main program
# ──────────────────────────────────────────────────────
def main():
    global R_SERIES
    parser = argparse.ArgumentParser(description="Memristor LTP/LTD pulse modulation v1")
    parser.add_argument("--address", default=DEFAULT_VISA_ADDRESS)
    parser.add_argument("--dev", default="DEV",
                        help="device label used in the folder name")
    parser.add_argument("--pulse-mode", choices=("software", "hardware"),
                        default="software", help="write-pulse mode (default software)")
    add_series_argument(parser)
    parser.set_defaults(r_series=R_SERIES)
    args = parser.parse_args()
    validate_series(args.r_series)
    R_SERIES = args.r_series
    for voltage in (V_READ, V_LTP, V_LTD, V_INIT_RESET):
        validate_safety_limits(voltage, I_CC)
    if args.pulse_mode == "software" and min(T_PULSE, T_INIT_RESET) < 20e-3:
        parser.error("pulse width must not be below 20 ms in software mode")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if R_SERIES > 0:
        ts_str = f"{series_tag(R_SERIES)}_{ts_str}"   # ..._RS10k_<timestamp>
    run_name = (
        f"{args.dev}_"
        f"LTP{V_LTP:+.1f}_LTD{V_LTD:+.1f}_"
        f"msPulse{int(T_PULSE*1000)}_"
        f"N{N_LTP}x{N_LTD}_"
        f"{ts_str}"
    )
    RUN_DIR = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(RUN_DIR, exist_ok=True)
    csv_path = os.path.join(RUN_DIR, f"ltp_ltd_{ts_str}.csv")
    img_path = os.path.join(RUN_DIR, f"ltp_ltd_{ts_str}.png")

    # Confirm parameters
    print("=" * 56)
    print("  LTP/LTD pulse-modulation experiment  v1  (procedure check)")
    print("=" * 56)
    print(f"  V_LTP (pot.)  = {V_LTP:+.2f} V")
    print(f"  V_LTD (dep.)  = {V_LTD:+.2f} V")
    print(f"  Pulse width   = {T_PULSE*1e3:.1f} ms  (ms-scale quasi-pulse)")
    print(f"  Relax wait    = {T_RELAX*1e3:.1f} ms")
    print(f"  Pulses        = LTP {N_LTP} + LTD {N_LTD}")
    print(f"  Cycles        = {N_CYCLES}")
    print(f"  Read voltage  = {V_READ} V  ({N_READ_AVG}-sample average)")
    print(f"  I_cc          = {I_CC*1e6:.0f} uA")
    print(f"  Pulse mode    = {args.pulse_mode}")
    if R_SERIES > 0:
        print(f"  Series R      = {fmt_series(R_SERIES)} (subtracted from R; device voltage logged)")
    n_pulses = (N_LTP + N_LTD) * N_CYCLES
    est_s    = n_pulses * (T_PULSE + T_RELAX + T_READ_SETTLE + 0.02)
    print(f"  Total pulses  = {n_pulses}   est. time ≈ {est_s:.0f} s")
    print(f"  Device        = {args.dev}")
    print(f"  Output dir    = {RUN_DIR}")
    print("=" * 56)
    print("  Note: 2450 SMU ms-scale quasi-pulses are a slow proxy for ns-pulse behaviour")
    print("=" * 56)

    ans = input("\nStart? (Enter / q to quit): ").strip().lower()
    if ans == "q":
        return

    inst = connect(args.address)
    all_records = []
    reset_diag = None
    G0 = float("nan")
    try:
        init_smu(inst, I_CC, NPLC)
        G0, R0, I0 = read_conductance(inst, V_READ, N_READ_AVG, T_READ_SETTLE)
        print(f"\n[BASELINE] Initial G = {G0*1e6:.3f} uS   R = {R0/1e3:.2f} kΩ")

        stop_requested = False
        if ENABLE_INIT_RESET:
            _, _, reset_diag, reset_records, stop_requested = initial_reset(
                inst, args.pulse_mode
            )
            all_records.extend(reset_records)
        for cyc in range(N_CYCLES):
            if stop_requested:
                break
            recs, stop_requested = run_ltp_ltd_cycle(
                inst, cyc, args.pulse_mode
            )
            all_records.extend(recs)
            if stop_requested:
                print("\n[SAFETY STOP] Compliance or incomplete pulse; remaining pulses stopped.")
                break
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Stopped by the user; saving the data collected so far...")
    finally:
        safe_shutdown(inst)
        inst.close()

    if not all_records:
        print("[RESULT] No data"); return

    # Summary
    g_all = [r[4]*1e6 for r in all_records]
    print("\n" + "=" * 56)
    print("  Summary")
    print("=" * 56)
    print(f"  Initial G     : {G0*1e6:.3f} uS")
    print(f"  G range       : {min(g_all):.3f} ~ {max(g_all):.3f} uS")
    if min(g_all) > 0:
        ratio = max(g_all) / min(g_all)
        print(f"  G_max/G_min   : {ratio:.2f}×")
    print(f"  Points        : {len(all_records)}")

    # ── LTP/LTD monotonicity diagnosis ───────────────
    ltp_g = [r[4]*1e6 for r in all_records if r[1] == "LTP"]
    ltd_g = [r[4]*1e6 for r in all_records if r[1] == "LTD"]
    if len(ltp_g) > 2 and len(ltd_g) > 2:
        # Net change vs point-to-point scatter
        ltp_net = ltp_g[-1] - ltp_g[0]          # LTP net rise
        ltd_net = ltd_g[0] - ltd_g[-1]          # LTD net fall
        noise   = float(np.std(np.diff(ltp_g + ltd_g)))  # point-to-point scatter
        span    = max(g_all) - min(g_all)
        print(f"  LTP net change: {ltp_net:+.3f} uS")
        print(f"  LTD net change: {ltd_net:+.3f} uS (positive = fall)")
        print(f"  Scatter std   : {noise:.3f} uS")
        print("  " + "-" * 50)
        # Diagnostic conclusion
        if ltp_net > 2 * noise and ltd_net > 2 * noise:
            print("  [OK] LTP rise and LTD fall both clearly > scatter → device can be modulated")
            print("       the earlier problem was the start point; effective once RESET is added")
        else:
            print("  (!) LTP/LTD net change not clearly above the scatter")
            print("      → dynamic range too small; improve the device first (on/off ratio limits it)")
            print("      → see notes: Pt electrode / AlOx position / thickness gradient")
    if reset_diag:
        print("  " + "-" * 50)
        print(f"  RESET diag.   : {reset_diag}")
    print("=" * 56)

    save_csv(all_records, csv_path)
    plot_ltp_ltd(all_records, img_path)


if __name__ == "__main__":
    main()