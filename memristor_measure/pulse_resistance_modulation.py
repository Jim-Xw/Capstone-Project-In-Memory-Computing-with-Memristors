"""Positive/negative pulse resistance-modulation experiment for memristors.

Purpose: applies potentiation/depression write pulses one at a time and, after each
write, reads the resistance as the average of several small-signal reads.
Wiring: connect the device to FORCE HI and FORCE LO on the Keithley front panel and
select the FRONT terminals; do not connect the device to SENSE in place of FORCE.
In software mode the pulses are timed by the host, with a minimum width of 20 ms;
in hardware mode they are triggered by the 2461, which gives more accurate timing but
is not available on the 2450. Example:
``python memristor_measure/pulse_resistance_modulation.py --pulse-mode software
--v-pot 0.5 --v-dep -0.5 --t-pulse 0.02``. The CSV keeps the original column order
and appends the write-pulse set and readback voltage, current, compliance, mode,
source status and stop reason; the JSON keeps the original fields and appends the
mode and safety information. Safety warning: the default Icc is 50 uA and can be
raised with --icc; no voltage may exceed 2 V in magnitude; --r-series corrects for a
series resistor; --icc-pot/--icc-dep set the compliance of the two write-pulse types
separately (default: same as --icc; reads always use --icc). Any fault stops the
remaining pulses immediately and returns the output safely to zero.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import matplotlib
import platform

_fonts = {"Windows": "Microsoft YaHei", "Darwin": "PingFang SC", "Linux": "WenQuanYi Micro Hei"}
matplotlib.rcParams["font.family"] = _fonts.get(platform.system(), "DejaVu Sans")
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams.update({
    "axes.labelsize": 12, "axes.titlesize": 12, "axes.linewidth": 1.2,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "xtick.minor.width": 0.9, "ytick.minor.width": 0.9,
})
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

try:
    from .keithley_smu import (
        AUTO_ADDRESS,
        MAX_SAFE_CURRENT_A,
        InstrumentConnectionError,
        apply_voltage_pulse,
        configure_voltage_source,
        connect_smu,
        read_vi,
        safe_shutdown,
        series_tag,
        set_current_limit,
        validate_safety_limits,
    )
except ImportError:
    from keithley_smu import (
        AUTO_ADDRESS,
        MAX_SAFE_CURRENT_A,
        InstrumentConnectionError,
        apply_voltage_pulse,
        configure_voltage_source,
        connect_smu,
        read_vi,
        safe_shutdown,
        series_tag,
        set_current_limit,
        validate_safety_limits,
    )

DEFAULT_VISA_ADDRESS = AUTO_ADDRESS
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pulse_Modulation")

V_READ = 0.1
V_POT = 0.9
V_DEP = -0.9
T_PULSE = 0.02
T_RELAX = 0.05
N_POT = 0
N_DEP = 3000
N_CYCLES = 1
I_CC = MAX_SAFE_CURRENT_A
NPLC = 0.1
N_READ_AVG = 10
READ_SETTLE = 0.02
R_SERIES = [0.0]  # set from --r-series in main()


def connect(address: str):
    try:
        return connect_smu(address)
    except InstrumentConnectionError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


def init_smu(inst, i_cc: float, nplc: float):
    actual_icc = configure_voltage_source(inst, i_cc, nplc)
    print(f"[Init] Voltage source ready, Icc={actual_icc:.6e} A, NPLC={nplc}")
    return actual_icc


def resistance_from(voltage: float, current: float) -> float:
    if not np.isfinite(current) or abs(current) <= 1e-14:
        return float("inf")
    return abs(voltage / current)


def fmt_resistance(resistance: float) -> str:
    if not np.isfinite(resistance):
        return "inf Ohm"
    if resistance >= 1e9:
        return f"{resistance / 1e9:.3f} GOhm"
    if resistance >= 1e6:
        return f"{resistance / 1e6:.3f} MOhm"
    if resistance >= 1e3:
        return f"{resistance / 1e3:.3f} kOhm"
    return f"{resistance:.3f} Ohm"


def read_resistance(inst, v_read: float, icc: float, n_avg: int,
                    settle: float, r_series: float = 0.0):
    currents = []
    compliance = False
    # Pulses may leave a higher per-phase limit on the instrument; restore it.
    set_current_limit(inst, icc)
    try:
        inst.write(f":SOUR:VOLT:LEV {v_read:.9g}")
        inst.write(":OUTP ON")
        time.sleep(settle)
        for _ in range(n_avg):
            try:
                reading = read_vi(inst, v_read, icc)
                currents.append(reading.current_a)
                compliance = compliance or reading.compliance
            except Exception as exc:
                print(f"[Read warning] {exc}")
            time.sleep(0.002)
    finally:
        safe_shutdown(inst)

    if not currents:
        return float("nan"), float("nan"), float("nan")

    current = float(np.mean(currents))
    if compliance:
        return current, float("nan"), float("nan")
    resistance = resistance_from(v_read, current) - r_series  # device alone
    conductance = 0.0 if not np.isfinite(resistance) or resistance <= 0 else 1.0 / resistance
    return current, resistance, conductance


def add_record(records, cycle, phase, pulse_index, global_pulse, elapsed,
               v_pulse, current, resistance, conductance, pulse=None):
    records.append(
        {
            "index": len(records),
            "cycle": cycle,
            "phase": phase,
            "pulse_index": pulse_index,
            "global_pulse": global_pulse,
            "time_s": elapsed,
            "pulse_voltage_V": v_pulse,
            "current_A": current,
            "resistance_Ohm": resistance,
            "conductance_S": conductance,
            "pulse_voltage_set_V": float("nan") if pulse is None else pulse.voltage_set_v,
            "pulse_voltage_actual_V": float("nan") if pulse is None else pulse.voltage_actual_v,
            "pulse_current_A": float("nan") if pulse is None else pulse.current_a,
            "compliance": False if pulse is None else pulse.compliance,
            "pulse_mode": "" if pulse is None else pulse.mode,
            "source_status": "" if pulse is None or pulse.source_status is None else pulse.source_status,
            "stopped_reason": "" if pulse is None else pulse.stopped_reason,
            "completed": True if pulse is None else pulse.completed,
            "pulse_relative_time_s": "" if pulse is None or pulse.relative_time_s is None else pulse.relative_time_s,
            "pulse_device_voltage_V": float("nan") if pulse is None
            else pulse.voltage_actual_v - pulse.current_a * R_SERIES[0],
        }
    )


def save_csv(records, csv_path: str):
    columns = [
        "index",
        "cycle",
        "phase",
        "pulse_index",
        "global_pulse",
        "time_s",
        "pulse_voltage_V",
        "current_A",
        "resistance_Ohm",
        "conductance_S",
        "pulse_voltage_set_V",
        "pulse_voltage_actual_V",
        "pulse_current_A",
        "compliance",
        "pulse_mode",
        "source_status",
        "stopped_reason",
        "completed",
        "pulse_relative_time_s",
        "pulse_device_voltage_V",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in records:
            out = dict(row)
            for key in ("time_s", "pulse_voltage_V", "current_A",
                        "resistance_Ohm", "conductance_S",
                        "pulse_voltage_set_V", "pulse_voltage_actual_V",
                        "pulse_current_A", "pulse_relative_time_s",
                        "pulse_device_voltage_V"):
                value = out[key]
                if isinstance(value, float):
                    out[key] = (
                        f"{value:.8e}" if np.isfinite(value)
                        else ("nan" if np.isnan(value) else "inf")
                    )
            writer.writerow(out)
    print(f"[CSV] Saved -> {csv_path}")


def save_metadata(args, meta_path: str):
    payload = vars(args).copy()
    payload["created_at"] = datetime.now().isoformat(timespec="seconds")
    payload["note"] = "Keithley 2450/2461 safe shared pulse API."
    payload["max_safe_current_A"] = MAX_SAFE_CURRENT_A
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[JSON] Saved -> {meta_path}")


def plot_records(records, img_path: str, title: str):
    if not records:
        return

    x = np.array([r["global_pulse"] for r in records], dtype=float)
    resistances = np.array([r["resistance_Ohm"] for r in records], dtype=float)
    conductances = np.array([r["conductance_S"] for r in records], dtype=float)
    phases = [r["phase"] for r in records]

    fig, (ax_r, ax_g) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.patch.set_facecolor("#000000")
    for ax in (ax_r, ax_g):
        ax.set_facecolor("#000000")
        ax.tick_params(which="both", colors="#FFFFFF", labelsize=11)
        for spine in ax.spines.values():
            spine.set_color("#FFFFFF")
        ax.grid(True, color="#3A3A3A", lw=0.5, which="both")

    colors = {"baseline": "#B0BEC5", "potentiation": "#4DB6AC", "depression": "#FFB74D"}
    for phase in sorted(set(phases)):
        mask = np.array([p == phase for p in phases])
        finite_r = mask & np.isfinite(resistances)
        ax_r.plot(x[finite_r], resistances[finite_r], "o-", ms=4, lw=1.2, color=colors.get(phase, "#90CAF9"), label=phase)
        ax_g.plot(x[mask], conductances[mask] * 1e6, "o-", ms=4, lw=1.2, color=colors.get(phase, "#90CAF9"), label=phase)

    if np.any(np.isfinite(resistances)):
        ax_r.set_yscale("log")
    ax_r.set_ylabel("Resistance (Ohm)", color="#FFFFFF")
    ax_r.set_title("Resistance after each pulse", color="#FFFFFF", fontsize=11)
    ax_r.legend(facecolor="#000000", edgecolor="#FFFFFF", labelcolor="#FFFFFF", fontsize=8)

    ax_g.set_xlabel("Pulse number", color="#FFFFFF")
    ax_g.set_ylabel("Conductance (uS)", color="#FFFFFF")
    ax_g.set_title("Conductance modulation", color="#FFFFFF", fontsize=11)

    fig.suptitle(title, color="#FFFFFF", fontsize=10)
    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[PNG] Saved -> {img_path}")
    plt.show()


def make_run_dir(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = series_tag(args.r_series)
    ts = f"{tag}_{ts}" if tag else ts  # e.g. RS10k_20261009_170000
    run_name = (
        f"{args.dev}_POT{args.v_pot:+.2f}_DEP{args.v_dep:+.2f}_"
        f"pulse{int(args.t_pot * 1000)}x{int(args.t_dep * 1000)}ms_N{args.n_pot}x{args.n_dep}_cyc{args.cycles}_{ts}"
    )
    run_dir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, ts


def run_phase(inst, args, records, cycle, phase, voltage, count, global_pulse, t_zero, icc, width):
    print(f"\n[Cycle {cycle}] {phase} pulses: V={voltage:+.4g} V, width={width:.4g} s, "
          f"count={count}, Icc={icc:.3e} A")
    for idx in range(1, count + 1):
        pulse = apply_voltage_pulse(
            inst, mode=args.pulse_mode, voltage_v=voltage,
            width_s=width, relax_s=args.t_relax,
            current_limit_a=icc,
        )
        overcurrent = (
            np.isfinite(pulse.current_a)
            and abs(pulse.current_a) > icc * 1.02
        )
        if overcurrent or (not pulse.completed and not pulse.compliance):
            current = resistance = conductance = float("nan")
        else:
            current, resistance, conductance = read_resistance(
                inst, args.v_read, args.icc, args.n_read_avg, args.read_settle, args.r_series
            )
        global_pulse += 1
        add_record(records, cycle, phase, idx, global_pulse,
                   time.time() - t_zero, voltage, current, resistance,
                   conductance, pulse)
        note = "  [compliance]" if pulse.compliance else ""
        print(
            f"  pulse={global_pulse:04d}  {phase:>13} #{idx:03d}  "
            f"R={fmt_resistance(resistance):>14}  G={conductance*1e6:9.3f} uS"
            f"{note}"
        )
        if overcurrent or (not pulse.completed and not pulse.compliance):
            safe_shutdown(inst)
            print(f"[Safety stop] {pulse.stopped_reason or 'pulse_incomplete'}")
            return global_pulse, True
    return global_pulse, False


def main():
    parser = argparse.ArgumentParser(description="Memristor pulse resistance modulation test")
    parser.add_argument("--address", default=DEFAULT_VISA_ADDRESS, help="VISA resource address")
    parser.add_argument("--dev", default="DEV", help="Device label for output folder naming")
    parser.add_argument("--v-read", type=float, default=V_READ, help="Non-destructive read voltage in V")
    parser.add_argument("--v-pot", type=float, default=V_POT, help="Potentiation pulse voltage in V")
    parser.add_argument("--v-dep", type=float, default=V_DEP, help="Depression pulse voltage in V")
    parser.add_argument("--t-pulse", type=float, default=T_PULSE, help="Pulse width in seconds (both phases unless overridden)")
    parser.add_argument("--t-pot", type=float, default=None, help="Potentiation pulse width in s (default: --t-pulse)")
    parser.add_argument("--t-dep", type=float, default=None, help="Depression pulse width in s (default: --t-pulse)")
    parser.add_argument("--t-relax", type=float, default=T_RELAX, help="Relaxation time after each pulse in seconds")
    parser.add_argument("--n-pot", type=int, default=N_POT, help="Number of potentiation pulses per cycle")
    parser.add_argument("--n-dep", type=int, default=N_DEP, help="Number of depression pulses per cycle")
    parser.add_argument("--cycles", type=int, default=N_CYCLES, help="Number of potentiation/depression cycles")
    parser.add_argument("--icc", type=float, default=I_CC, help="Source current compliance in A (reads, and pulses unless overridden)")
    parser.add_argument("--icc-pot", type=float, default=None, help="Compliance for potentiation pulses in A (default: --icc)")
    parser.add_argument("--icc-dep", type=float, default=None, help="Compliance for depression pulses in A (default: --icc)")
    parser.add_argument("--nplc", type=float, default=NPLC, help="Current measurement NPLC")
    parser.add_argument("--n-read-avg", type=int, default=N_READ_AVG, help="Number of read samples to average")
    parser.add_argument("--read-settle", type=float, default=READ_SETTLE, help="Settling time before each read in seconds")
    parser.add_argument("--pulse-mode", choices=("software", "hardware"),
                        default="software", help="Pulse mode (default: software)")
    parser.add_argument("--r-series", type=float, default=0.0,
                        help="Series resistor in the HI line, Ohm; subtracted from reads, device voltage logged")
    parser.add_argument("--no-plot", action="store_true", help="Save CSV only")
    args = parser.parse_args()
    R_SERIES[0] = args.r_series
    if args.icc_pot is None:
        args.icc_pot = args.icc
    if args.icc_dep is None:
        args.icc_dep = args.icc
    if args.t_pot is None:
        args.t_pot = args.t_pulse
    if args.t_dep is None:
        args.t_dep = args.t_pulse

    validate_safety_limits(args.v_read, args.icc)
    validate_safety_limits(args.v_pot, args.icc_pot)
    validate_safety_limits(args.v_dep, args.icc_dep)
    for width in (args.t_pot, args.t_dep):
        if width <= 0:
            raise ValueError("pulse widths must be > 0")
        if args.pulse_mode == "software" and width < 20e-3:
            raise ValueError("software pulse width must be >= 0.02 s")
    if args.t_relax < 0:
        raise ValueError("--t-relax must be >= 0")
    if args.n_pot < 0 or args.n_dep < 0:
        raise ValueError("--n-pot and --n-dep must be >= 0")
    if args.cycles <= 0:
        raise ValueError("--cycles must be > 0")
    if args.n_read_avg <= 0:
        raise ValueError("--n-read-avg must be > 0")

    run_dir, ts = make_run_dir(args)
    csv_path = os.path.join(run_dir, f"pulse_modulation_{ts}.csv")
    img_path = os.path.join(run_dir, f"pulse_modulation_{ts}.png")
    meta_path = os.path.join(run_dir, f"pulse_modulation_{ts}.json")

    total_pulses = (args.n_pot + args.n_dep) * args.cycles
    est_time = args.cycles * (args.n_pot * args.t_pot + args.n_dep * args.t_dep) + total_pulses * (args.t_relax + args.read_settle + 0.01)
    print("=" * 68)
    print("  Pulse Resistance Modulation Test")
    print("=" * 68)
    print(f"  Device       : {args.dev}")
    print(f"  Potentiation : {args.v_pot:+.4g} V x {args.n_pot} pulses")
    print(f"  Depression   : {args.v_dep:+.4g} V x {args.n_dep} pulses")
    print(f"  Pulse width  : pot {args.t_pot:.4g} s, dep {args.t_dep:.4g} s")
    print(f"  Relax        : {args.t_relax:.4g} s")
    print(f"  Read         : {args.v_read:+.4g} V, avg={args.n_read_avg}")
    print(f"  Cycles       : {args.cycles}")
    print(f"  Total pulses : {total_pulses}, estimated time {est_time:.1f} s")
    print(f"  Compliance   : read {args.icc:.3e} A, pot {args.icc_pot:.3e} A, dep {args.icc_dep:.3e} A")
    print(f"  Pulse mode   : {args.pulse_mode}")
    if args.r_series:
        print(f"  Series R     : {args.r_series/1e3:g} kOhm (subtracted from R; device voltage logged)")
    print(f"  Output       : {run_dir}")
    print("=" * 68)
    print("  Note: hardware mode requires Keithley 2461.")
    print("=" * 68)
    ans = input("\nConfirm start? (Enter / q to quit): ").strip().lower()
    if ans == "q":
        return

    records = []
    global_pulse = 0
    inst = connect(args.address)
    t_zero = time.time()
    try:
        init_smu(inst, args.icc, args.nplc)
        current, resistance, conductance = read_resistance(
            inst, args.v_read, args.icc, args.n_read_avg, args.read_settle, args.r_series
        )
        add_record(records, 0, "baseline", 0, global_pulse, time.time() - t_zero, 0.0, current, resistance, conductance)
        print(f"\n[Baseline] R={fmt_resistance(resistance)}  G={conductance*1e6:.3f} uS")

        for cycle in range(1, args.cycles + 1):
            global_pulse, stopped = run_phase(
                inst, args, records, cycle, "potentiation", args.v_pot, args.n_pot, global_pulse, t_zero, args.icc_pot, args.t_pot
            )
            if stopped:
                break
            global_pulse, stopped = run_phase(
                inst, args, records, cycle, "depression", args.v_dep, args.n_dep, global_pulse, t_zero, args.icc_dep, args.t_dep
            )
            if stopped:
                break
    except KeyboardInterrupt:
        print("\n[Stop] Ctrl+C received, saving collected data.")
    finally:
        safe_shutdown(inst)
        print("[Safe] Output OFF, voltage returned to 0 V.")
        inst.close()

    save_csv(records, csv_path)
    save_metadata(args, meta_path)

    finite_r = np.array([r["resistance_Ohm"] for r in records if np.isfinite(r["resistance_Ohm"])], dtype=float)
    finite_g = np.array([r["conductance_S"] for r in records if np.isfinite(r["conductance_S"])], dtype=float)
    if len(finite_r):
        print(f"[Summary] R range: {fmt_resistance(float(np.min(finite_r)))} to {fmt_resistance(float(np.max(finite_r)))}")
        if np.min(finite_r) > 0:
            print(f"[Summary] Rmax/Rmin: {float(np.max(finite_r) / np.min(finite_r)):.3f}x")
    if len(finite_g) > 1:
        print(f"[Summary] G span: {(float(np.max(finite_g) - np.min(finite_g)) * 1e6):.3f} uS")

    if not args.no_plot:
        title = f"{args.dev} pulse modulation: POT {args.v_pot:+.2f} V, DEP {args.v_dep:+.2f} V"
        plot_records(records, img_path, title)


if __name__ == "__main__":
    main()

