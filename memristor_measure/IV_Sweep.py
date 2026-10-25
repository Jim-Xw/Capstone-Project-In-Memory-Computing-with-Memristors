"""Bipolar I-V cycling of a memristor, saving the evolution of the resistance state.

Wiring: Keithley FORCE HI to the top electrode and FORCE LO to the bottom electrode.
The SET polarity is set by ``--set-polarity`` or in the parameter section of the file;
never rewire while the output is on. Parameter/command example:
``python IV_Sweep.py --v-max 1.5 --v-min -1.5 --icc-set 50e-6 --icc-reset 50e-6``.
If the default ±2 V is exceeded, an extra warning line is shown before the usual
Enter/q confirmation. Output: full CSV, per-cycle CSVs and I-V/resistance PNGs; the
original CSV columns keep their order, and the set/measured voltage, compliance, source
status and stop reason are appended. Safety warning: the sweep voltage is limited to
±2 V by default and both the SET and RESET Icc are hard-limited to 50 uA; compliance
points are still sampled but their resistance is recorded as nan. The run stops and
returns safely to zero only on an abnormal overcurrent, an invalid reading or a
communication error. Software protection does not replace correct FORCE wiring and
supervision, and cannot guarantee that a device will never break down.
``--r-series <Ohm>``: series resistor on the HI lead; resistance and plots are corrected
for it, the device voltage is recorded and the output name carries an RS tag.
"""

import os, sys, time, argparse
from datetime import datetime

import matplotlib
import platform
_fonts = {"Windows": "Microsoft YaHei", "Darwin": "PingFang SC", "Linux": "WenQuanYi Micro Hei"}
matplotlib.rcParams["font.family"]        = _fonts.get(platform.system(), "DejaVu Sans")
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams.update({
    "axes.labelsize": 12, "axes.titlesize": 12, "axes.linewidth": 1.2,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "xtick.minor.width": 0.9, "ytick.minor.width": 0.9,
})
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.ticker as ticker
import numpy as np

try:
    from .keithley_smu import (
        AUTO_ADDRESS,
        DEFAULT_MAX_VOLTAGE_V,
        InstrumentConnectionError,
        connect_smu,
        configure_voltage_source,
        read_vi,
        safe_shutdown,
        set_current_limit as smu_set_current_limit,
        validate_safety_limits,
        add_series_argument,
        validate_series,
        series_tag,
        device_voltage,
        device_resistance,
    )
except ImportError:
    from keithley_smu import (
        AUTO_ADDRESS,
        DEFAULT_MAX_VOLTAGE_V,
        InstrumentConnectionError,
        connect_smu,
        configure_voltage_source,
        read_vi,
        safe_shutdown,
        set_current_limit as smu_set_current_limit,
        validate_safety_limits,
        add_series_argument,
        validate_series,
        series_tag,
        device_voltage,
        device_resistance,
    )

# ══════════════════════════════════════════════════════
#  ★ Adjustable parameters ★
# ══════════════════════════════════════════════════════
DEFAULT_VISA_ADDRESS = AUTO_ADDRESS

V_MAX    =  1.8 # positive sweep limit (V)
V_MIN    = -1.8  # negative sweep limit (V)
N_POINTS =  50    # points per quarter segment (full cycle = N_POINTS × 4)
N_CYCLES =  1     # number of sweep cycles
NPLC     =  0.5    # integration time

# ── Bipolar safety compliance ─────────────────────────
#   SET and RESET are both protected by a 50 uA hard limit
#   SET_POLARITY selects which polarity is the SET direction:
#     "positive" = positive SET / negative RESET (default, matches the +1.8 V SET plot)
#     "negative" = negative SET / positive RESET (change here if the polarity is reversed)
SET_POLARITY = "positive"

I_CC_SET   = 50e-6  # SET-direction compliance current, hard limit 50 uA
I_CC_RESET = 50e-6  # RESET-direction compliance current, also hard-limited to 50 uA
OVERCURRENT_ABORT_RATIO = 1.02  # >2% above the present Icc is abnormal: shut down at once
R_SERIES = 0.0  # series resistor on the HI lead (Ohm), 0 = none; overridden by --r-series
# ══════════════════════════════════════════════════════

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IV_sweep")
CYCLE_PALETTE = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]


def _series_title() -> str:
    """Series-resistor label for plot titles; empty when R_SERIES=0."""
    return f"   Rs={R_SERIES/1e3:g} kOhm" if R_SERIES > 0 else ""


def _voltage_label() -> str:
    return "Device Voltage (V)" if R_SERIES > 0 else "Voltage (V)"


# ──────────────────────────────────────────────────────
#  Return the I_cc for this step from the voltage sign and the polarity setting
# ──────────────────────────────────────────────────────
def icc_for_voltage(v: float) -> float:
    """
    Positive-SET mode: V>=0 uses I_CC_SET, V<0 uses I_CC_RESET
    Negative-SET mode: the reverse
    V=0 uses the SET limit by default (more conservative)
    """
    if SET_POLARITY == "positive":
        return I_CC_SET if v >= 0 else I_CC_RESET
    else:  # negative
        return I_CC_SET if v <= 0 else I_CC_RESET


# ──────────────────────────────────────────────────────
def connect(address):
    try:
        return connect_smu(address, timeout_ms=10000)
    except InstrumentConnectionError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


def set_current_limit(inst, requested_icc: float) -> float:
    """Set Icc through the shared safety layer and read it back."""
    return smu_set_current_limit(inst, requested_icc)


def init_smu(inst, i_cc_init, nplc):
    actual_icc = configure_voltage_source(inst, i_cc_init, nplc)
    time.sleep(0.3)
    print("[INIT] Done, no errors")
    return actual_icc


def build_cycle(v_max, v_min, n):
    s1 = np.linspace(0,     v_max, n, endpoint=False)
    s2 = np.linspace(v_max, 0,     n, endpoint=False)
    s3 = np.linspace(0,     v_min, n, endpoint=False)
    s4 = np.linspace(v_min, 0,     n, endpoint=True)
    return np.concatenate([s1, s2, s3, s4])


def compute_resistance_with_icc_hold(voltages, currents, icc_used,
                                     trigger_ratio=0.985, r_series=0.0):
    """
    Resistance from the actual source voltage; nan in the compliance region. With
    r_series>0 the series resistor is subtracted (device resistance).

    The original function name is kept for existing imports, but the compliance region
    is no longer masked with a boundary resistance.
    """
    voltages = np.asarray(voltages, dtype=float)
    currents = np.asarray(currents, dtype=float)
    icc_used = np.asarray(icc_used, dtype=float)
    resistance = np.full(len(voltages), np.nan, dtype=float)

    for k, (v, i, icc) in enumerate(zip(voltages, currents, icc_used)):
        if not (np.isfinite(v) and np.isfinite(i)) or abs(i) <= 1e-12:
            continue

        at_compliance = np.isfinite(icc) and icc > 0 and abs(i) >= abs(icc) * trigger_ratio
        if not at_compliance:
            resistance[k] = device_resistance(v, i, r_series)

    return resistance


# ──────────────────────────────────────────────────────
def run_sweep(inst, v_max, v_min, n_points, n_cycles, nplc):
    """
    Point-by-point stepping in Python with dynamic asymmetric compliance:
        I_cc switched at each step by voltage sign (limited for SET / relaxed for RESET)
    """
    one_cycle = build_cycle(v_max, v_min, n_points)
    cycle_len = len(one_cycle)
    total     = cycle_len * n_cycles

    step_delay = max(0.02, nplc / 60 * 1.5 + 0.02)
    est_s = total * step_delay
    print(f"[SWEEP] {n_cycles} cycles × {cycle_len} points = {total} points")
    print(f"[SWEEP] Step delay {step_delay*1000:.0f} ms/point, estimated {est_s:.0f} s")
    print(f"[SWEEP] Safety compliance: SET={I_CC_SET*1e6:.0f}uA  "
          f"RESET={I_CC_RESET*1e6:.0f}uA  (SET polarity={SET_POLARITY})")

    voltages   = np.zeros(total)
    currents   = np.zeros(total)
    timestamps = np.zeros(total)
    icc_used   = np.zeros(total)   # compliance actually used at each point
    actual_voltages = np.full(total, np.nan)
    compliance = np.zeros(total, dtype=bool)
    source_status = np.full(total, np.nan)
    stopped_reasons = [""] * total
    collected = 0

    inst.write(":OUTP ON")
    t_start = time.time()

    # Track the present I_cc; send a command only when it changes (less communication)
    cur_icc = None
    compliance_count = 0

    print(f"\n  {'Step':>6}  {'V (V)':>9}  {'I (uA)':>11}  {'R (kΩ)':>10}  {'I_cc':>8}")
    print("  " + "─" * 56)

    for idx in range(total):
        v_set = one_cycle[idx % cycle_len]

        # ── Switch the compliance dynamically ─────────────
        want_icc = icc_for_voltage(v_set)
        if cur_icc is None or abs(want_icc - cur_icc) > 1e-12:
            cur_icc = set_current_limit(inst, want_icc)

        inst.write(f":SOUR:VOLT:LEV {v_set:.6f}")
        time.sleep(step_delay)

        try:
            reading = read_vi(inst, v_set, cur_icc)
            i = reading.current_a
        except Exception as e:
            print(f"  [READ ERROR #{idx}] {e}")
            stopped_reasons[idx] = f"read_error: {e}"
            safe_shutdown(inst)
            break

        voltages[idx]   = v_set
        currents[idx]   = i
        timestamps[idx] = time.time() - t_start
        icc_used[idx]   = cur_icc
        actual_voltages[idx] = reading.voltage_actual_v
        compliance[idx] = reading.compliance
        source_status[idx] = (
            np.nan if reading.source_status is None else reading.source_status
        )
        collected = idx + 1
        if not (np.isfinite(i) and np.isfinite(reading.voltage_actual_v)):
            stopped_reasons[idx] = "non_finite_reading"
            print("  [SAFETY STOP] Invalid V/I reading detected; returning to zero.")
            safe_shutdown(inst)
            break

        if abs(i) > cur_icc * OVERCURRENT_ABORT_RATIO:
            stopped_reasons[idx] = "overcurrent"
            print(
                f"  [SAFETY STOP] Measured current {abs(i)*1e6:.3f} uA exceeds the "
                f"Icc tolerance limit of {cur_icc*OVERCURRENT_ABORT_RATIO*1e6:.3f} uA."
            )
            safe_shutdown(inst)
            break

        if reading.compliance:
            compliance_count += 1
            if compliance_count == 1:
                print(
                    "  [COMPLIANCE] Compliance reached; holding the 50 uA hard limit and continuing, "
                    "with the resistance in this region recorded as nan."
                )

        report_every = max(1, cycle_len // 4)
        if idx % report_every == 0 or idx == total - 1:
            r_str = (
                "limited" if reading.compliance
                else (
                    f"{device_resistance(reading.voltage_actual_v, i, R_SERIES)/1e3:.2f}"
                    if abs(i) > 1e-11 else "inf"
                )
            )
            pct   = (idx + 1) / total * 100
            icc_s = (f"{cur_icc*1e3:.1f}mA" if cur_icc >= 1e-3
                     else f"{cur_icc*1e6:.0f}uA")
            print(f"  {pct:5.1f}%  {v_set:>9.4f}  {i*1e6:>11.4f}  "
                  f"{r_str:>10}  {icc_s:>8}")
    safe_shutdown(inst)

    elapsed = time.time() - t_start
    print(f"\n[SWEEP] Complete, elapsed time {elapsed:.1f} s")
    print(f"[SWEEP] Compliance points: {compliance_count}/{collected}")
    sl = slice(0, collected)
    return (voltages[sl], currents[sl], timestamps[sl], cycle_len,
            icc_used[sl], actual_voltages[sl], compliance[sl],
            source_status[sl], stopped_reasons[:collected])


# ──────────────────────────────────────────────────────
def save_csv(voltages, currents, timestamps, resistance, icc_used,
             cycle_len, path, output_dir=None, ts_str=None,
             actual_voltages=None, compliance=None, source_status=None,
             stopped_reasons=None):
    import csv
    n = len(voltages)
    n_cycles = n // cycle_len

    header = ["index", "cycle", "index_in_cycle", "time_s",
              "voltage_V", "current_A", "current_uA",
              "resistance_Ohm", "icc_A",
              "voltage_set_V", "voltage_actual_V", "compliance",
              "source_status", "stopped_reason",
              "device_voltage_V", "r_series_Ohm"]

    def row(k):
        r = float("nan") if compliance[k] else resistance[k]
        return [k, k // cycle_len, k % cycle_len,
                f"{timestamps[k]:.4f}",
                f"{voltages[k]:.6f}",
                f"{currents[k]:.6e}",
                f"{currents[k]*1e6:.4f}",
                f"{r:.2f}" if np.isfinite(r) else "nan",
                f"{icc_used[k]:.6e}",
                f"{voltages[k]:.6f}",
                f"{actual_voltages[k]:.6f}",
                int(compliance[k]),
                "" if np.isnan(source_status[k]) else int(source_status[k]),
                stopped_reasons[k],
                f"{device_voltage(actual_voltages[k], currents[k], R_SERIES):.6f}",
                f"{R_SERIES:g}"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for k in range(n):
            w.writerow(row(k))
    print(f"[CSV] Full    -> {path}")

    if output_dir and ts_str:
        for cyc in range(n_cycles):
            lo  = cyc * cycle_len
            hi  = (cyc + 1) * cycle_len
            cyc_path = os.path.join(output_dir,
                                    f"iv_cycle{cyc+1}_{ts_str}.csv")
            with open(cyc_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header)
                for k in range(lo, hi):
                    w.writerow(row(k))
            print(f"[CSV] Cycle {cyc+1} -> {cyc_path}")


# ──────────────────────────────────────────────────────
def _icc_reference_lines(ax):
    """Draw the asymmetric compliance reference lines on the I-V plot (different +/- values)."""
    if SET_POLARITY == "positive":
        icc_pos, icc_neg = I_CC_SET, I_CC_RESET
    else:
        icc_pos, icc_neg = I_CC_RESET, I_CC_SET

    ax.axhline( icc_pos*1e6, color="#FF5722", lw=0.9, linestyle="--",
               alpha=0.7,
               label=f"+I_cc = {icc_pos*1e6:.0f} uA" if icc_pos < 1e-3
                     else f"+I_cc = {icc_pos*1e3:.1f} mA")
    ax.axhline(-icc_neg*1e6, color="#FF9800", lw=0.9, linestyle="--",
               alpha=0.7,
               label=f"-I_cc = {icc_neg*1e6:.0f} uA" if icc_neg < 1e-3
                     else f"-I_cc = {icc_neg*1e3:.1f} mA")


# ──────────────────────────────────────────────────────
def plot_individual_cycles(voltages, currents, resistance,
                           n_cycles, cycle_len,
                           v_max, v_min, output_dir, ts_str):
    v_all = voltages
    i_all = currents * 1e6
    r_all = resistance / 1e3

    for cyc in range(n_cycles):
        lo  = cyc * cycle_len
        hi  = (cyc + 1) * cycle_len

        v_c = v_all[lo:hi]
        i_c = i_all[lo:hi]
        r_c = r_all[lo:hi]
        x_c = np.arange(cycle_len)
        col = CYCLE_PALETTE[cyc % len(CYCLE_PALETTE)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("#000000")
        for ax in (ax1, ax2):
            ax.set_facecolor("#000000")
            ax.tick_params(which="both", colors="#FFFFFF", labelsize=11)
            for sp in ax.spines.values(): sp.set_color("#FFFFFF")

        seg_labels = [
            (0,             cycle_len//4,   "#69F0AE", "0 → V_max"),
            (cycle_len//4,  cycle_len//2,   "#FFD54F", "V_max → 0"),
            (cycle_len//2,  3*cycle_len//4, "#80DEEA", "0 → V_min"),
            (3*cycle_len//4, cycle_len,     "#EF9A9A", "V_min → 0"),
        ]
        for lo2, hi2, seg_col, seg_label in seg_labels:
            ax1.plot(v_c[lo2:hi2], i_c[lo2:hi2],
                     color=seg_col, lw=2.0, alpha=0.9,
                     label=seg_label, zorder=3)

        for lo2, hi2, seg_col, _ in seg_labels:
            mid = (lo2 + hi2) // 2
            if mid + 1 >= cycle_len: continue
            dv = v_c[mid+1] - v_c[mid]
            di = i_c[mid+1] - i_c[mid]
            if abs(dv) + abs(di) < 1e-9: continue
            ax1.annotate("", xy=(v_c[mid+1], i_c[mid+1]),
                xytext=(v_c[mid], i_c[mid]),
                arrowprops=dict(arrowstyle="-|>", color=seg_col,
                                lw=1.0, mutation_scale=10))

        ax1.scatter([v_c[0]], [i_c[0]], color="#FFFFFF", s=70,
                    zorder=6, marker="o", edgecolors=col, linewidths=1.5,
                    label="Start")

        _icc_reference_lines(ax1)
        ax1.axhline(0, color="#888888", lw=0.5)
        ax1.axvline(0, color="#888888", lw=0.5)

        vp = max((v_c.max()-v_c.min())*0.08, 0.2)
        ip = max((i_c.max()-i_c.min())*0.08, 10.0)
        ax1.set_xlim(v_c.min()-vp, v_c.max()+vp)
        ax1.set_ylim(i_c.min()-ip, i_c.max()+ip)
        ax1.set_xlabel(_voltage_label(),  color="#FFFFFF", fontsize=12)
        ax1.set_ylabel("Current (uA)", color="#FFFFFF", fontsize=12)
        ax1.set_title(f"I–V  |  Cycle {cyc+1}",
                      color="#FFFFFF", fontsize=11, pad=8)
        ax1.legend(facecolor="#000000", edgecolor="#FFFFFF",
                   labelcolor="#FFFFFF", fontsize=7.5, loc="best")
        ax1.grid(True, color="#3A3A3A", lw=0.5, zorder=0)

        fin = np.isfinite(r_c)
        ax2.scatter(x_c[fin], r_c[fin], c=col, s=6, alpha=0.75, zorder=3)

        win = max(5, int(cycle_len * 0.08)); half = win // 2
        sm = np.full(cycle_len, np.nan)
        for k in range(cycle_len):
            ch = r_c[max(0, k-half):min(cycle_len, k+half+1)]
            vc2 = ch[np.isfinite(ch)]
            if len(vc2): sm[k] = np.nanmedian(vc2)
        ax2.plot(x_c, sm, color=col, lw=2.0, alpha=0.9, label="Median smooth")

        for sep in [cycle_len//4, cycle_len//2, 3*cycle_len//4]:
            ax2.axvline(sep, color="#888888", lw=0.7, linestyle=":", alpha=0.7)

        fin_r = r_c[fin]
        if len(fin_r) >= 5:
            rlo = np.percentile(fin_r, 5)
            rhi = np.percentile(fin_r, 95)
            ax2.axhline(rlo, color="#4DB6AC", lw=1.0, linestyle=":", alpha=0.9,
                        label=f"P5  ≈ {rlo:.1f} kΩ")
            ax2.axhline(rhi, color="#FFB74D", lw=1.0, linestyle=":", alpha=0.9,
                        label=f"P95 ≈ {rhi:.1f} kΩ")
            if rlo > 0:
                ax2.text(0.97, 0.04, f"P95/P5 ≈ {rhi/rlo:.1f}×",
                         transform=ax2.transAxes, color="#CE93D8",
                         fontsize=9, ha="right",
                         bbox=dict(boxstyle="round,pad=0.3",
                                   fc="#000000", ec="#9C27B0", alpha=0.85))

        ax2.set_yscale("log")
        ax2.yaxis.set_major_formatter(ticker.LogFormatter(minor_thresholds=(2, 0.5)))
        ax2.yaxis.set_minor_formatter(ticker.LogFormatter(minor_thresholds=(2, 0.5)))
        ax2.set_xlabel("Sample Index (within cycle)",
                       color="#FFFFFF", fontsize=12)
        ax2.set_ylabel("Resistance (kΩ)", color="#FFFFFF", fontsize=12)
        ax2.set_title(f"Resistance  |  Cycle {cyc+1}",
                      color="#FFFFFF", fontsize=11, pad=8)
        ax2.legend(facecolor="#000000", edgecolor="#FFFFFF",
                   labelcolor="#FFFFFF", fontsize=8, loc="upper right")
        ax2.grid(True, color="#3A3A3A", lw=0.5, which="both", zorder=0)

        icc_str = (f"SET={I_CC_SET*1e6:.0f}uA / "
                   f"RESET={I_CC_RESET*1e6:.0f}uA")
        fig.suptitle(
            f"Ni/HfOx/Ti   Cycle {cyc+1}/{n_cycles}   "
            f"V: {v_min}~{v_max} V   {cycle_len} pts   {icc_str}"
            f"{_series_title()}",
            color="#FFFFFF", fontsize=9, y=1.01)

        plt.tight_layout()
        img_path = os.path.join(output_dir,
                                f"iv_cycle{cyc+1}_{ts_str}.png")
        plt.savefig(img_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[PNG] Cycle {cyc+1} -> {img_path}")


# ──────────────────────────────────────────────────────
def plot_and_save(voltages, currents, resistance,
                  n_cycles, cycle_len,
                  v_max, v_min, img_path):

    v   = voltages
    i   = currents * 1e6
    n   = len(v)
    idx = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#000000")
    for ax in (ax1, ax2):
        ax.set_facecolor("#000000")
        ax.tick_params(which="both", colors="#FFFFFF", labelsize=11)
        for sp in ax.spines.values(): sp.set_color("#FFFFFF")

    LINESTYLES = ["-", "--", "-.", (0,(3,1,1,1)), (0,(5,1))]
    LW         = [1.8, 1.8, 1.8, 1.6, 1.6]

    for cyc in range(n_cycles):
        lo  = cyc * cycle_len
        hi  = (cyc + 1) * cycle_len
        v_c = v[lo:hi]
        i_c = i[lo:hi]
        col = CYCLE_PALETTE[cyc % len(CYCLE_PALETTE)]
        ls  = LINESTYLES[cyc % len(LINESTYLES)]
        lw  = LW[cyc % len(LW)]

        ax1.plot(v_c, i_c, color=col, lw=lw, linestyle=ls,
                 alpha=0.85, zorder=2+cyc*0.1)
        ax1.scatter([v_c[0]], [i_c[0]], color=col, s=55, zorder=6,
                    marker="o", edgecolors="white", linewidths=0.6)

        m = len(v_c)
        for ap in np.linspace(0, m-2, 4, dtype=int):
            dv = v_c[ap+1] - v_c[ap]
            di = i_c[ap+1] - i_c[ap]
            if abs(dv) + abs(di) < 1e-9: continue
            ax1.annotate("", xy=(v_c[ap+1], i_c[ap+1]),
                xytext=(v_c[ap], i_c[ap]),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=0.8, mutation_scale=9))

        ax1.plot([], [], color=col, lw=lw, linestyle=ls,
                 label=f"Cycle {cyc+1}")

    _icc_reference_lines(ax1)
    ax1.axhline(0, color="#888888", lw=0.5)
    ax1.axvline(0, color="#888888", lw=0.5)

    vp = max((v.max()-v.min())*0.08, 0.2)
    ip = max((i.max()-i.min())*0.08, 10.0)
    ax1.set_xlim(v.min()-vp, v.max()+vp)
    ax1.set_ylim(i.min()-ip, i.max()+ip)
    ax1.set_xlabel(_voltage_label(),   color="#FFFFFF", fontsize=12)
    ax1.set_ylabel("Current (uA)",  color="#FFFFFF", fontsize=12)
    ax1.set_title("I–V Characteristics", color="#FFFFFF", fontsize=11, pad=8)
    ax1.legend(facecolor="#000000", edgecolor="#FFFFFF",
               labelcolor="#FFFFFF", fontsize=8.5, loc="best")
    ax1.grid(True, color="#3A3A3A", lw=0.5, zorder=0)

    r_k = resistance / 1e3
    for cyc in range(n_cycles):
        lo   = cyc * cycle_len
        hi   = (cyc + 1) * cycle_len
        r_c  = r_k[lo:hi]
        x_c  = idx[lo:hi]
        col  = CYCLE_PALETTE[cyc % len(CYCLE_PALETTE)]
        fin  = np.isfinite(r_c)
        ax2.scatter(x_c[fin], r_c[fin], c=col, s=5, alpha=0.7, zorder=3)

        win = max(5, int(cycle_len * 0.08))
        half = win // 2
        sm = np.full(cycle_len, np.nan)
        for k in range(cycle_len):
            ch = r_c[max(0, k-half):min(cycle_len, k+half+1)]
            vc = ch[np.isfinite(ch)]
            if len(vc): sm[k] = np.nanmedian(vc)
        ax2.plot(x_c, sm, color=col, lw=1.5, alpha=0.9, label=f"Cycle {cyc+1}")

    for cyc in range(1, n_cycles):
        ax2.axvline(cyc*cycle_len, color="#888888", lw=0.8,
                    linestyle="--", alpha=0.6)

    fin_all = r_k[np.isfinite(r_k)]
    if len(fin_all) >= 5:
        rlo = np.percentile(fin_all, 5)
        rhi = np.percentile(fin_all, 95)
        ax2.axhline(rlo, color="#4DB6AC", lw=1.0, linestyle=":", alpha=0.9,
                    label=f"P5  ≈ {rlo:.1f} kΩ")
        ax2.axhline(rhi, color="#FFB74D", lw=1.0, linestyle=":", alpha=0.9,
                    label=f"P95 ≈ {rhi:.1f} kΩ")
        if rlo > 0:
            ax2.text(0.97, 0.04, f"P95/P5 ≈ {rhi/rlo:.1f}×",
                     transform=ax2.transAxes, color="#CE93D8",
                     fontsize=9.5, ha="right",
                     bbox=dict(boxstyle="round,pad=0.3",
                               fc="#000000", ec="#9C27B0", alpha=0.85))

    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(ticker.LogFormatter(minor_thresholds=(2, 0.5)))
    ax2.yaxis.set_minor_formatter(ticker.LogFormatter(minor_thresholds=(2, 0.5)))
    ax2.set_xlabel("Sample Index",    color="#FFFFFF", fontsize=12)
    ax2.set_ylabel("Resistance (kΩ)", color="#FFFFFF", fontsize=12)
    ax2.set_title("Resistance vs. Sample Index",
                  color="#FFFFFF", fontsize=11, pad=8)
    ax2.legend(facecolor="#000000", edgecolor="#FFFFFF",
               labelcolor="#FFFFFF", fontsize=8.5)
    ax2.grid(True, color="#3A3A3A", lw=0.5, which="both", zorder=0)

    icc_str = f"SET={I_CC_SET*1e6:.0f}uA / RESET={I_CC_RESET*1e6:.0f}uA"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.suptitle(
        f"Ni/AlOx/TiOx/Ti   V: {v_min}~{v_max} V   "
        f"{n_cycles} cycles × {cycle_len} pts   "
        f"{icc_str}   NPLC={NPLC}{_series_title()}   {ts}",
        color="#FFFFFF", fontsize=9, y=1.01)

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[PNG] Saved: {img_path}")
    plt.show()


# ──────────────────────────────────────────────────────
def main():
    global V_MAX, V_MIN, N_POINTS, N_CYCLES, NPLC
    global I_CC_SET, I_CC_RESET, SET_POLARITY, R_SERIES

    parser = argparse.ArgumentParser(
        description="Keithley 2450/2461 safe memristor I-V sweep"
    )
    parser.add_argument("--address", default=DEFAULT_VISA_ADDRESS,
                        help="VISA resource address, default auto")
    parser.add_argument("--v-max", type=float, default=V_MAX,
                        help="positive sweep limit in V; default V_MAX from the parameter section")
    parser.add_argument("--v-min", type=float, default=V_MIN,
                        help="negative sweep limit in V; default V_MIN from the parameter section")
    parser.add_argument("--points", type=int, default=N_POINTS,
                        help="points per quarter sweep segment")
    parser.add_argument("--cycles", type=int, default=N_CYCLES,
                        help="number of sweep cycles")
    parser.add_argument("--nplc", type=float, default=NPLC,
                        help="current-measurement integration time NPLC")
    parser.add_argument("--icc-set", type=float, default=I_CC_SET,
                        help="SET-direction compliance current in A, hard limit 50e-6")
    parser.add_argument("--icc-reset", type=float, default=I_CC_RESET,
                        help="RESET-direction compliance current in A, hard limit 50e-6")
    parser.add_argument("--set-polarity", choices=("positive", "negative"),
                        default=SET_POLARITY,
                        help="positive: positive-voltage SET; negative: negative-voltage SET")
    add_series_argument(parser)
    args = parser.parse_args()
    validate_series(args.r_series)

    V_MAX = args.v_max
    V_MIN = args.v_min
    N_POINTS = args.points
    N_CYCLES = args.cycles
    NPLC = args.nplc
    I_CC_SET = args.icc_set
    I_CC_RESET = args.icc_reset
    SET_POLARITY = args.set_polarity
    R_SERIES = args.r_series

    if N_POINTS <= 0 or N_CYCLES <= 0:
        raise ValueError("points and cycles must be positive integers")
    if NPLC <= 0:
        raise ValueError("NPLC must be greater than 0")
    if V_MAX <= 0 or V_MIN >= 0:
        raise ValueError("Use a positive --v-max and a negative --v-min")
    if I_CC_SET <= 0 or I_CC_RESET <= 0:
        raise ValueError("SET/RESET Icc must be greater than 0")

    overvoltage = (
        abs(V_MAX) > DEFAULT_MAX_VOLTAGE_V
        or abs(V_MIN) > DEFAULT_MAX_VOLTAGE_V
    )
    validate_safety_limits(
        V_MAX, I_CC_SET, allow_overvoltage=overvoltage
    )
    validate_safety_limits(
        V_MIN, I_CC_RESET, allow_overvoltage=overvoltage
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # The series-resistor tag goes before the timestamp; file names are unchanged when R_SERIES=0
    rs_tag = series_tag(R_SERIES)
    name_ts = f"{rs_tag}_{ts_str}" if rs_tag else ts_str

    # File name reflects the asymmetric compliance
    run_name = (
        f"V{V_MIN}_{V_MAX}_"
        f"cyc{N_CYCLES}_"
        f"pts{N_POINTS}_"
        f"SET{int(I_CC_SET*1e6)}uA_"
        f"RST{I_CC_RESET*1e6:.0f}uA_"
        f"NPLC{NPLC}_"
        f"{name_ts}"
    )
    RUN_DIR  = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"[OUTPUT] {RUN_DIR}")

    csv_path = os.path.join(RUN_DIR, f"iv_sweep_{name_ts}.csv")
    img_path = os.path.join(RUN_DIR, f"iv_sweep_{name_ts}.png")

    cycle_len  = N_POINTS * 4
    total      = cycle_len * N_CYCLES
    step_delay = max(0.02, NPLC / 60 * 1.5 + 0.02)

    print("=" * 56)
    print("  IV sweep parameters (bipolar, 50 uA safety compliance)")
    print("=" * 56)
    print(f"  V_MAX        = {V_MAX} V")
    print(f"  V_MIN        = {V_MIN} V")
    print(f"  N_POINTS     = {N_POINTS}  (per quarter segment)")
    print(f"  N_CYCLES     = {N_CYCLES}")
    print(f"  SET polarity = {SET_POLARITY}")
    print(f"  I_CC_SET     = {I_CC_SET*1e6:.0f} uA   (limit, controls the filament)")
    print(f"  I_CC_RESET   = {I_CC_RESET*1e6:.0f} uA   (hard safety limit)")
    print(f"  NPLC         = {NPLC}")
    if R_SERIES > 0:
        print(f"  Series R = {R_SERIES/1e3:g} kOhm "
              "(subtracted from R; device voltage logged)")
    print(f"  Total points = {total}")
    print(f"  Est. time    ≈ {total*step_delay:.0f} s")
    print(f"  Output dir   = {RUN_DIR}")
    print("=" * 56)
    if SET_POLARITY == "positive":
        print("  Polarity: positive segment SET / negative segment RESET")
    else:
        print("  Polarity: negative segment SET / positive segment RESET")
    print("  (!) SET/RESET both hard-limited to 50 uA; the sweep continues at compliance")
    if overvoltage:
        print(
            f"  [WARNING] Sweep range exceeds the default ±{DEFAULT_MAX_VOLTAGE_V:g} V "
            "and may permanently alter or break down the device"
        )
    print("=" * 56)

    ans = input("\nStart the sweep? (Enter / q to quit): ").strip().lower()
    if ans == "q":
        return

    inst = connect(args.address)
    try:
        init_smu(inst, I_CC_SET, NPLC)   # start with the SET limit (conservative)
        (voltages, currents, timestamps, cycle_len, icc_used,
         actual_voltages, compliance, source_status,
         stopped_reasons) = run_sweep(
            inst, V_MAX, V_MIN, N_POINTS, N_CYCLES, NPLC)
    finally:
        safe_shutdown(inst)
        inst.close()

    resistance = compute_resistance_with_icc_hold(
        actual_voltages, currents, icc_used, r_series=R_SERIES
    )
    resistance[compliance] = np.nan

    fin = resistance[np.isfinite(resistance)]
    print(f"\n  Points     : {len(voltages)}")
    print(f"  Set range  : {voltages.min():.4f} ~ {voltages.max():.4f} V")
    print(
        f"  Meas. range: {np.nanmin(actual_voltages):.4f} ~ "
        f"{np.nanmax(actual_voltages):.4f} V"
    )
    print(f"  Peak I     : {np.abs(currents).max()*1e6:.3f} uA")
    if len(fin):
        print(f"  R range    : {fin.min()/1e3:.3f} ~ {fin.max()/1e3:.3f} kΩ")

    # ── RESET diagnosis: check for a sudden current drop in the RESET direction ──
    print("\n" + "─" * 56)
    print("  RESET diagnosis")
    print("─" * 56)
    one_len = cycle_len
    v_last = voltages[-one_len:]
    i_last = currents[-one_len:]
    if SET_POLARITY == "positive":
        # RESET is on the negative segment
        reset_mask = v_last < -0.1
    else:
        reset_mask = v_last > 0.1

    if reset_mask.sum() > 5:
        v_r = v_last[reset_mask]
        i_r = np.abs(i_last[reset_mask])
        # Key point: check whether the current falls at the same voltage magnitude, which
        # excludes the natural decrease on the way back to zero.
        # A true RESET: the current suddenly falls at a fixed high voltage (R jumps up)
        # A resistance criterion is more reliable: does R jump clearly in the RESET segment?
        with np.errstate(divide="ignore", invalid="ignore"):
            r_r = np.where(i_r > 1e-11, np.abs(v_r) / i_r - R_SERIES, np.nan)
        r_r_fin = r_r[np.isfinite(r_r)]
        if len(r_r_fin) >= 5:
            r_lo = np.percentile(r_r_fin, 10)   # low resistance (before RESET)
            r_hi = np.percentile(r_r_fin, 90)   # high resistance (after RESET)
            ratio = r_hi / r_lo if r_lo > 0 else 1.0
            print(f"  RESET-segment R: low {r_lo/1e3:.2f}kΩ → high {r_hi/1e3:.2f}kΩ  "
                  f"(ratio {ratio:.2f}×)")
            if ratio >= 3.0:
                print("  [OK] RESET-segment R jumps clearly (>3×) → a true RESET may be present!")
                print("       → check the plot for a sudden change; if so, the limit masked RESET")
            elif ratio >= 1.5:
                print("  (~) RESET-segment R rises slightly (1.5-3×) → partial/incomplete RESET")
                print("      → try a larger RESET current or a more negative voltage")
            else:
                print("  (!) RESET-segment R almost unchanged (<1.5×) → RESET failed")
                print("      → still no RESET with the limit relaxed: Ti oxygen gettering confirmed")
                print("      → use a Pt bottom electrode or add an AlOx barrier on Ti (see notes)")
        else:
            print("  [WARNING] Too few valid resistance points in the RESET segment to decide")
    else:
        print("  [WARNING] Too few samples in the RESET segment")
    print("─" * 56)

    save_csv(voltages, currents, timestamps, resistance, icc_used,
             cycle_len, csv_path, RUN_DIR, name_ts,
             actual_voltages, compliance, source_status, stopped_reasons)

    completed_cycles = len(voltages) // cycle_len
    # With R_SERIES>0 the x axis is the device voltage (source voltage minus the Rs drop)
    plot_voltages = (device_voltage(actual_voltages, currents, R_SERIES)
                     if R_SERIES > 0 else actual_voltages)
    if completed_cycles:
        print("\n[PLOT] Generating plots of the completed cycles...")
        plot_individual_cycles(plot_voltages[:completed_cycles * cycle_len],
                               currents[:completed_cycles * cycle_len],
                               resistance[:completed_cycles * cycle_len],
                               completed_cycles, cycle_len,
                               V_MAX, V_MIN, RUN_DIR, name_ts)
        print("[PLOT] Generating the summary plot...")
        plot_and_save(plot_voltages[:completed_cycles * cycle_len],
                      currents[:completed_cycles * cycle_len],
                      resistance[:completed_cycles * cycle_len],
                      completed_cycles, cycle_len, V_MAX, V_MIN, img_path)
    else:
        print("[PLOT] No full cycle completed before the safety stop; skipping cycle plots.")


if __name__ == "__main__":
    main()
