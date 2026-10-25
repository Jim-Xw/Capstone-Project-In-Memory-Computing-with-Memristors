"""Progressive electroforming of a memristor with current-compliance protection.

Wiring: Keithley FORCE HI to the top electrode and FORCE LO to the bottom electrode;
confirm the forward forming polarity before applying power, and never rewire while the
output is on. Parameter/command example: ``python forming.py --v-max 2 --dev DEV-01``.
Above 2 V, ``--allow-overvoltage`` must also be given and the program asks for a typed
confirmation. Output: a CSV of each forming stage with an LRS summary, and a forming
I-V/resistance PNG; the original CSV columns keep their order, and the set/measured
voltage, Icc, compliance, source status and stop reason are appended.
Safety warning: default V_MAX = 2 V and Icc hard limit 50 uA; on compliance the
resistance is written as nan and the run stops immediately and returns safely to zero.
High-voltage forming can permanently damage the device and must be supervised.
Series resistor: with ``--r-series <Ohm>``, the resistance is corrected for the series
resistor, a device-voltage column is appended and the output name carries an RS tag.
"""

import os
import sys
import time
import argparse
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
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import numpy as np

try:
    from .keithley_smu import (
        AUTO_ADDRESS,
        DEFAULT_MAX_VOLTAGE_V,
        MAX_SAFE_CURRENT_A,
        InstrumentConnectionError,
        add_series_argument,
        connect_smu,
        configure_voltage_source,
        device_voltage,
        read_vi,
        safe_shutdown,
        series_tag,
        set_current_limit,
        validate_safety_limits,
        validate_series,
    )
except ImportError:
    from keithley_smu import (
        AUTO_ADDRESS,
        DEFAULT_MAX_VOLTAGE_V,
        MAX_SAFE_CURRENT_A,
        InstrumentConnectionError,
        add_series_argument,
        connect_smu,
        configure_voltage_source,
        device_voltage,
        read_vi,
        safe_shutdown,
        series_tag,
        set_current_limit,
        validate_safety_limits,
        validate_series,
    )

# ══════════════════════════════════════════════════════
#  ★ Adjustable parameters ★
# ══════════════════════════════════════════════════════
DEFAULT_VISA_ADDRESS = AUTO_ADDRESS

V_START       =  0.0     # start voltage (V)
V_MAX         =  2.0     # default voltage limit (V)
V_STEP        =  0.05    # voltage increment per step (V)
DWELL_TIME    =  0.5     # dwell time per step (s)
NPLC          =  1.0     # integration time

# I_cc ladder: after a failed forming attempt, retry at the next level up
# Format: [(I_cc_A, description), ...]
ICC_LADDER = [
    (50e-6,  "Stage 1: 50 uA  (conservative)"),
]

# Forming criterion: triggered when the current reaches this multiple of I_cc
FORM_THRESHOLD = 0.8

# Step detection: a dI/dV above this value counts as a current step
# (a feature of the multi-interface superlattice)
STEP_DETECT_FACTOR = 5.0   # multiple of the mean dI/dV over the first 10 points

# Post-forming LRS baseline read
V_READ        =  0.1     # read voltage (V)
N_READ        =  10      # number of samples
READ_SETTLE   =  5.0     # relaxation wait after forming (s)

# Series resistor (on the HI lead, in series with the device); overridden by --r-series
R_SERIES      =  0.0     # Ohm; 0 means no series resistor
# ══════════════════════════════════════════════════════

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Forming")


# ──────────────────────────────────────────────────────
#  Connection
# ──────────────────────────────────────────────────────
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
    time.sleep(0.3)
    print(f"[INIT] I_cc = {actual_icc*1e6:.0f} uA, done")
    return actual_icc


def fmt_series(r_series: float) -> str:
    """Series-resistor label, e.g. '10 kOhm'."""
    return f"{r_series / 1e3:g} kOhm" if r_series >= 1e3 else f"{r_series:g} Ohm"


def device_voltages(res: dict, r_series: float) -> np.ndarray:
    """Device voltage: measured voltage (set value if invalid) minus the series-resistor drop."""
    v_set = np.array(res["voltages"], dtype=float)
    v_act = np.array(res["actual_voltages"], dtype=float)
    i_arr = np.array(res["currents"], dtype=float)
    v_src = np.where(np.isfinite(v_act), v_act, v_set)
    return device_voltage(v_src, i_arr, r_series)


# ──────────────────────────────────────────────────────
#  Step-detection helper
# ──────────────────────────────────────────────────────
def detect_step(voltages, currents, factor=STEP_DETECT_FACTOR) -> list:
    """
    Detect abrupt current steps in the acquired (V, I) data.
    Returns a list of step positions: [(V_step, I_step), ...]
    """
    if len(currents) < 12:
        return []

    steps = []
    i_arr = np.array(currents)
    v_arr = np.array(voltages)
    di    = np.abs(np.diff(i_arr))

    # Baseline noise: mean dI over the first 10 steps
    baseline = np.mean(di[:10]) if len(di) >= 10 else np.mean(di)
    if baseline < 1e-12:
        baseline = 1e-12

    threshold = baseline * factor
    in_step = False
    for k in range(len(di)):
        if di[k] > threshold and not in_step:
            steps.append((float(v_arr[k+1]), float(i_arr[k+1])))
            in_step = True
        elif di[k] <= threshold:
            in_step = False

    return steps


# ──────────────────────────────────────────────────────
#  Main forming loop (single I_cc stage)
# ──────────────────────────────────────────────────────
def forming_ramp(inst, i_cc: float, stage_label: str,
                 v_start, v_max, v_step, dwell) -> dict:
    """
    Staircase voltage ramp; returns a dict of forming results.
    result["formed"] = True/False
    """
    v_seq     = np.arange(v_start, v_max + v_step * 0.5, v_step)
    voltages  = []
    actual_voltages = []
    currents  = []
    compliances = []
    source_statuses = []
    stopped_reasons = []
    formed    = False
    v_forming = None
    i_forming = None

    print(f"\n{'─'*54}")
    print(f"  {stage_label}")
    print(f"  V: {v_start} → {v_max} V   step {v_step} V   dwell {dwell} s")
    print(f"{'─'*54}")
    print(f"  {'V (V)':>8}  {'I (uA)':>11}  {'I/I_cc':>8}  {'Status':>6}")
    print(f"  {'─'*40}")

    inst.write(":OUTP ON")

    for v in v_seq:
        inst.write(f":SOUR:VOLT:LEV {v:.4f}")
        time.sleep(dwell)

        reading = read_vi(inst, v, i_cc)
        i = reading.current_a
        voltages.append(v)
        actual_voltages.append(reading.voltage_actual_v)
        currents.append(i)
        compliances.append(reading.compliance)
        source_statuses.append(reading.source_status)
        stopped_reasons.append("")

        ratio = abs(i) / i_cc if i_cc > 0 else 0
        status = ""

        # ── Forming criterion ─────────────────────────────
        if reading.compliance or ratio >= FORM_THRESHOLD:
            status    = "★ FORMED"
            formed    = True
            v_forming = v
            i_forming = i
            stopped_reasons[-1] = (
                "compliance" if reading.compliance else "forming_threshold"
            )
            print(f"  {v:>8.3f}  {i*1e6:>11.4f}  {ratio:>8.3f}  {status}")
            safe_shutdown(inst)
            break

        # ── Step detection (live) ─────────────────────────
        if len(currents) >= 3:
            prev_di = abs(currents[-1] - currents[-2])
            base_di = (abs(currents[-2] - currents[-3])
                       if len(currents) >= 3 else prev_di)
            base_di = max(base_di, 1e-12)
            if prev_di > base_di * STEP_DETECT_FACTOR and prev_di > 1e-8:
                status = "↑ STEP"

        print(f"  {v:>8.3f}  {i*1e6:>11.4f}  {ratio:>8.3f}  {status}")

    # ── Return to zero immediately ────────────────────────
    safe_shutdown(inst)

    if formed:
        print(f"\n  [FORMED] V_forming = {v_forming:.3f} V  "
              f"I_forming = {i_forming*1e6:.2f} uA")
    else:
        print(f"\n  [NOT FORMED] Reached V_MAX = {v_max} V; current below the threshold")

    return {
        "formed":    formed,
        "v_forming": v_forming,
        "i_forming": i_forming,
        "voltages":  voltages,
        "actual_voltages": actual_voltages,
        "currents":  currents,
        "compliances": compliances,
        "source_statuses": source_statuses,
        "stopped_reasons": stopped_reasons,
        "i_cc":      i_cc,
        "stage":     stage_label,
    }


# ──────────────────────────────────────────────────────
#  Post-forming LRS baseline read
# ──────────────────────────────────────────────────────
def read_lrs(inst, v_read: float, n_read: int,
             settle: float, i_cc: float, r_series: float = R_SERIES) -> dict:
    print(f"\n[LRS] Waiting {settle:.0f} s for relaxation...")
    time.sleep(settle)

    # Switch to low-compliance read mode
    actual_icc = set_current_limit(inst, i_cc)
    inst.write(f":SOUR:VOLT:LEV {v_read:.4f}")
    inst.write(":OUTP ON")
    time.sleep(0.3)

    readings = []
    for _ in range(n_read):
        reading = read_vi(inst, v_read, actual_icc)
        if reading.compliance:
            print("[LRS] Compliance detected; stopping the reads and returning to zero.")
            safe_shutdown(inst)
            break
        if np.isfinite(reading.resistance_ohm):
            readings.append(reading.resistance_ohm - r_series)  # device resistance
        time.sleep(0.1)

    safe_shutdown(inst)

    if readings:
        r_mean = float(np.mean(readings))
        r_std  = float(np.std(readings))
    else:
        r_mean = float("inf")
        r_std  = 0.0

    status = ("OK ✓" if r_mean < 50e3
              else "High ⚠" if r_mean < 200e3
              else "Forming doubtful ✗")
    print(f"[LRS] R_mean = {r_mean/1e3:.2f} kΩ  "
          f"R_std = {r_std/1e3:.2f} kΩ  → {status}")

    return {"r_mean": r_mean, "r_std": r_std,
            "status": status, "v_read": v_read}


# ──────────────────────────────────────────────────────
#  Save CSV
# ──────────────────────────────────────────────────────
def save_csv(results: list, lrs: dict, path: str, r_series: float = R_SERIES):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "index", "voltage_V",
                    "current_A", "current_uA",
                    "resistance_Ohm", "i_cc_A",
                    "voltage_set_V", "voltage_actual_V", "icc_A",
                    "compliance", "source_status", "stopped_reason",
                    "device_voltage_V"])
        for res in results:
            for k, (v, i) in enumerate(zip(res["voltages"], res["currents"])):
                compliance = res["compliances"][k]
                actual_v = res["actual_voltages"][k]
                r = (abs(actual_v / i) - r_series   # device resistance
                     if not compliance and abs(i) > 1e-12 else float("nan"))
                v_dev = device_voltage(actual_v if np.isfinite(actual_v) else v, i, r_series)
                w.writerow([
                    res["stage"], k,
                    f"{v:.4f}", f"{i:.6e}", f"{i*1e6:.4f}",
                    f"{r:.2f}" if np.isfinite(r) else "nan",
                    f"{res['i_cc']:.6e}",
                    f"{v:.4f}",
                    f"{actual_v:.6f}",
                    f"{res['i_cc']:.6e}",
                    int(compliance),
                    "" if res["source_statuses"][k] is None else res["source_statuses"][k],
                    res["stopped_reasons"][k],
                    f"{v_dev:.6f}" if np.isfinite(v_dev) else "nan",
                ])

        # LRS baseline appended at the end
        w.writerow([])
        w.writerow(["# LRS baseline",
                    f"R_mean={lrs['r_mean']:.2f} Ohm",
                    f"R_std={lrs['r_std']:.2f} Ohm",
                    f"V_read={lrs['v_read']} V",
                    lrs["status"],
                    f"r_series_Ohm={r_series:g}"])

    print(f"[CSV] Saved: {path}")


# ──────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────
STAGE_COLORS = ["#2196F3", "#FF9800", "#E91E63", "#4CAF50"]


def plot_forming(results: list, lrs: dict,
                 v_max: float, img_path: str, r_series: float = R_SERIES):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#000000")
    for ax in (ax1, ax2):
        ax.set_facecolor("#000000")
        ax.tick_params(which="both", colors="#FFFFFF", labelsize=11)
        for sp in ax.spines.values(): sp.set_color("#FFFFFF")

    # ── Left panel: forming I-V curve ───────────────────
    for k, res in enumerate(results):
        col   = STAGE_COLORS[k % len(STAGE_COLORS)]
        v_arr = np.array(res["voltages"])
        if r_series > 0:
            v_arr = device_voltages(res, r_series)   # device voltage on the x axis
        i_arr = np.array(res["currents"]) * 1e6
        i_cc  = res["i_cc"]
        label = res["stage"].split(":")[0]   # "Stage 1" etc.

        ax1.plot(v_arr, i_arr, color=col, lw=1.8,
                 alpha=0.9, label=label, zorder=3)
        ax1.scatter(v_arr, i_arr, color=col, s=12,
                    alpha=0.6, zorder=4)

        # I_cc reference lines (colour matched to each stage)
        ax1.axhline(i_cc * 1e6, color=col, lw=0.8,
                    linestyle=":", alpha=0.6)
        ax1.axhline(i_cc * 1e6 * FORM_THRESHOLD,
                    color=col, lw=0.8, linestyle="--", alpha=0.5)

        # Step annotations
        steps = detect_step(v_arr if r_series > 0 else res["voltages"],
                            res["currents"])
        for n_step, (vs, ist) in enumerate(steps):
            ax1.annotate(
                f"Step {n_step+1}\n{vs:.2f} V",
                xy=(vs, ist*1e6),
                xytext=(vs + 0.1, ist*1e6 + i_cc*1e6*0.08),
                color="#FFD54F", fontsize=7.5,
                arrowprops=dict(arrowstyle="->", color="#FFD54F", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="#000000", ec="#FFD54F", alpha=0.85))

        # Forming-point annotation
        if res["formed"] and res["v_forming"] is not None:
            vf = res["v_forming"]
            if r_series > 0:
                vf = float(v_arr[len(res["currents"]) - 1])   # the forming point is the last point
            if_ = res["i_forming"] * 1e6
            ax1.scatter([vf], [if_], color="#FF4081", s=120,
                        zorder=7, marker="*",
                        edgecolors="white", linewidths=0.5)
            ax1.annotate(
                f"Formed\n{vf:.2f} V",
                xy=(vf, if_),
                xytext=(vf - 0.6, if_ - i_cc*1e6*0.15),
                color="#FF4081", fontsize=8,
                arrowprops=dict(arrowstyle="->",
                                color="#FF4081", lw=0.9),
                bbox=dict(boxstyle="round,pad=0.25",
                          fc="#000000", ec="#FF4081", alpha=0.9))

    ax1.axhline(0, color="#888888", lw=0.5)
    ax1.axvline(0, color="#888888", lw=0.5)
    ax1.set_xlim(-0.1, v_max + 0.2)
    ax1.set_xlabel("Device Voltage (V)" if r_series > 0 else "Voltage (V)",
                   color="#FFFFFF", fontsize=12)
    ax1.set_ylabel("Current (uA)", color="#FFFFFF", fontsize=12)
    ax1.set_title("Forming I–V  (staircase ramp)",
                  color="#FFFFFF", fontsize=11, pad=8)
    ax1.legend(facecolor="#000000", edgecolor="#FFFFFF",
               labelcolor="#FFFFFF", fontsize=8.5, loc="upper left")
    ax1.grid(True, color="#3A3A3A", lw=0.5, zorder=0)

    # Additional legend entries
    patch_icc  = mpatches.Patch(color="gray", linestyle=":",
                                 label=f"I_cc (per stage)")
    patch_thr  = mpatches.Patch(color="gray", linestyle="--",
                                 label=f"0.8 × I_cc (forming threshold)")
    ax1.legend(handles=ax1.get_legend_handles_labels()[0]
               + [patch_icc, patch_thr],
               labels=ax1.get_legend_handles_labels()[1]
               + ["I_cc (per stage)", "0.8×I_cc threshold"],
               facecolor="#000000", edgecolor="#FFFFFF",
               labelcolor="#FFFFFF", fontsize=7.5, loc="upper left")

    # ── Right panel: R-V (resistance evolution during forming) ──
    for k, res in enumerate(results):
        col   = STAGE_COLORS[k % len(STAGE_COLORS)]
        v_arr = np.array(res["voltages"])
        i_arr = np.array(res["currents"])
        safe  = np.abs(i_arr) > 1e-12
        r_arr = (np.where(safe, np.abs(v_arr / i_arr), np.nan) - r_series) / 1e3   # device resistance
        v_x   = device_voltages(res, r_series) if r_series > 0 else v_arr

        # Plot only points with V > 0.1 V (avoids the singularity near zero)
        mask = v_arr > 0.1
        if mask.sum() > 1:
            label = res["stage"].split(":")[0]
            ax2.scatter(v_x[mask], r_arr[mask],
                        c=col, s=15, alpha=0.75,
                        zorder=3, label=label)
            ax2.plot(v_x[mask], r_arr[mask],
                     color=col, lw=1.2, alpha=0.6, zorder=2)

    # LRS baseline annotation
    if lrs and np.isfinite(lrs["r_mean"]):
        r_lrs_k = lrs["r_mean"] / 1e3
        ax2.axhline(r_lrs_k, color="#69F0AE", lw=1.5,
                    linestyle="--", alpha=0.9,
                    label=f"LRS baseline ≈ {r_lrs_k:.1f} kΩ")
        ax2.text(0.15, r_lrs_k,
                 f"R_LRS = {r_lrs_k:.2f} kΩ\n"
                 f"(read at {lrs['v_read']} V, after relaxation)",
                 color="#69F0AE", fontsize=8, va="bottom", clip_on=True)

    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(ticker.LogFormatter(minor_thresholds=(2, 0.5)))
    ax2.yaxis.set_minor_formatter(ticker.LogFormatter(minor_thresholds=(2, 0.5)))
    ax2.set_xlim(-0.05, v_max + 0.2)
    ax2.set_xlabel("Device Voltage (V)" if r_series > 0 else "Voltage (V)",
                   color="#FFFFFF", fontsize=12)
    ax2.set_ylabel("Resistance (kΩ)", color="#FFFFFF", fontsize=12)
    ax2.set_title("Resistance vs. Forming Voltage",
                  color="#FFFFFF", fontsize=11, pad=8)
    ax2.legend(facecolor="#000000", edgecolor="#FFFFFF",
               labelcolor="#FFFFFF", fontsize=8.5)
    ax2.grid(True, color="#3A3A3A", lw=0.5, which="both", zorder=0)

    # Overall title
    formed_any = any(r["formed"] for r in results)
    v_f_str = "N/A"
    for res in results:
        if res["formed"] and res["v_forming"]:
            v_f_str = f"{res['v_forming']:.2f} V  ({res['stage'].split(':')[0]})"
            if r_series > 0 and res.get("i_forming") is not None:   # series R: give the device voltage too
                v_dev = device_voltage(res["v_forming"], res["i_forming"], r_series)
                v_f_str = f"{res['v_forming']:.2f} V applied / {v_dev:.2f} V device  ({res['stage'].split(':')[0]})"
            break

    r_lrs_str = (f"{lrs['r_mean']/1e3:.2f} kΩ"
                 if lrs and np.isfinite(lrs["r_mean"]) else "N/A")

    status_str = "✓ Formed" if formed_any else "✗ Not formed"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rs_str = f"Rs={fmt_series(r_series)}   " if r_series > 0 else ""
    fig.suptitle(
        f"Ni/TiOx-AlOx/Ti   Soft Forming   {status_str}   "
        f"V_form={v_f_str}   R_LRS={r_lrs_str}   {rs_str}{ts}",
        color="#FFFFFF", fontsize=9, y=1.01)

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[PNG] Saved: {img_path}")
    plt.show()


# ──────────────────────────────────────────────────────
#  Main program
# ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Progressive memristor forming")
    parser.add_argument("--address", default=DEFAULT_VISA_ADDRESS)
    parser.add_argument("--dev",     default="DEV",
                        help="device label used in the folder name (e.g. DEV-01)")
    parser.add_argument("--v-max", type=float, default=V_MAX,
                        help="forming voltage limit, default 2 V")
    parser.add_argument("--allow-overvoltage", action="store_true",
                        help="explicitly allow more than 2 V; a typed confirmation is still required")
    add_series_argument(parser)
    parser.set_defaults(r_series=R_SERIES)
    args = parser.parse_args()
    validate_series(args.r_series)

    for icc, _ in ICC_LADDER:
        validate_safety_limits(0.0, icc)
    validate_safety_limits(V_READ, min(icc for icc, _ in ICC_LADDER))
    if abs(args.v_max) > DEFAULT_MAX_VOLTAGE_V:
        if not args.allow_overvoltage:
            parser.error(
                "--v-max above 2 V requires --allow-overvoltage to be given explicitly"
            )
        confirm = input(
            f"DANGER: requested V_MAX={args.v_max:g} V exceeds 2 V. "
            "Type OVERVOLTAGE to confirm: "
        ).strip()
        if confirm != "OVERVOLTAGE":
            print("[CANCELLED] Overvoltage confirmation not completed.")
            return
    validate_safety_limits(
        args.v_max,
        max(icc for icc, _ in ICC_LADDER),
        allow_overvoltage=args.allow_overvoltage,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.r_series > 0:
        ts_str = f"{series_tag(args.r_series)}_{ts_str}"   # ..._RS10k_<timestamp>

    # Folder name: device label + forming parameters + timestamp
    run_name = (
        f"{args.dev}_"
        f"Vmax{args.v_max}_"
        f"step{int(V_STEP*1000)}mV_"
        f"dwell{int(DWELL_TIME*1000)}ms_"
        f"{ts_str}"
    )
    RUN_DIR = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(RUN_DIR, exist_ok=True)

    csv_path = os.path.join(RUN_DIR, f"forming_{ts_str}.csv")
    img_path = os.path.join(RUN_DIR, f"forming_{ts_str}.png")

    # Confirm parameters
    print("=" * 54)
    print("  Progressive forming parameters")
    print("=" * 54)
    print(f"  V_MAX       = {args.v_max} V")
    print(f"  V_STEP      = {V_STEP*1000:.0f} mV")
    print(f"  DWELL_TIME  = {DWELL_TIME*1000:.0f} ms / step")
    print(f"  NPLC        = {NPLC}")
    print(f"  FORM_THR    = {FORM_THRESHOLD} × I_cc")
    print(f"  I_cc ladder:")
    for icc, label in ICC_LADDER:
        print(f"    {icc*1e6:>7.0f} uA  — {label}")
    n_steps = int((args.v_max - V_START) / V_STEP) + 1
    est_s   = n_steps * DWELL_TIME * len(ICC_LADDER)
    print(f"  Max. time   ≈ {est_s:.0f} s (if no stage forms)")
    print(f"  Device      = {args.dev}")
    if args.r_series > 0:
        print(f"  Series R    = {fmt_series(args.r_series)} (subtracted from R; device voltage logged)")
    print(f"  Output dir  = {RUN_DIR}")
    print("=" * 54)

    ans = input("\nStart forming? (Enter / q to quit): ").strip().lower()
    if ans == "q":
        return

    inst = connect(args.address)

    results = []
    formed  = False

    try:
        # ── Try each I_cc stage in turn ──────────────────
        for stage_idx, (i_cc, label) in enumerate(ICC_LADDER):
            init_smu(inst, i_cc, NPLC)
            res = forming_ramp(
                inst, i_cc, label,
                V_START, args.v_max, V_STEP, DWELL_TIME
            )
            results.append(res)
            if res["formed"]:
                formed = True
                break
            if stage_idx < len(ICC_LADDER) - 1:
                print(f"\n  [NEXT STAGE] Trying the next I_cc...")
                time.sleep(1.0)

        lrs = {"r_mean": float("inf"), "r_std": 0.0,
               "status": "N/A", "v_read": V_READ}
        if formed:
            read_i_cc = min(i_cc, MAX_SAFE_CURRENT_A)
            init_smu(inst, read_i_cc, NPLC)
            lrs = read_lrs(inst, V_READ, N_READ, READ_SETTLE, read_i_cc,
                           args.r_series)
        else:
            print("\n[RESULT] No stage formed the device; skipping the LRS read")
    finally:
        safe_shutdown(inst)
        inst.close()

    # ── Summary printout ─────────────────────────────
    print("\n" + "=" * 54)
    print("  Forming summary")
    print("=" * 54)
    print(f"  Formed    : {'yes ✓' if formed else 'no ✗'}")
    for res in results:
        tag = "★" if res["formed"] else " "
        vf  = f"{res['v_forming']:.3f} V" if res["v_forming"] else "—"
        if res["v_forming"] and args.r_series > 0 and res.get("i_forming") is not None:
            vf += f"  (device {device_voltage(res['v_forming'], res['i_forming'], args.r_series):.3f} V)"
        print(f"  {tag} {res['stage']:<28} V_form = {vf}")
    if formed:
        print(f"  R_LRS     : {lrs['r_mean']/1e3:.2f} kΩ  {lrs['status']}")

        steps_all = []
        for res in results:
            if res["formed"]:
                steps_all = detect_step(res["voltages"], res["currents"])
                break
        if steps_all:
            print(f"  I steps   : {len(steps_all)} (superlattice interface feature)")
            for n, (vs, _) in enumerate(steps_all):
                print(f"    Step {n+1}: {vs:.3f} V")
    print("=" * 54)

    # ── Save & plot ──────────────────────────────────
    save_csv(results, lrs, csv_path, args.r_series)
    plot_forming(results, lrs, args.v_max, img_path, args.r_series)


if __name__ == "__main__":
    main()