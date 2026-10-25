"""
iv_sweep.py  v5 — cycle-duration control + R-V / G-V views
=================================================
Additions to the v4 asymmetric-compliance version:

  [Feature 1] Sweep-rate control
    CYCLE_DURATION_S: desired total duration of one cycle (s)
      - None  → v4 behaviour: step_delay derived from NPLC (~20-40 ms/point)
      - float → step_delay = CYCLE_DURATION_S / (N_POINTS*4), and the actual
                sweep rate dV/dt (V/s) is printed for comparing hysteresis

  [Feature 2] R-V and G-V views
    Single-cycle figure: 2×2 layout
      top left: I-V       top right: R-V (resistance vs voltage, log)
      bottom left: G-V    bottom right: R vs sample index
    Summary figure: 2×2 layout (as above, cycles overlaid)

    R-V: shows the LRS/HRS branches splitting along the voltage axis (hysteresis window)
    G-V: G = I/V (µS), which corresponds directly to the CIM weights

  --r-series <Ohm>: series resistor on the HI lead; R/G are corrected for it, the device
                    voltage is recorded and the output name carries an RS tag

Device stack: Ni / AlOx / TiOx / Ti
Dependencies: pip install pyvisa pyvisa-py numpy matplotlib
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
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
import numpy as np

try:
    from .keithley_smu import (
        add_series_argument,
        validate_series,
        series_tag,
        device_voltage,
    )
except ImportError:
    from keithley_smu import (
        add_series_argument,
        validate_series,
        series_tag,
        device_voltage,
    )

# ══════════════════════════════════════════════════════
#  ★ Adjustable parameters ★
# ══════════════════════════════════════════════════════
DEFAULT_VISA_ADDRESS = "USB0::0x05E6::0x2450::04437634::INSTR"

V_MAX    =  1.0   # positive sweep limit (V)
V_MIN    = -1.0    # negative sweep limit (V)
N_POINTS =  100    # points per quarter segment (full cycle = N_POINTS × 4)
N_CYCLES =  1      # number of sweep cycles
NPLC     =  0.5    # integration time (sets the step delay only if CYCLE_DURATION_S=None)

# ── [New] Cycle-duration control ─────────────────────────────
# None  → step delay derived from NPLC (v4 behaviour)
# float → fixes one cycle at this duration (s), so the sweep rate can be set for hysteresis
#   e.g. CYCLE_DURATION_S = 10   → 10 s/cycle
#        CYCLE_DURATION_S = 60   → 60 s/cycle (slow sweep, fuller hysteresis)
#        CYCLE_DURATION_S = 2    → 2  s/cycle (fast sweep, smaller hysteresis)
# Note: step_delay cannot be below the instrument response time (~20 ms);
# values that are too short are raised automatically
CYCLE_DURATION_S = 120  # s/cycle; None derives it from NPLC

# ── Asymmetric compliance ────────────────────────────────────
SET_POLARITY = "positive"   # "positive" = positive SET / negative RESET
I_CC_SET   = 300e-6        # SET-direction compliance current (A)
I_CC_RESET = 300e-6        # RESET-direction compliance current (A)

# ── Series resistor ──────────────────────────────────────────
R_SERIES = 0.0             # series resistor on the HI lead (Ohm), 0 = none; set by --r-series
# ══════════════════════════════════════════════════════

OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IV_sweep")
CYCLE_PALETTE = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]
_MIN_STEP_DELAY = 0.020     # shortest safe step interval of the instrument (s)


def _series_title() -> str:
    """Series-resistor label for plot titles; empty when R_SERIES=0."""
    return f"   Rs={R_SERIES/1e3:g} kOhm" if R_SERIES > 0 else ""


def _voltage_label() -> str:
    return "Device Voltage (V)" if R_SERIES > 0 else "Voltage (V)"


# ──────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────
def _compute_step_delay(n_points: int, nplc: float,
                        cycle_duration_s) -> float:
    """
    Compute the step delay.
    - cycle_duration_s=None: derived from NPLC (v4 behaviour)
    - cycle_duration_s=float: from the target cycle duration, not below _MIN_STEP_DELAY
    """
    if cycle_duration_s is None:
        return max(_MIN_STEP_DELAY, nplc / 60 * 1.5 + 0.020)
    else:
        cycle_pts = n_points * 4
        d = cycle_duration_s / cycle_pts
        if d < _MIN_STEP_DELAY:
            print(f"[WARNING] Step delay {d*1e3:.1f} ms < minimum {_MIN_STEP_DELAY*1e3:.0f} ms,"
                  f" raised to {_MIN_STEP_DELAY*1e3:.0f} ms")
            d = _MIN_STEP_DELAY
        return d


def _sweep_rate(v_max: float, v_min: float,
                n_points: int, step_delay: float) -> float:
    """Peak-to-peak amplitude / half-cycle time → dV/dt (V/s), mean slope of a quarter segment"""
    # Per quarter segment: ΔV = v_max, time = n_points * step_delay
    return v_max / (n_points * step_delay)


def icc_for_voltage(v: float) -> float:
    if SET_POLARITY == "positive":
        return I_CC_SET if v >= 0 else I_CC_RESET
    else:
        return I_CC_SET if v <= 0 else I_CC_RESET


# ──────────────────────────────────────────────────────
def connect(address):
    try:
        import pyvisa
    except ImportError:
        print("[ERROR] pip install pyvisa pyvisa-py"); sys.exit(1)

    rm = pyvisa.ResourceManager()
    print(f"[CONNECT] {address}")
    try:
        inst = rm.open_resource(address)
    except Exception as e:
        print(f"[ERROR] {e}")
        for r in rm.list_resources(): print(f"  Available: {r}")
        sys.exit(1)

    inst.write_termination = "\n"
    inst.read_termination  = "\n"
    inst.timeout           = 10000

    try:
        inst.clear(); time.sleep(0.3)
    except Exception:
        pass
    inst.write("*CLS"); time.sleep(0.1)

    idn = inst.query("*IDN?").strip()
    print(f"[CONNECT] Connected -> {idn}\n")
    return inst


def set_current_limit(inst, requested_icc: float) -> float:
    candidates = [
        (":SOUR:VOLT:ILIM:LEV", ":SOUR:VOLT:ILIM:LEV?"),
        (":SOUR:VOLT:ILIM",     ":SOUR:VOLT:ILIM?"),
    ]
    errors  = []
    actual  = requested_icc
    accepted = False

    for set_cmd, query_cmd in candidates:
        inst.write(f"{set_cmd} {requested_icc:.6e}")
        time.sleep(0.02)

        err = inst.query(":SYST:ERR?").strip()
        if not err.startswith("0"):
            errors.append(f"{set_cmd}: {err}")
            continue

        try:
            actual = float(inst.query(query_cmd).strip())
            q_err  = inst.query(":SYST:ERR?").strip()
            if not q_err.startswith("0"):
                errors.append(f"{query_cmd}: {q_err}")
                actual = requested_icc
            else:
                accepted = True
                break
        except Exception as e:
            errors.append(f"{query_cmd}: {e}")
            actual   = requested_icc
            accepted = True
            break

    if not accepted:
        raise RuntimeError(
            f"Instrument rejected Icc={requested_icc:.6e} A "
            f"({requested_icc*1e3:.3f} mA). Tried: {' | '.join(errors)}"
        )
    if abs(actual - requested_icc) > max(abs(requested_icc)*0.01, 1e-9):
        print(f"[Warning] Requested Icc={requested_icc:.6e} A, "
              f"instrument reports {actual:.6e} A")
    return actual


def init_smu(inst, i_cc_init, nplc):
    cmds = [
        "*RST", "*CLS",
        ":SOUR:FUNC VOLT",
        ":SOUR:VOLT:RANG:AUTO 1",
        ":SOUR:VOLT:LEV 0",
        ':SENS:FUNC "CURR"',
        ":SENS:CURR:RANG:AUTO 1",
        f":SENS:CURR:NPLC {nplc:.2f}",
        ":SYST:AZER:ONCE",
        ":OUTP OFF",
    ]
    for c in cmds:
        inst.write(c)
    set_current_limit(inst, i_cc_init)
    time.sleep(0.3)

    err = inst.query(":SYST:ERR?").strip()
    if not err.startswith("0"):
        print(f"[WARNING] Initialisation: {err}")
    else:
        print("[INIT] Done, no errors")


def build_cycle(v_max, v_min, n):
    s1 = np.linspace(0,     v_max, n, endpoint=False)
    s2 = np.linspace(v_max, 0,     n, endpoint=False)
    s3 = np.linspace(0,     v_min, n, endpoint=False)
    s4 = np.linspace(v_min, 0,     n, endpoint=True)
    return np.concatenate([s1, s2, s3, s4])


def compute_resistance_with_icc_hold(voltages, currents, icc_used,
                                     trigger_ratio=0.985, r_series=0.0):
    # With r_series>0, returns the device resistance |V/I| - r_series
    voltages   = np.asarray(voltages,   dtype=float)
    currents   = np.asarray(currents,   dtype=float)
    icc_used   = np.asarray(icc_used,   dtype=float)
    resistance = np.full(len(voltages), np.nan, dtype=float)

    held_r       = None
    in_compliance = False

    for k, (v, i, icc) in enumerate(zip(voltages, currents, icc_used)):
        if not (np.isfinite(v) and np.isfinite(i)) or abs(i) <= 1e-12:
            resistance[k] = held_r if in_compliance and held_r is not None else np.nan
            continue

        raw_r       = abs(v / i) - r_series
        at_compliance = (np.isfinite(icc) and icc > 0
                         and abs(i) >= abs(icc) * trigger_ratio)

        if at_compliance:
            if not in_compliance:
                held_r = (resistance[k-1] if k > 0 and np.isfinite(resistance[k-1])
                          else raw_r)
                in_compliance = True
            resistance[k] = held_r
        else:
            in_compliance = False
            held_r        = None
            resistance[k] = raw_r

    return resistance


# ──────────────────────────────────────────────────────
def run_sweep(inst, v_max, v_min, n_points, n_cycles, nplc, step_delay):
    one_cycle = build_cycle(v_max, v_min, n_points)
    cycle_len = len(one_cycle)
    total     = cycle_len * n_cycles

    # Sweep-rate information
    dvdt = _sweep_rate(v_max, v_min, n_points, step_delay)
    est_s = total * step_delay

    print(f"[SWEEP] {n_cycles} cycles × {cycle_len} points = {total} points")
    print(f"[SWEEP] Step delay    {step_delay*1e3:.1f} ms/point")
    if CYCLE_DURATION_S is not None:
        print(f"[SWEEP] Target cycle  {CYCLE_DURATION_S:.1f} s  "
              f"(actual ≈ {cycle_len*step_delay:.1f} s)")
    print(f"[SWEEP] Rate dV/dt    ≈ {dvdt:.3f} V/s  "
          f"(peak-to-peak {v_max-v_min:.1f} V / half cycle)")
    print(f"[SWEEP] Est. total    {est_s:.0f} s")
    print(f"[SWEEP] Asymmetric compliance: SET={I_CC_SET*1e6:.0f}µA  "
          f"RESET={I_CC_RESET*1e3:.1f}mA  (SET polarity={SET_POLARITY})")

    voltages   = np.zeros(total)
    currents   = np.zeros(total)
    timestamps = np.zeros(total)
    icc_used   = np.zeros(total)
    cur_icc    = None

    inst.write(":OUTP ON")
    t_start = time.time()

    print(f"\n  {'Step':>6}  {'V (V)':>9}  {'I (µA)':>11}  {'R (kΩ)':>10}  {'I_cc':>8}")
    print("  " + "─" * 56)

    for idx in range(total):
        v_set    = one_cycle[idx % cycle_len]
        want_icc = icc_for_voltage(v_set)

        if cur_icc is None or abs(want_icc - cur_icc) > 1e-12:
            cur_icc = set_current_limit(inst, want_icc)

        inst.write(f":SOUR:VOLT:LEV {v_set:.6f}")
        time.sleep(step_delay)

        try:
            raw   = inst.query(":READ?").strip()
            parts = raw.split(",")
            i     = float(parts[1]) if len(parts) >= 2 else float(parts[0])
        except Exception as e:
            print(f"  [READ WARNING #{idx}] {e}")
            i = float("nan")

        voltages[idx]   = v_set
        currents[idx]   = i
        timestamps[idx] = time.time() - t_start
        icc_used[idx]   = cur_icc

        report_every = max(1, cycle_len // 4)
        if idx % report_every == 0 or idx == total - 1:
            r_str = (f"{(abs(v_set/i) - R_SERIES)/1e3:.2f}"
                     if abs(i) > 1e-11 else "  inf")
            pct   = (idx + 1) / total * 100
            icc_s = (f"{cur_icc*1e3:.1f}mA" if cur_icc >= 1e-3
                     else f"{cur_icc*1e6:.0f}µA")
            print(f"  {pct:5.1f}%  {v_set:>9.4f}  {i*1e6:>11.4f}  "
                  f"{r_str:>10}  {icc_s:>8}")

    inst.write(":OUTP OFF")
    inst.write(":SOUR:VOLT:LEV 0")
    elapsed = time.time() - t_start
    print(f"\n[SWEEP] Complete, elapsed time {elapsed:.1f} s")
    return voltages, currents, timestamps, cycle_len, icc_used


# ──────────────────────────────────────────────────────
def save_csv(voltages, currents, timestamps, resistance, icc_used,
             cycle_len, path, output_dir=None, ts_str=None,
             step_delay=None):
    import csv
    n        = len(voltages)
    n_cycles = n // cycle_len

    # Device voltage = set voltage - I*R_SERIES (the set voltage itself when R_SERIES=0)
    v_device = device_voltage(np.asarray(voltages, dtype=float),
                              np.asarray(currents, dtype=float), R_SERIES)

    # G = I/V_device (S), exclude near-zero voltage
    with np.errstate(divide="ignore", invalid="ignore"):
        conductance = np.where(np.abs(v_device) > 1e-6,
                               currents / v_device, np.nan)

    header = ["index", "cycle", "index_in_cycle", "time_s",
              "voltage_V", "current_A", "current_uA",
              "resistance_Ohm", "conductance_S", "icc_A",
              "device_voltage_V"]

    def row(k):
        r = resistance[k]
        g = conductance[k]
        return [k, k // cycle_len, k % cycle_len,
                f"{timestamps[k]:.4f}",
                f"{voltages[k]:.6f}",
                f"{currents[k]:.6e}",
                f"{currents[k]*1e6:.4f}",
                f"{r:.2f}"  if np.isfinite(r) else "inf",
                f"{g:.6e}"  if np.isfinite(g) else "nan",
                f"{icc_used[k]:.6e}",
                f"{v_device[k]:.6f}" if np.isfinite(v_device[k]) else "nan"]

    # Write the sweep-rate metadata into the CSV header
    with open(path, "w", newline="", encoding="utf-8") as f:
        if step_delay is not None:
            dvdt = _sweep_rate(V_MAX, V_MIN, N_POINTS, step_delay)
            f.write(f"# step_delay_ms={step_delay*1e3:.2f}  "
                    f"dvdt_V_per_s={dvdt:.4f}  "
                    f"cycle_duration_s={cycle_len*step_delay:.2f}  "
                    f"r_series_Ohm={R_SERIES:g}\n")
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
#  Plotting helpers
# ──────────────────────────────────────────────────────
def _conductance(voltages, currents, v_deadband=1e-6):
    """G = I/V (S), excluding points with |V| < v_deadband"""
    v = np.asarray(voltages, dtype=float)
    i = np.asarray(currents, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(np.abs(v) > v_deadband, i / v, np.nan)
    return g


def _dark_axes(ax):
    ax.set_facecolor("#000000")
    ax.tick_params(which="both", colors="#FFFFFF", labelsize=11)
    for sp in ax.spines.values():
        sp.set_color("#FFFFFF")


def _icc_reference_lines(ax):
    if SET_POLARITY == "positive":
        icc_pos, icc_neg = I_CC_SET, I_CC_RESET
    else:
        icc_pos, icc_neg = I_CC_RESET, I_CC_SET

    lbl_p = (f"+I_cc = {icc_pos*1e6:.0f} µA" if icc_pos < 1e-3
             else f"+I_cc = {icc_pos*1e3:.1f} mA")
    lbl_n = (f"-I_cc = {icc_neg*1e6:.0f} µA" if icc_neg < 1e-3
             else f"-I_cc = {icc_neg*1e3:.1f} mA")
    ax.axhline( icc_pos*1e6, color="#FF5722", lw=0.9, ls="--",
               alpha=0.7, label=lbl_p)
    ax.axhline(-icc_neg*1e6, color="#FF9800", lw=0.9, ls="--",
               alpha=0.7, label=lbl_n)


SEG_DEFS = [
    (0,    1/4, "#69F0AE", "0→V_max"),
    (1/4,  1/2, "#FFD54F", "V_max→0"),
    (1/2,  3/4, "#80DEEA", "0→V_min"),
    (3/4,  1,   "#EF9A9A", "V_min→0"),
]


def _seg_slice(cycle_len: int, frac_lo: float, frac_hi: float):
    return slice(int(frac_lo * cycle_len), int(frac_hi * cycle_len))


# ── Subplot functions ────────────────────────────────────────
def _plot_iv(ax, v, i_ua, col=None, label=None, arrows=True):
    """I-V curve with optional colour and label (for overlaying cycles)"""
    if col is None:
        # Single cycle: coloured segments
        cycle_len = len(v)
        for lo_f, hi_f, seg_col, seg_label in SEG_DEFS:
            sl = _seg_slice(cycle_len, lo_f, hi_f)
            ax.plot(v[sl], i_ua[sl], color=seg_col, lw=2.0,
                    alpha=0.9, label=seg_label, zorder=3)
            # Direction arrows
            if arrows:
                sl_list = list(range(*sl.indices(cycle_len)))
                if len(sl_list) < 2: continue
                mid = sl_list[len(sl_list)//2]
                if mid+1 >= cycle_len: continue
                dv = v[mid+1]-v[mid]; di = i_ua[mid+1]-i_ua[mid]
                if abs(dv)+abs(di) < 1e-9: continue
                ax.annotate("", xy=(v[mid+1], i_ua[mid+1]),
                    xytext=(v[mid], i_ua[mid]),
                    arrowprops=dict(arrowstyle="-|>", color=seg_col,
                                    lw=1.0, mutation_scale=10))
        ax.scatter([v[0]], [i_ua[0]], color="#FFFFFF", s=70, zorder=6,
                   marker="o", edgecolors=CYCLE_PALETTE[0], linewidths=1.5,
                   label="Start")
    else:
        # Overlaid cycles: single colour
        ax.plot(v, i_ua, color=col, lw=1.8, alpha=0.85,
                label=label, zorder=3)
        ax.scatter([v[0]], [i_ua[0]], color=col, s=55, zorder=6,
                   marker="o", edgecolors="white", linewidths=0.6)

    _icc_reference_lines(ax)
    ax.axhline(0, color="#888888", lw=0.5)
    ax.axvline(0, color="#888888", lw=0.5)
    vp = max((v.max()-v.min())*0.08, 0.2)
    ip = max((i_ua.max()-i_ua.min())*0.08, 10.0)
    ax.set_xlim(v.min()-vp, v.max()+vp)
    ax.set_ylim(i_ua.min()-ip, i_ua.max()+ip)
    ax.set_xlabel(_voltage_label(),  color="#FFFFFF", fontsize=12)
    ax.set_ylabel("Current (µA)", color="#FFFFFF", fontsize=12)
    ax.legend(facecolor="#000000", edgecolor="#FFFFFF",
              labelcolor="#FFFFFF", fontsize=7.5, loc="best")
    ax.grid(True, color="#3A3A3A", lw=0.5, zorder=0)


def _plot_rv(ax, v, r_kOhm, col=None, label=None):
    """
    R-V curve (log R vs V)
    The hysteresis window is visible along the voltage axis as the split in R
    between the SET and RESET paths.
    """
    cycle_len = len(v)
    if col is None:
        for lo_f, hi_f, seg_col, seg_label in SEG_DEFS:
            sl = _seg_slice(cycle_len, lo_f, hi_f)
            vsl = v[sl]; rsl = r_kOhm[sl]
            fin = np.isfinite(rsl) & (rsl > 0)
            if fin.sum() < 2: continue
            ax.plot(vsl[fin], rsl[fin], color=seg_col, lw=2.0,
                    alpha=0.9, label=seg_label, zorder=3)
    else:
        fin = np.isfinite(r_kOhm) & (r_kOhm > 0)
        ax.plot(v[fin], r_kOhm[fin], color=col, lw=1.8,
                alpha=0.85, label=label, zorder=3)

    ax.axvline(0, color="#888888", lw=0.5)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.LogFormatter(minor_thresholds=(2, 0.5)))
    ax.yaxis.set_minor_formatter(ticker.LogFormatter(minor_thresholds=(2, 0.5)))
    ax.set_xlabel(_voltage_label(),    color="#FFFFFF", fontsize=12)
    ax.set_ylabel("Resistance (kΩ)", color="#FFFFFF", fontsize=12)
    ax.legend(facecolor="#000000", edgecolor="#FFFFFF",
              labelcolor="#FFFFFF", fontsize=7.5, loc="best")
    ax.grid(True, color="#3A3A3A", lw=0.5, which="both", zorder=0)


def _plot_gv(ax, v, g_uS, col=None, label=None):
    """
    G-V curve (µS vs V)
    G = I/V; points with |V| < 1 mV are excluded (division-by-zero guard).
    The LRS → HRS transition appears here as a sudden drop in G.
    """
    cycle_len = len(v)
    if col is None:
        for lo_f, hi_f, seg_col, seg_label in SEG_DEFS:
            sl = _seg_slice(cycle_len, lo_f, hi_f)
            vsl = v[sl]; gsl = g_uS[sl]
            fin = np.isfinite(gsl)
            if fin.sum() < 2: continue
            ax.plot(vsl[fin], gsl[fin], color=seg_col, lw=2.0,
                    alpha=0.9, label=seg_label, zorder=3)
    else:
        fin = np.isfinite(g_uS)
        ax.plot(v[fin], g_uS[fin], color=col, lw=1.8,
                alpha=0.85, label=label, zorder=3)

    ax.axhline(0, color="#888888", lw=0.5)
    ax.axvline(0, color="#888888", lw=0.5)
    ax.set_xlabel(_voltage_label(),       color="#FFFFFF", fontsize=12)
    ax.set_ylabel("Conductance (µS)",  color="#FFFFFF", fontsize=12)
    ax.legend(facecolor="#000000", edgecolor="#FFFFFF",
              labelcolor="#FFFFFF", fontsize=7.5, loc="best")
    ax.grid(True, color="#3A3A3A", lw=0.5, zorder=0)


def _plot_r_index(ax, x, r_kOhm, col, label=None, cycle_len=None,
                  show_percentile=True):
    """R vs sample index (log y axis), for the evolution over time"""
    fin = np.isfinite(r_kOhm) & (r_kOhm > 0)
    ax.scatter(x[fin], r_kOhm[fin], c=col, s=5, alpha=0.7, zorder=3)

    # Median smoothing
    win = max(5, int(len(x) * 0.08)); half = win // 2
    sm  = np.full(len(x), np.nan)
    for k in range(len(x)):
        ch = r_kOhm[max(0,k-half):min(len(x),k+half+1)]
        vc = ch[np.isfinite(ch) & (ch > 0)]
        if len(vc): sm[k] = np.nanmedian(vc)
    ax.plot(x, sm, color=col, lw=2.0, alpha=0.9,
            label=label or "Median smooth")

    if cycle_len:
        for sep in [cycle_len//4, cycle_len//2, 3*cycle_len//4]:
            ax.axvline(x[0]+sep, color="#888888", lw=0.7, ls=":", alpha=0.7)

    if show_percentile:
        r_fin = r_kOhm[fin]
        if len(r_fin) >= 5:
            rlo = np.percentile(r_fin, 5)
            rhi = np.percentile(r_fin, 95)
            ax.axhline(rlo, color="#4DB6AC", lw=1.0, ls=":", alpha=0.9,
                       label=f"P5  ≈ {rlo:.1f} kΩ")
            ax.axhline(rhi, color="#FFB74D", lw=1.0, ls=":", alpha=0.9,
                       label=f"P95 ≈ {rhi:.1f} kΩ")
            if rlo > 0:
                ax.text(0.97, 0.04, f"P95/P5 ≈ {rhi/rlo:.1f}×",
                        transform=ax.transAxes, color="#CE93D8",
                        fontsize=9, ha="right",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  fc="#000000", ec="#9C27B0", alpha=0.85))

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.LogFormatter(minor_thresholds=(2, 0.5)))
    ax.yaxis.set_minor_formatter(ticker.LogFormatter(minor_thresholds=(2, 0.5)))
    ax.set_xlabel("Sample Index", color="#FFFFFF", fontsize=12)
    ax.set_ylabel("Resistance (kΩ)", color="#FFFFFF", fontsize=12)
    ax.legend(facecolor="#000000", edgecolor="#FFFFFF",
              labelcolor="#FFFFFF", fontsize=8, loc="upper right")
    ax.grid(True, color="#3A3A3A", lw=0.5, which="both", zorder=0)


# ──────────────────────────────────────────────────────
#  Single-cycle figure (2×2)
# ──────────────────────────────────────────────────────
def plot_individual_cycles(voltages, currents, resistance,
                           n_cycles, cycle_len,
                           v_max, v_min, output_dir, ts_str,
                           step_delay=None):
    v_all  = voltages
    i_all  = currents * 1e6
    r_all  = resistance / 1e3
    g_all  = _conductance(voltages, currents) * 1e6  # µS

    for cyc in range(n_cycles):
        lo = cyc * cycle_len
        hi = (cyc + 1) * cycle_len

        v_c  = v_all[lo:hi]
        i_c  = i_all[lo:hi]
        r_c  = r_all[lo:hi]
        g_c  = g_all[lo:hi]
        x_c  = np.arange(cycle_len)
        col  = CYCLE_PALETTE[cyc % len(CYCLE_PALETTE)]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor("#000000")
        for ax in axes.flat:
            _dark_axes(ax)

        ax_iv  = axes[0, 0]
        ax_rv  = axes[0, 1]
        ax_gv  = axes[1, 0]
        ax_ri  = axes[1, 1]

        # I-V
        _plot_iv(ax_iv, v_c, i_c)
        ax_iv.set_title(f"I–V  |  Cycle {cyc+1}", color="#FFFFFF",
                        fontsize=11, pad=8)

        # R-V
        _plot_rv(ax_rv, v_c, r_c)
        ax_rv.set_title(f"R–V  |  Cycle {cyc+1}", color="#FFFFFF",
                        fontsize=11, pad=8)

        # G-V
        _plot_gv(ax_gv, v_c, g_c)
        ax_gv.set_title(f"G–V  |  Cycle {cyc+1}", color="#FFFFFF",
                        fontsize=11, pad=8)

        # R vs index
        _plot_r_index(ax_ri, x_c, r_c, col, cycle_len=cycle_len)
        ax_ri.set_title(f"R vs Index  |  Cycle {cyc+1}", color="#FFFFFF",
                        fontsize=11, pad=8)

        icc_str = (f"SET={I_CC_SET*1e6:.0f}µA / "
                   f"RESET={I_CC_RESET*1e3:.1f}mA")
        dvdt_str = ""
        if step_delay is not None:
            dvdt = _sweep_rate(v_max, v_min, cycle_len//4, step_delay)
            dvdt_str = f"   dV/dt≈{dvdt:.3f}V/s"
        fig.suptitle(
            f"Ni/AlOx/TiOx/Ti   Cycle {cyc+1}/{n_cycles}   "
            f"V: {v_min}~{v_max} V   {cycle_len} pts   "
            f"{icc_str}{dvdt_str}{_series_title()}",
            color="#FFFFFF", fontsize=9, y=1.01)

        plt.tight_layout()
        img_path = os.path.join(output_dir,
                                f"iv_cycle{cyc+1}_{ts_str}.png")
        plt.savefig(img_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[PNG] Cycle {cyc+1} -> {img_path}")


# ──────────────────────────────────────────────────────
#  Summary figure (2×2, cycles overlaid)
# ──────────────────────────────────────────────────────
def plot_and_save(voltages, currents, resistance,
                  n_cycles, cycle_len,
                  v_max, v_min, img_path, step_delay=None):

    g_all = _conductance(voltages, currents) * 1e6   # µS
    r_all = resistance / 1e3
    v_all = voltages
    i_all = currents * 1e6

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#000000")
    for ax in axes.flat:
        _dark_axes(ax)

    ax_iv  = axes[0, 0]
    ax_rv  = axes[0, 1]
    ax_gv  = axes[1, 0]
    ax_ri  = axes[1, 1]

    LINESTYLES = ["-", "--", "-.", (0,(3,1,1,1)), (0,(5,1))]

    for cyc in range(n_cycles):
        lo  = cyc * cycle_len
        hi  = (cyc + 1) * cycle_len
        col = CYCLE_PALETTE[cyc % len(CYCLE_PALETTE)]
        ls  = LINESTYLES[cyc % len(LINESTYLES)]
        lbl = f"Cycle {cyc+1}"
        idx = np.arange(lo, hi)

        v_c  = v_all[lo:hi]
        i_c  = i_all[lo:hi]
        r_c  = r_all[lo:hi]
        g_c  = g_all[lo:hi]

        # I-V (single colour + arrows)
        ax_iv.plot(v_c, i_c, color=col, lw=1.8, ls=ls, alpha=0.85,
                   zorder=2+cyc*0.1, label=lbl)
        ax_iv.scatter([v_c[0]], [i_c[0]], color=col, s=55, zorder=6,
                      marker="o", edgecolors="white", linewidths=0.6)
        m = len(v_c)
        for ap in np.linspace(0, m-2, 4, dtype=int):
            dv = v_c[ap+1]-v_c[ap]; di = i_c[ap+1]-i_c[ap]
            if abs(dv)+abs(di) < 1e-9: continue
            ax_iv.annotate("", xy=(v_c[ap+1], i_c[ap+1]),
                xytext=(v_c[ap], i_c[ap]),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=0.8, mutation_scale=9))

        # R-V
        _plot_rv(ax_rv, v_c, r_c, col=col, label=lbl)

        # G-V
        _plot_gv(ax_gv, v_c, g_c, col=col, label=lbl)

        # R vs index
        _plot_r_index(ax_ri, idx, r_c, col, label=lbl,
                      show_percentile=(cyc == n_cycles-1))

    # I-V final formatting
    _icc_reference_lines(ax_iv)
    ax_iv.axhline(0, color="#888888", lw=0.5)
    ax_iv.axvline(0, color="#888888", lw=0.5)
    vp = max((v_all.max()-v_all.min())*0.08, 0.2)
    ip = max((i_all.max()-i_all.min())*0.08, 10.0)
    ax_iv.set_xlim(v_all.min()-vp, v_all.max()+vp)
    ax_iv.set_ylim(i_all.min()-ip, i_all.max()+ip)
    ax_iv.set_xlabel(_voltage_label(),  color="#FFFFFF", fontsize=12)
    ax_iv.set_ylabel("Current (µA)", color="#FFFFFF", fontsize=12)
    ax_iv.set_title("I–V Characteristics", color="#FFFFFF", fontsize=11, pad=8)
    ax_iv.legend(facecolor="#000000", edgecolor="#FFFFFF",
                 labelcolor="#FFFFFF", fontsize=8.5, loc="best")
    ax_iv.grid(True, color="#3A3A3A", lw=0.5, zorder=0)

    ax_rv.set_title("R–V (Resistance vs Voltage)", color="#FFFFFF",
                    fontsize=11, pad=8)
    ax_gv.set_title("G–V (Conductance vs Voltage)", color="#FFFFFF",
                    fontsize=11, pad=8)

    # R vs index: cycle separators
    for cyc in range(1, n_cycles):
        ax_ri.axvline(cyc*cycle_len, color="#888888", lw=0.8, ls="--", alpha=0.6)
    ax_ri.set_title("Resistance vs Sample Index", color="#FFFFFF",
                    fontsize=11, pad=8)

    icc_str = f"SET={I_CC_SET*1e6:.0f}µA / RESET={I_CC_RESET*1e3:.1f}mA"
    dvdt_str = ""
    if step_delay is not None:
        dvdt = _sweep_rate(v_max, v_min, cycle_len//4, step_delay)
        dvdt_str = f"   dV/dt≈{dvdt:.3f}V/s"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.suptitle(
        f"Ni/AlOx/TiOx/Ti   V: {v_min}~{v_max} V   "
        f"{n_cycles} cycles × {cycle_len} pts   "
        f"{icc_str}   NPLC={NPLC}{dvdt_str}{_series_title()}   {ts}",
        color="#FFFFFF", fontsize=9, y=1.01)

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[PNG] Summary  -> {img_path}")
    plt.show()


# ──────────────────────────────────────────────────────
def main():
    global R_SERIES

    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default=DEFAULT_VISA_ADDRESS)
    add_series_argument(parser)
    args = parser.parse_args()
    validate_series(args.r_series)
    R_SERIES = args.r_series

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # The series-resistor tag goes before the timestamp; file names are unchanged when R_SERIES=0
    rs_tag  = series_tag(R_SERIES)
    name_ts = f"{rs_tag}_{ts_str}" if rs_tag else ts_str

    cycle_len  = N_POINTS * 4
    step_delay = _compute_step_delay(N_POINTS, NPLC, CYCLE_DURATION_S)
    dvdt       = _sweep_rate(V_MAX, V_MIN, N_POINTS, step_delay)
    total      = cycle_len * N_CYCLES

    # File name includes the sweep rate
    dur_tag = (f"dur{CYCLE_DURATION_S:.0f}s"
               if CYCLE_DURATION_S is not None else f"NPLC{NPLC}")
    run_name = (
        f"V{V_MIN}_{V_MAX}_"
        f"cyc{N_CYCLES}_"
        f"pts{N_POINTS}_"
        f"SET{int(I_CC_SET*1e6)}uA_"
        f"RST{I_CC_RESET*1e3:.0f}mA_"
        f"{dur_tag}_"
        f"{name_ts}"
    )
    RUN_DIR = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"[OUTPUT] {RUN_DIR}")

    csv_path = os.path.join(RUN_DIR, f"iv_sweep_{name_ts}.csv")
    img_path = os.path.join(RUN_DIR, f"iv_sweep_{name_ts}.png")

    print("=" * 60)
    print("  IV sweep parameters (v5: cycle-duration control + R-V/G-V)")
    print("=" * 60)
    print(f"  V_MAX            = {V_MAX} V")
    print(f"  V_MIN            = {V_MIN} V")
    print(f"  N_POINTS         = {N_POINTS}  (per quarter segment)")
    print(f"  N_CYCLES         = {N_CYCLES}")
    print(f"  SET polarity     = {SET_POLARITY}")
    print(f"  I_CC_SET         = {I_CC_SET*1e6:.0f} µA")
    print(f"  I_CC_RESET       = {I_CC_RESET*1e3:.1f} mA")
    if CYCLE_DURATION_S is not None:
        print(f"  CYCLE_DURATION_S = {CYCLE_DURATION_S:.1f} s  (fixed cycle duration)")
    else:
        print(f"  NPLC             = {NPLC}  (CYCLE_DURATION_S=None, automatic)")
    if R_SERIES > 0:
        print(f"  Series R = {R_SERIES/1e3:g} kOhm "
              "(subtracted from R; device voltage logged)")
    print(f"  Step delay       = {step_delay*1e3:.1f} ms/point")
    print(f"  Rate dV/dt       ≈ {dvdt:.3f} V/s")
    print(f"  Total points     = {total}")
    print(f"  Est. total time  ≈ {total*step_delay:.0f} s")
    print(f"  Output dir       = {RUN_DIR}")
    print("=" * 60)
    if I_CC_RESET >= 1.0:
        print("  [WARNING] I_CC_RESET >= 1 A; check the units: 5000e-6 = 5 mA")
    print()

    ans = input("Start the sweep? (Enter / q to quit): ").strip().lower()
    if ans == "q":
        return

    inst = connect(args.address)
    init_smu(inst, I_CC_SET, NPLC)

    voltages, currents, timestamps, cycle_len, icc_used = run_sweep(
        inst, V_MAX, V_MIN, N_POINTS, N_CYCLES, NPLC, step_delay)
    inst.close()

    resistance = compute_resistance_with_icc_hold(voltages, currents, icc_used,
                                                  r_series=R_SERIES)

    fin = resistance[np.isfinite(resistance)]
    print(f"\n  Points     : {len(voltages)}")
    print(f"  V range    : {voltages.min():.4f} ~ {voltages.max():.4f} V")
    print(f"  Peak I     : {np.abs(currents).max()*1e6:.3f} µA")
    if len(fin):
        print(f"  R range    : {fin.min()/1e3:.3f} ~ {fin.max()/1e3:.3f} kΩ")

    # RESET diagnosis
    print("\n" + "─" * 56)
    print("  RESET diagnosis")
    print("─" * 56)
    v_last = voltages[-cycle_len:]
    i_last = currents[-cycle_len:]
    reset_mask = v_last < -0.1 if SET_POLARITY == "positive" else v_last > 0.1

    if reset_mask.sum() > 5:
        v_r  = v_last[reset_mask]
        i_r  = np.abs(i_last[reset_mask])
        with np.errstate(divide="ignore", invalid="ignore"):
            r_r = np.where(i_r > 1e-11, np.abs(v_r)/i_r - R_SERIES, np.nan)
        r_r_fin = r_r[np.isfinite(r_r)]
        if len(r_r_fin) >= 5:
            r_lo = np.percentile(r_r_fin, 10)
            r_hi = np.percentile(r_r_fin, 90)
            ratio = r_hi/r_lo if r_lo > 0 else 1.0
            print(f"  RESET-segment R: low {r_lo/1e3:.2f}kΩ → high {r_hi/1e3:.2f}kΩ  (ratio {ratio:.2f}×)")
            if   ratio >= 3.0: print("  [OK] R jumps >3× → possibly a true RESET!")
            elif ratio >= 1.5: print("  (~) R rises slightly → partial RESET")
            else:              print("  (!) R almost unchanged → RESET failed; check the Ti electrode")
        else:
            print("  [WARNING] Too few valid resistance points")
    else:
        print("  [WARNING] Too few samples in the RESET segment")
    print("─" * 56)

    save_csv(voltages, currents, timestamps, resistance, icc_used,
             cycle_len, csv_path, RUN_DIR, name_ts, step_delay=step_delay)

    # With R_SERIES>0 the x axis is the device voltage, and G is computed from it too
    plot_voltages = (device_voltage(voltages, currents, R_SERIES)
                     if R_SERIES > 0 else voltages)

    print("\n[PLOT] Per-cycle figures (2×2)...")
    plot_individual_cycles(plot_voltages, currents, resistance,
                           N_CYCLES, cycle_len, V_MAX, V_MIN,
                           RUN_DIR, name_ts, step_delay=step_delay)

    print("[PLOT] Summary figure (2×2)...")
    plot_and_save(plot_voltages, currents, resistance,
                  N_CYCLES, cycle_len, V_MAX, V_MIN,
                  img_path, step_delay=step_delay)


if __name__ == "__main__":
    main()
