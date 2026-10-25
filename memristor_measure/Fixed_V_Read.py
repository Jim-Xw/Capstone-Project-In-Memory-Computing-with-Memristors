"""Continuous current and resistance reads of a memristor at a fixed voltage.

Wiring: Keithley FORCE HI to the device top electrode and FORCE LO to the bottom
electrode; check the polarity and common ground before starting, and never rewire
while the output is on. Parameter/command example: ``python Fixed_V_Read.py --voltage 0.1
--icc 50e-6 --interval 0.1 --duration 60``; ``--duration 0`` runs until Ctrl+C.
Output: point-by-point CSV (original columns kept, with set/measured voltage, Icc,
compliance state and other columns appended) and an optional PNG.
Safety warning: the read voltage may not exceed 2 V in magnitude and Icc is
hard-limited to 50 uA; if compliance is triggered, the resistance is recorded as nan
and the run stops immediately and returns safely to zero. Software protection is
never a substitute for correct wiring.
Series resistor: with ``--r-series <Ohm>``, the resistance is corrected for the series
resistor, a device-voltage column is appended and the output name carries an RS tag.
"""

import argparse
import csv
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
        DEFAULT_MAX_VOLTAGE_V,
        InstrumentConnectionError,
        add_series_argument,
        connect_smu,
        configure_voltage_source,
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
        DEFAULT_MAX_VOLTAGE_V,
        InstrumentConnectionError,
        add_series_argument,
        connect_smu,
        configure_voltage_source,
        device_voltage,
        read_vi,
        safe_shutdown,
        series_tag,
        validate_safety_limits,
        validate_series,
    )

DEFAULT_VISA_ADDRESS = AUTO_ADDRESS

V_READ = 0.1
I_CC = 50e-6
NPLC = 0.5
SAMPLE_INTERVAL = 0.01
DURATION = 5.0
SETTLE_TIME = 1.0
PRE_TIME = 1.0
BASELINE_VOLTAGE = 0.1

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fixed_V_Read")


def connect(address: str):
    try:
        return connect_smu(address)
    except InstrumentConnectionError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


def init_smu(inst, voltage: float, i_cc: float, nplc: float):
    actual_icc = configure_voltage_source(inst, i_cc, nplc)
    time.sleep(0.3)
    print(f"[Init] Fixed voltage mode ready, V={voltage:.6g} V, Icc={actual_icc:.6e} A")
    return actual_icc


def resistance_from(voltage: float, current: float) -> float:
    if not np.isfinite(current) or abs(current) <= 1e-14:
        return float("inf")
    return abs(voltage / current)


def fmt_current(current: float) -> str:
    if not np.isfinite(current):
        return "nan"
    if abs(current) >= 1e-3:
        return f"{current * 1e3:.4f} mA"
    if abs(current) >= 1e-6:
        return f"{current * 1e6:.4f} uA"
    if abs(current) >= 1e-9:
        return f"{current * 1e9:.4f} nA"
    return f"{current * 1e12:.4f} pA"


def fmt_resistance(resistance: float) -> str:
    if not np.isfinite(resistance):
        return "inf Ohm"
    if resistance >= 1e9:
        return f"{resistance / 1e9:.4f} GOhm"
    if resistance >= 1e6:
        return f"{resistance / 1e6:.4f} MOhm"
    if resistance >= 1e3:
        return f"{resistance / 1e3:.4f} kOhm"
    return f"{resistance:.4f} Ohm"


def fmt_series(r_series: float) -> str:
    """Series resistor label for titles/console, e.g. '10 kOhm'."""
    return f"{r_series / 1e3:g} kOhm" if r_series >= 1e3 else f"{r_series:g} Ohm"


def run_monitor(
    inst,
    voltage: float,
    baseline_voltage: float,
    interval: float,
    duration: float,
    settle: float,
    pre_time: float,
    csv_path: str,
    i_cc: float,
    r_series: float = 0.0,
):
    times = []
    currents = []
    resistances = []
    voltages = []
    sample_durations = []
    stopped_reason = ""

    print(f"[Run] Output ON at baseline voltage {baseline_voltage:.6g} V")
    inst.write(f":SOUR:VOLT:LEV {baseline_voltage:.6f}")
    inst.write(":OUTP ON")

    print(f"[Run] Set voltage to {voltage:.6g} V and start sampling immediately")

    print()
    print(f"  {'Index':>7}  {'Phase':>9}  {'Time (s)':>10}  {'V (V)':>9}  {'I':>14}  {'R':>14}")
    print("  " + "-" * 74)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "phase",
            "time_s",
            "voltage_V",
            "current_A",
            "current_uA",
            "resistance_Ohm",
            "voltage_set_V",
            "voltage_actual_V",
            "icc_A",
            "compliance",
            "source_status",
            "stopped_reason",
            "device_voltage_V",
            "r_series_Ohm",
        ])

        index = 0

        def sample(phase: str, t_zero: float, applied_voltage: float):
            nonlocal index, stopped_reason
            read_started = time.perf_counter()
            try:
                reading = read_vi(inst, applied_voltage, i_cc)
            except Exception as e:
                print(f"  [Read warning #{index}] {e}")
                stopped_reason = f"read_error: {e}"
                return False
            elapsed = time.time() - t_zero
            current = reading.current_a
            resistance = reading.resistance_ohm - r_series  # device alone; nan stays nan
            v_for_dev = reading.voltage_actual_v if np.isfinite(reading.voltage_actual_v) else applied_voltage
            v_device = device_voltage(v_for_dev, current, r_series)
            row_reason = "compliance" if reading.compliance else ""
            if reading.compliance:
                stopped_reason = row_reason

            times.append(elapsed)
            voltages.append(reading.voltage_actual_v)
            currents.append(current)
            resistances.append(resistance)

            writer.writerow([
                index,
                phase,
                f"{elapsed:.6f}",
                f"{applied_voltage:.6f}",
                f"{current:.6e}",
                f"{current * 1e6:.6f}" if np.isfinite(current) else "nan",
                f"{resistance:.6e}" if np.isfinite(resistance) else "nan",
                f"{reading.voltage_set_v:.6f}",
                f"{reading.voltage_actual_v:.6f}",
                f"{reading.icc_a:.6e}",
                int(reading.compliance),
                "" if reading.source_status is None else reading.source_status,
                row_reason,
                f"{v_device:.6f}" if np.isfinite(v_device) else "nan",
                f"{r_series:g}",
            ])
            f.flush()

            print(
                f"  {index:7d}  {phase:>9}  {elapsed:10.3f}  {applied_voltage:9.4f}  "
                f"{fmt_current(current):>14}  {fmt_resistance(resistance):>14}"
            )
            index += 1
            sample_durations.append(time.perf_counter() - read_started)
            if reading.compliance:
                print("  [SAFETY STOP] Compliance detected; stopping and returning to zero.")
                safe_shutdown(inst)
                return False
            return True

        if pre_time > 0:
            t_pre_start = time.time()
            next_sample = t_pre_start
            est_baseline_points = max(1, int(np.ceil(pre_time / interval)))
            print(
                f"[Run] Recording {baseline_voltage:.6g} V baseline repeatedly "
                f"for {pre_time:.3f} s, about {est_baseline_points} samples"
            )
            while True:
                now = time.time()
                if now - t_pre_start >= pre_time:
                    break
                if now < next_sample:
                    time.sleep(min(0.01, next_sample - now))
                    continue
                if not sample("baseline", t_pre_start + pre_time, baseline_voltage):
                    break
                next_sample += interval

        if not stopped_reason:
            inst.write(f":SOUR:VOLT:LEV {voltage:.6f}")
            t_start = time.time()
            next_sample = t_start
            total_duration = duration + max(settle, 0.0) if duration > 0 else 0

            while True:
                now = time.time()
                elapsed = now - t_start
                if total_duration > 0 and elapsed >= total_duration:
                    stopped_reason = "duration_complete"
                    break

                if now < next_sample:
                    time.sleep(min(0.01, next_sample - now))
                    continue

                phase = "settling" if elapsed < settle else "read"
                if not sample(phase, t_start, voltage):
                    break
                next_sample += interval

    throughput = float(np.mean(sample_durations)) if sample_durations else 0.0
    return (np.array(times), np.array(voltages), np.array(currents),
            np.array(resistances), stopped_reason, throughput)


def plot_and_save(times, voltages, currents, resistances, voltage: float, baseline_voltage: float, settle: float, img_path: str,
                  r_series: float = 0.0):
    if len(times) == 0:
        print("[PNG] No samples collected, skip plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.patch.set_facecolor("#000000")
    for ax in (ax1, ax2):
        ax.set_facecolor("#000000")
        ax.tick_params(which="both", colors="#FFFFFF", labelsize=11)
        for spine in ax.spines.values():
            spine.set_color("#FFFFFF")
        ax.grid(True, color="#3A3A3A", lw=0.5, which="both")

    ax1.plot(times, currents * 1e6, color="#4DB6AC", lw=1.6)
    ax1.scatter(times, currents * 1e6, color="#80CBC4", s=10, alpha=0.75)
    ax1.axvline(0, color="#777", lw=0.8, linestyle="--")
    if settle > 0:
        ax1.axvline(settle, color="#777", lw=0.8, linestyle=":")
    ax1.axhline(0, color="#888888", lw=0.7)
    ax1.set_ylabel("Current (uA)", color="#FFFFFF")
    ax1.set_title("Current vs. Time", color="#FFFFFF", fontsize=11, pad=8)

    finite_r = np.isfinite(resistances)
    if np.any(finite_r):
        ax2.plot(times[finite_r], resistances[finite_r] / 1e3, color="#FFB74D", lw=1.4)
        ax2.scatter(times[finite_r], resistances[finite_r] / 1e3, color="#FFD54F", s=10, alpha=0.75)
        ax2.set_yscale("log")
    ax2.axvline(0, color="#777", lw=0.8, linestyle="--")
    if settle > 0:
        ax2.axvline(settle, color="#777", lw=0.8, linestyle=":")
    ax2.set_xlabel("Time (s)", color="#FFFFFF")
    ax2.set_ylabel("Resistance (kOhm)", color="#FFFFFF")
    ax2.set_title("Resistance vs. Time", color="#FFFFFF", fontsize=11, pad=8)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    actual_v = voltages[np.isfinite(voltages)]
    v_label = f"{np.nanmin(actual_v):.3g}~{np.nanmax(actual_v):.3g} V" if len(actual_v) else f"{voltage:.6g} V"
    rs_label = f"   Rs={fmt_series(r_series)}" if r_series > 0 else ""
    fig.suptitle(
        f"Fixed Voltage Read   baseline={baseline_voltage:.6g} V, read={voltage:.6g} V   "
        f"V={v_label}{rs_label}   Samples={len(times)}   {ts}",
        color="#FFFFFF", fontsize=10)
    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[PNG] Saved -> {img_path}")
    plt.show()


def make_run_dir(dev: str, voltage: float, i_cc: float, nplc: float, interval: float, duration: float,
                 r_series: float = 0.0):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dur_label = "untilstop" if duration <= 0 else f"{int(duration)}s"
    run_name = (
        f"{dev}_"
        f"V{voltage:+.3f}_"
        f"dt{int(interval * 1000)}ms_"
        f"dur{dur_label}_"
        f"Icc{int(i_cc * 1e6)}uA_"
        f"NPLC{nplc}_"
        f"{series_tag(r_series) + '_' if r_series > 0 else ''}"
        f"{ts_str}"
    )
    run_dir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    if r_series > 0:
        ts_str = f"{series_tag(r_series)}_{ts_str}"  # file names: ..._RS10k_<timestamp>
    return run_dir, ts_str


def main():
    parser = argparse.ArgumentParser(description="Keithley 2450/2461 fixed-voltage continuous current read")
    parser.add_argument("--address", default=DEFAULT_VISA_ADDRESS, help="VISA resource address")
    parser.add_argument("--dev", default="DEV", help="Device label for output folder naming")
    parser.add_argument("--voltage", type=float, default=V_READ, help="Fixed source voltage in V")
    parser.add_argument("--baseline-voltage", type=float, default=BASELINE_VOLTAGE, help="Small read voltage used during --pre-time baseline")
    parser.add_argument("--icc", type=float, default=I_CC, help="Source current compliance in A")
    parser.add_argument("--nplc", type=float, default=NPLC, help="Current measurement NPLC")
    parser.add_argument("--interval", type=float, default=SAMPLE_INTERVAL, help="Sample interval in seconds")
    parser.add_argument("--duration", type=float, default=DURATION, help="Run duration in seconds; 0 means until Ctrl+C")
    parser.add_argument("--settle", type=float, default=SETTLE_TIME, help="Settling window in seconds; data is recorded during this window")
    parser.add_argument("--pre-time", type=float, default=PRE_TIME, help="Baseline sampling time before applying target voltage; sampled repeatedly at --interval")
    parser.add_argument("--no-plot", action="store_true", help="Only save CSV, skip PNG plot")
    add_series_argument(parser)
    args = parser.parse_args()
    validate_series(args.r_series)

    validate_safety_limits(args.voltage, args.icc)
    validate_safety_limits(args.baseline_voltage, args.icc)
    if args.interval <= 0:
        raise ValueError("--interval must be > 0")
    if args.icc <= 0:
        raise ValueError("--icc must be > 0")
    if args.nplc <= 0:
        raise ValueError("--nplc must be > 0")
    if args.settle < 0:
        raise ValueError("--settle must be >= 0")
    if args.pre_time < 0:
        raise ValueError("--pre-time must be >= 0")

    run_dir, ts_str = make_run_dir(args.dev, args.voltage, args.icc, args.nplc, args.interval, args.duration,
                                   args.r_series)
    csv_path = os.path.join(run_dir, f"fixed_v_read_{ts_str}.csv")
    img_path = os.path.join(run_dir, f"fixed_v_read_{ts_str}.png")

    duration_label = "until Ctrl+C" if args.duration <= 0 else f"{args.duration:.1f} s"
    baseline_points = int(np.ceil(args.pre_time / args.interval)) if args.pre_time > 0 else 0
    print("=" * 58)
    print("  Fixed Voltage Current Monitor")
    print("=" * 58)
    print(f"  Device label    = {args.dev}")
    print(f"  Voltage         = {args.voltage:.6g} V")
    print(f"  Baseline V      = {args.baseline_voltage:.6g} V")
    print(f"  I compliance    = {args.icc:.6e} A")
    print(f"  NPLC            = {args.nplc}")
    print(f"  Sample interval = {args.interval:.3f} s")
    print(f"  Duration        = {duration_label}")
    print(f"  Settling window = {args.settle:.3f} s  (recorded, not skipped)")
    print(f"  Baseline time   = {args.pre_time:.3f} s")
    print(f"  Baseline points = about {baseline_points}")
    if args.r_series > 0:
        print(f"  Series R        = {fmt_series(args.r_series)}  (subtracted from R; device voltage logged)")
    print(f"  Output dir      = {run_dir}")
    print("=" * 58)

    ans = input("\nConfirm start? (Enter / q to quit): ").strip().lower()
    if ans == "q":
        return

    inst = connect(args.address)
    times = np.array([])
    voltages = np.array([])
    currents = np.array([])
    resistances = np.array([])
    stopped_reason = ""
    actual_throughput = 0.0

    try:
        actual_icc = init_smu(inst, args.voltage, args.icc, args.nplc)
        times, voltages, currents, resistances, stopped_reason, actual_throughput = run_monitor(
            inst,
            args.voltage,
            args.baseline_voltage,
            args.interval,
            args.duration,
            args.settle,
            args.pre_time,
            csv_path,
            actual_icc,
            args.r_series,
        )
    except KeyboardInterrupt:
        print("\n[Stop] Ctrl+C received, stopping measurement.")
        stopped_reason = "keyboard_interrupt"
    finally:
        try:
            safe_shutdown(inst)
            print("[Safe] Output OFF, voltage returned to 0 V.")
        finally:
            inst.close()

    print(f"\n[CSV] Saved -> {csv_path}")
    if stopped_reason:
        print(f"[Stop reason] {stopped_reason}")
    if actual_throughput > args.interval:
        print(
            f"[WARNING] Requested interval={args.interval:.6g} s is shorter than the actual "
            f"single-read time of about {actual_throughput:.6g} s; samples will be further apart."
        )
    if len(currents):
        finite_i = currents[np.isfinite(currents)]
        finite_r = resistances[np.isfinite(resistances)]
        print(f"[Summary] Samples: {len(currents)}")
        if len(finite_i):
            print(f"[Summary] Current mean: {fmt_current(float(np.mean(finite_i)))}")
            print(f"[Summary] Current min/max: {fmt_current(float(np.min(finite_i)))} / {fmt_current(float(np.max(finite_i)))}")
        if len(finite_r):
            print(f"[Summary] Resistance median: {fmt_resistance(float(np.median(finite_r)))}")

    if not args.no_plot:
        plot_and_save(times, voltages, currents, resistances, args.voltage, args.baseline_voltage, args.settle, img_path,
                      args.r_series)


if __name__ == "__main__":
    main()
