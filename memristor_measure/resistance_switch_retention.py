"""Resistance switching and retention experiment for memristors.

Purpose: each cycle applies one SET and one RESET write pulse and, after each write,
reads the resistance state repeatedly with a small signal to observe retention.
Wiring: connect the device to FORCE HI and FORCE LO on the Keithley front panel and
select the FRONT terminals; SENSE must not be used in place of FORCE. Software mode
is timed by the host, with a minimum pulse width of 20 ms; hardware mode uses the
internal trigger of the 2461, which gives more accurate timing and is not supported
on the 2450. Example: ``python memristor_measure/resistance_switch_retention.py
--pulse-mode software --v-set 0.6 --v-reset -0.6 --write-relax 0.05``.
The CSV keeps the original column order and appends the write-pulse set and readback
voltage, current, compliance, mode, source status and stop reason; the JSON keeps the
original fields and appends the new parameters and safety limits. Safety warning: the
default Icc is 50 uA and can be raised with --icc; --icc-set/--icc-reset set the
compliance of the SET and RESET write pulses separately (default: same as --icc; reads
always use --icc); no voltage may exceed 2 V in magnitude. When a write pulse reaches
compliance the reads continue (normal SET behaviour); an overcurrent or an incomplete
pulse stops the run immediately, returns the output to zero and switches it off.
Series resistor: with ``--r-series <Ohm>``, R and G are corrected for the series
resistor, a device-voltage column is appended and the output name carries an RS tag.
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
        add_series_argument,
        apply_voltage_pulse,
        configure_voltage_source,
        connect_smu,
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
        set_current_limit,
        validate_safety_limits,
        validate_series,
    )

DEFAULT_VISA_ADDRESS = AUTO_ADDRESS
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Switch_Retention")

V_READ = 0.1
V_SET = 0.6
V_RESET = -0.6
T_SET = 0.02
T_RESET = 0.02
WRITE_RELAX = 0.05
READ_INTERVAL = 0.2
RETENTION_TIME = 5.0
BASELINE_TIME = 1.0
I_CC = MAX_SAFE_CURRENT_A
NPLC = 0.5
N_READ_AVG = 3
READ_SETTLE = 0.02
R_SERIES = 0.0


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


def fmt_series(r_series: float) -> str:
    """Series resistor label for titles/console, e.g. '10 kOhm'."""
    return f"{r_series / 1e3:g} kOhm" if r_series >= 1e3 else f"{r_series:g} Ohm"


def write_failed(pulse, icc: float) -> bool:
    """Abort on overcurrent or an incomplete pulse; reaching compliance is allowed.

    A SET is expected to reach its compliance; the pulse is then cut short and
    the state is verified by the following read.
    """
    overcurrent = np.isfinite(pulse.current_a) and abs(pulse.current_a) > icc * 1.02
    return overcurrent or (not pulse.completed and not pulse.compliance)


def read_resistance(inst, v_read: float, icc: float, n_avg: int,
                    settle: float, r_series: float = R_SERIES):
    currents = []
    compliance = False
    # Write pulses may leave a higher limit on the instrument; restore it.
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


def append_sample(records, cycle, phase, elapsed, v_write, v_read, current,
                  resistance, conductance, pulse=None, r_series: float = R_SERIES):
    if pulse is None:
        pulse_v_dev = float("nan")
    else:
        pulse_v = pulse.voltage_actual_v if np.isfinite(pulse.voltage_actual_v) else pulse.voltage_set_v
        pulse_v_dev = device_voltage(pulse_v, pulse.current_a, r_series)
    records.append(
        {
            "index": len(records),
            "cycle": cycle,
            "phase": phase,
            "time_s": elapsed,
            "write_voltage_V": v_write,
            "read_voltage_V": v_read,
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
            "device_voltage_V": device_voltage(v_read, current, r_series),
            "pulse_device_voltage_V": pulse_v_dev,
        }
    )


def sample_for_duration(inst, records, cycle, phase, duration, interval,
                        v_read, icc, n_avg, settle, t_zero, v_write=0.0,
                        r_series: float = R_SERIES):
    end_time = time.time() + max(duration, 0.0)
    next_sample = time.time()
    while True:
        now = time.time()
        if now >= end_time and len(records) > 0 and records[-1]["phase"] == phase:
            break
        if now < next_sample:
            time.sleep(min(0.01, next_sample - now))
            continue

        current, resistance, conductance = read_resistance(
            inst, v_read, icc, n_avg, settle, r_series
        )
        elapsed = time.time() - t_zero
        append_sample(records, cycle, phase, elapsed, v_write, v_read, current, resistance, conductance,
                      r_series=r_series)
        print(
            f"  {len(records)-1:5d}  cycle={cycle:02d}  {phase:>15}  "
            f"t={elapsed:8.3f}s  R={fmt_resistance(resistance):>14}  G={conductance*1e6:9.3f} uS"
        )
        next_sample += interval
        if duration <= 0:
            break


def save_csv(records, csv_path: str):
    columns = [
        "index",
        "cycle",
        "phase",
        "time_s",
        "write_voltage_V",
        "read_voltage_V",
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
        "device_voltage_V",
        "pulse_device_voltage_V",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in records:
            out = dict(row)
            for key in ("time_s", "write_voltage_V", "read_voltage_V",
                        "current_A", "resistance_Ohm", "conductance_S",
                        "pulse_voltage_set_V", "pulse_voltage_actual_V",
                        "pulse_current_A", "pulse_relative_time_s",
                        "device_voltage_V", "pulse_device_voltage_V"):
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
    payload["max_safe_current_A"] = MAX_SAFE_CURRENT_A
    payload["r_series_Ohm"] = getattr(args, "r_series", R_SERIES)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[JSON] Saved -> {meta_path}")


def plot_records(records, img_path: str, title: str):
    if not records:
        return

    times = np.array([r["time_s"] for r in records], dtype=float)
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

    colors = {
        "baseline": "#B0BEC5",
        "after_set": "#4DB6AC",
        "set_retention": "#26A69A",
        "after_reset": "#FFB74D",
        "reset_retention": "#FF9800",
    }
    for phase in sorted(set(phases)):
        mask = np.array([p == phase for p in phases])
        finite_r = np.isfinite(resistances) & mask
        ax_r.plot(times[finite_r], resistances[finite_r], "o-", ms=4, lw=1.2, color=colors.get(phase, "#90CAF9"), label=phase)
        ax_g.plot(times[mask], conductances[mask] * 1e6, "o-", ms=4, lw=1.2, color=colors.get(phase, "#90CAF9"), label=phase)

    if np.any(np.isfinite(resistances)):
        ax_r.set_yscale("log")
    ax_r.set_ylabel("Resistance (Ohm)", color="#FFFFFF")
    ax_r.set_title("Resistance retention", color="#FFFFFF", fontsize=11)
    ax_r.legend(facecolor="#000000", edgecolor="#FFFFFF", labelcolor="#FFFFFF", fontsize=8)

    ax_g.set_xlabel("Time (s)", color="#FFFFFF")
    ax_g.set_ylabel("Conductance (uS)", color="#FFFFFF")
    ax_g.set_title("Conductance change", color="#FFFFFF", fontsize=11)

    fig.suptitle(title, color="#FFFFFF", fontsize=10)
    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[PNG] Saved -> {img_path}")
    plt.show()


def make_run_dir(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.r_series > 0:
        ts = f"{series_tag(args.r_series)}_{ts}"  # folder and file names: ..._RS10k_<timestamp>
    run_name = (
        f"{args.dev}_SET{args.v_set:+.2f}_RESET{args.v_reset:+.2f}_"
        f"read{args.v_read:+.2f}_cyc{args.cycles}_{ts}"
    )
    run_dir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, ts


def main():
    parser = argparse.ArgumentParser(description="Memristor resistance switching and retention test")
    parser.add_argument("--address", default=DEFAULT_VISA_ADDRESS, help="VISA resource address")
    parser.add_argument("--dev", default="DEV", help="Device label for output folder naming")
    parser.add_argument("--v-read", type=float, default=V_READ, help="Non-destructive read voltage in V")
    parser.add_argument("--v-set", type=float, default=V_SET, help="SET write voltage in V")
    parser.add_argument("--v-reset", type=float, default=V_RESET, help="RESET write voltage in V")
    parser.add_argument("--t-set", type=float, default=T_SET, help="SET write duration in seconds")
    parser.add_argument("--t-reset", type=float, default=T_RESET, help="RESET write duration in seconds")
    parser.add_argument("--write-relax", type=float, default=WRITE_RELAX,
                        help="Independent relaxation time after each write pulse")
    parser.add_argument("--baseline-time", type=float, default=BASELINE_TIME, help="Baseline read duration in seconds")
    parser.add_argument("--retention-time", type=float, default=RETENTION_TIME, help="Retention read duration after each write")
    parser.add_argument("--interval", type=float, default=READ_INTERVAL, help="Read sample interval in seconds")
    parser.add_argument("--icc", type=float, default=I_CC, help="Source current compliance in A (reads, and writes unless overridden)")
    parser.add_argument("--icc-set", type=float, default=None, help="Compliance for the SET write in A (default: --icc)")
    parser.add_argument("--icc-reset", type=float, default=None, help="Compliance for the RESET write in A (default: --icc)")
    parser.add_argument("--nplc", type=float, default=NPLC, help="Current measurement NPLC")
    parser.add_argument("--n-read-avg", type=int, default=N_READ_AVG, help="Number of read samples to average")
    parser.add_argument("--read-settle", type=float, default=READ_SETTLE, help="Settling time before each read in seconds")
    parser.add_argument("--cycles", type=int, default=1, help="Number of SET/RESET cycles")
    parser.add_argument("--pulse-mode", choices=("software", "hardware"),
                        default="software", help="Pulse mode (default: software)")
    parser.add_argument("--no-plot", action="store_true", help="Save CSV only")
    add_series_argument(parser)
    args = parser.parse_args()
    validate_series(args.r_series)
    if args.icc_set is None:
        args.icc_set = args.icc
    if args.icc_reset is None:
        args.icc_reset = args.icc

    if args.interval <= 0:
        raise ValueError("--interval must be > 0")
    for voltage in (args.v_read, args.v_set, args.v_reset):
        validate_safety_limits(voltage, args.icc)
    validate_safety_limits(args.v_set, args.icc_set)
    validate_safety_limits(args.v_reset, args.icc_reset)
    if args.t_set <= 0 or args.t_reset <= 0:
        raise ValueError("--t-set and --t-reset must be > 0")
    if args.pulse_mode == "software" and min(args.t_set, args.t_reset) < 20e-3:
        raise ValueError("software pulse widths must be >= 0.02 s")
    if args.write_relax < 0:
        raise ValueError("--write-relax must be >= 0")
    if args.cycles <= 0:
        raise ValueError("--cycles must be > 0")
    if args.n_read_avg <= 0:
        raise ValueError("--n-read-avg must be > 0")

    run_dir, ts = make_run_dir(args)
    csv_path = os.path.join(run_dir, f"switch_retention_{ts}.csv")
    img_path = os.path.join(run_dir, f"switch_retention_{ts}.png")
    meta_path = os.path.join(run_dir, f"switch_retention_{ts}.json")

    print("=" * 68)
    print("  Resistance Switching and Retention Test")
    print("=" * 68)
    print(f"  Device       : {args.dev}")
    print(f"  SET          : {args.v_set:+.4g} V for {args.t_set:.3f} s")
    print(f"  RESET        : {args.v_reset:+.4g} V for {args.t_reset:.3f} s")
    print(f"  Read         : {args.v_read:+.4g} V, avg={args.n_read_avg}, interval={args.interval:.3f} s")
    print(f"  Retention    : {args.retention_time:.3f} s after each write")
    print(f"  Write relax  : {args.write_relax:.3f} s")
    print(f"  Pulse mode   : {args.pulse_mode}")
    print(f"  Cycles       : {args.cycles}")
    print(f"  Compliance   : read {args.icc:.3e} A, SET {args.icc_set:.3e} A, RESET {args.icc_reset:.3e} A")
    if args.r_series > 0:
        print(f"  Series R     : {fmt_series(args.r_series)} (subtracted from R; device voltage logged)")
    print(f"  Output       : {run_dir}")
    print("=" * 68)
    ans = input("\nConfirm start? (Enter / q to quit): ").strip().lower()
    if ans == "q":
        return

    records = []
    inst = connect(args.address)
    t_zero = time.time()
    try:
        init_smu(inst, args.icc, args.nplc)
        sample_for_duration(
            inst, records, 0, "baseline", args.baseline_time, args.interval,
            args.v_read, args.icc, args.n_read_avg, args.read_settle, t_zero,
            r_series=args.r_series,
        )

        for cycle in range(1, args.cycles + 1):
            print(f"\n[Cycle {cycle}] SET write {args.v_set:+.4g} V, {args.t_set:.3f} s")
            pulse = apply_voltage_pulse(
                inst, mode=args.pulse_mode, voltage_v=args.v_set,
                width_s=args.t_set, relax_s=args.write_relax,
                current_limit_a=args.icc_set,
            )
            if write_failed(pulse, args.icc_set):
                safe_shutdown(inst)
                append_sample(
                    records, cycle, "after_set", time.time() - t_zero,
                    args.v_set, args.v_read, float("nan"), float("nan"),
                    float("nan"), pulse, args.r_series,
                )
                print(f"[Safety stop] {pulse.stopped_reason or 'pulse_incomplete'}")
                break
            current, resistance, conductance = read_resistance(
                inst, args.v_read, args.icc, args.n_read_avg,
                args.read_settle, args.r_series
            )
            append_sample(records, cycle, "after_set", time.time() - t_zero, args.v_set, args.v_read, current, resistance, conductance, pulse,
                          args.r_series)
            print(f"  after SET   R={fmt_resistance(resistance):>14}  G={conductance*1e6:9.3f} uS")
            sample_for_duration(
                inst, records, cycle, "set_retention", args.retention_time, args.interval,
                args.v_read, args.icc, args.n_read_avg, args.read_settle,
                t_zero, args.v_set, args.r_series
            )

            print(f"\n[Cycle {cycle}] RESET write {args.v_reset:+.4g} V, {args.t_reset:.3f} s")
            pulse = apply_voltage_pulse(
                inst, mode=args.pulse_mode, voltage_v=args.v_reset,
                width_s=args.t_reset, relax_s=args.write_relax,
                current_limit_a=args.icc_reset,
            )
            if write_failed(pulse, args.icc_reset):
                safe_shutdown(inst)
                append_sample(
                    records, cycle, "after_reset", time.time() - t_zero,
                    args.v_reset, args.v_read, float("nan"), float("nan"),
                    float("nan"), pulse, args.r_series,
                )
                print(f"[Safety stop] {pulse.stopped_reason or 'pulse_incomplete'}")
                break
            current, resistance, conductance = read_resistance(
                inst, args.v_read, args.icc, args.n_read_avg,
                args.read_settle, args.r_series
            )
            append_sample(records, cycle, "after_reset", time.time() - t_zero, args.v_reset, args.v_read, current, resistance, conductance, pulse,
                          args.r_series)
            print(f"  after RESET R={fmt_resistance(resistance):>14}  G={conductance*1e6:9.3f} uS")
            sample_for_duration(
                inst, records, cycle, "reset_retention", args.retention_time, args.interval,
                args.v_read, args.icc, args.n_read_avg, args.read_settle,
                t_zero, args.v_reset, args.r_series
            )
    except KeyboardInterrupt:
        print("\n[Stop] Ctrl+C received, saving collected data.")
    finally:
        safe_shutdown(inst)
        print("[Safe] Output OFF, voltage returned to 0 V.")
        inst.close()

    save_csv(records, csv_path)
    save_metadata(args, meta_path)

    finite_r = np.array([r["resistance_Ohm"] for r in records if np.isfinite(r["resistance_Ohm"])], dtype=float)
    if len(finite_r):
        print(f"[Summary] R range: {fmt_resistance(float(np.min(finite_r)))} to {fmt_resistance(float(np.max(finite_r)))}")
        if np.min(finite_r) > 0:
            print(f"[Summary] Rmax/Rmin: {float(np.max(finite_r) / np.min(finite_r)):.3f}x")

    if not args.no_plot:
        title = f"{args.dev} switching retention: SET {args.v_set:+.2f} V, RESET {args.v_reset:+.2f} V"
        if args.r_series > 0:
            title += f", Rs={fmt_series(args.r_series)}"
        plot_records(records, img_path, title)


if __name__ == "__main__":
    main()

