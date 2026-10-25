"""Closed-loop SET/RESET pulses that keep R where the full write voltage can be applied.

Purpose: changes the resistance state step by step with SET pulses of about +1 V and
RESET pulses of about -1 V. Before each write the resistance is read at 0.1 V to
estimate the write current. If |V|/R would reach Icc, the SET direction stops or
switches to RESET, while the RESET direction only stops (a further SET at this point
would make the filament thicker). Wiring: FORCE HI/LO on the Keithley front panel,
FRONT terminals selected. Software mode has a minimum pulse width of 20 ms; hardware
mode is available on the 2461 only. Example:
``python memristor_measure/pulse_set_reset.py --mode cycle --n-set 4 --n-reset 4``.
Safety: default Icc 50 uA and |V| <= 2 V; the run stops immediately and returns to
zero on an overcurrent or when the full RESET voltage cannot be applied.
Series resistor: with ``--r-series <Ohm>``, the resistance and the decision criteria
refer to the device itself (reading minus Rs) and the file name carries an RS tag.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Literal

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
        PulseResult,
        add_series_argument,
        apply_voltage_pulse,
        configure_voltage_source,
        connect_smu,
        device_resistance,
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
        PulseResult,
        add_series_argument,
        apply_voltage_pulse,
        configure_voltage_source,
        connect_smu,
        device_resistance,
        device_voltage,
        read_vi,
        safe_shutdown,
        series_tag,
        validate_safety_limits,
        validate_series,
    )

DEFAULT_VISA_ADDRESS = AUTO_ADDRESS
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pulse_Set_Reset")

V_READ = 0.1
V_SET = 1.0
V_RESET = -1.0
T_PULSE = 0.02
T_RELAX = 0.05
N_SET = 4
N_RESET = 4
N_CYCLES = 1
I_CC = 50e-6
NPLC = 0.1
N_READ_AVG = 20
READ_SETTLE = 0.02
GUARD_RATIO = 0.85
R_MARGIN = 1.2
V_DROP_RATIO = 0.90
MAX_GUARD = 4

Action = Literal["set", "reset", "stop"]


def r_floor_ohm(v_write: float, icc: float, margin: float) -> float:
    """Lowest read resistance at which the full |V| can be applied under Icc."""
    return abs(v_write) / icc * margin


def predicted_write_current(r_read: float, v_write: float, r_series: float = 0.0) -> float:
    # r_read is the device resistance. The series resistor also limits the current, so
    # the write current is estimated from the whole loop, R_dev + Rs.
    r_total = r_read + r_series
    if not math.isfinite(r_total) or r_total <= 0:
        return float("inf")
    return abs(v_write) / r_total


def write_would_hit_icc(
    r_read: float,
    v_write: float,
    icc: float,
    *,
    guard_ratio: float = GUARD_RATIO,
    margin: float = R_MARGIN,
    r_series: float = 0.0,
) -> bool:
    """True if R read at 0.1 V is so low that |V|/R approaches Icc, in which case the
    write pulse cannot reach its full voltage."""
    # Icc limits the current through the whole loop, so R_dev + Rs is compared with the limit.
    r_total = r_read + r_series
    if not math.isfinite(r_read) or r_total <= 0:
        return True
    if r_total <= r_floor_ohm(v_write, icc, margin):
        return True
    return predicted_write_current(r_read, v_write, r_series) >= icc * guard_ratio


def pulse_lost_voltage(pulse: PulseResult, v_cmd: float, drop_ratio: float = V_DROP_RATIO) -> bool:
    """Readback voltage has collapsed and the current is close to Icc, so the write
    voltage is taken as not applied."""
    if pulse.compliance:
        return True
    if not math.isfinite(pulse.voltage_actual_v) or not math.isfinite(pulse.current_a):
        return False
    return (
        abs(pulse.voltage_actual_v) < abs(v_cmd) * drop_ratio
        and abs(pulse.current_a) >= pulse.current_limit_a * GUARD_RATIO
    )


def decide_next_action(
    planned: Action,
    r_read: float,
    v_set: float,
    v_reset: float,
    icc: float,
    *,
    on_limit: str = "reverse",
    guard_ratio: float = GUARD_RATIO,
    margin: float = R_MARGIN,
    r_min: float | None = None,
    r_max: float | None = None,
    r_series: float = 0.0,
) -> tuple[Action, str]:
    """Choose the next step from the present read resistance: set / reset / stop."""
    if planned not in ("set", "reset"):
        return "stop", "invalid_planned"

    if r_min is not None and math.isfinite(r_read) and planned == "set" and r_read <= r_min:
        reset_blocked = write_would_hit_icc(
            r_read, v_reset, icc, guard_ratio=1.0, margin=1.0, r_series=r_series
        )
        if on_limit == "reverse" and not reset_blocked:
            return "reset", "below_r_min"
        return "stop", "below_r_min"

    if r_max is not None and math.isfinite(r_read) and planned == "reset" and r_read >= r_max:
        return "stop", "reached_r_max"

    v_cmd = v_set if planned == "set" else v_reset
    if not write_would_hit_icc(
        r_read, v_cmd, icc, guard_ratio=guard_ratio, margin=margin, r_series=r_series
    ):
        return planned, "ok"

    if planned == "set":
        reset_blocked = write_would_hit_icc(
            r_read, v_reset, icc, guard_ratio=1.0, margin=1.0, r_series=r_series
        )
        if on_limit == "reverse" and not reset_blocked:
            return "reset", "set_would_exceed_icc"
        return "stop", "set_would_exceed_icc"

    return "stop", "reset_cannot_apply_full_voltage"


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


def resistance_from(voltage: float, current: float, r_series: float = 0.0) -> float:
    if not np.isfinite(current) or abs(current) <= 1e-14:
        return float("inf")
    return device_resistance(voltage, current, r_series)


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


def read_resistance(inst, v_read: float, icc: float, n_avg: int, settle: float,
                    r_series: float = 0.0):
    currents = []
    compliance = False
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
    resistance = resistance_from(v_read, current, r_series)
    conductance = 0.0 if not np.isfinite(resistance) or resistance <= 0 else 1.0 / resistance
    return current, resistance, conductance


def add_record(records, cycle, phase, pulse_index, global_pulse, elapsed,
               v_pulse, current, resistance, conductance, pulse=None,
               planned="", action="", guard_reason="", i_pred=float("nan"),
               r_floor=float("nan"), r_series=0.0):
    if pulse is None:
        pulse_device_v = float("nan")
    else:
        v_pulse_dev = (
            pulse.voltage_actual_v if math.isfinite(pulse.voltage_actual_v)
            else pulse.voltage_set_v
        )
        pulse_device_v = device_voltage(v_pulse_dev, pulse.current_a, r_series)
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
            "planned": planned,
            "action": action,
            "guard_reason": guard_reason,
            "i_pred_A": i_pred,
            "r_floor_Ohm": r_floor,
            "pulse_device_voltage_V": pulse_device_v,
        }
    )


def save_csv(records, csv_path: str):
    columns = [
        "index", "cycle", "phase", "pulse_index", "global_pulse", "time_s",
        "pulse_voltage_V", "current_A", "resistance_Ohm", "conductance_S",
        "pulse_voltage_set_V", "pulse_voltage_actual_V", "pulse_current_A",
        "compliance", "pulse_mode", "source_status", "stopped_reason",
        "completed", "pulse_relative_time_s", "planned", "action",
        "guard_reason", "i_pred_A", "r_floor_Ohm", "pulse_device_voltage_V",
    ]
    float_keys = (
        "time_s", "pulse_voltage_V", "current_A", "resistance_Ohm",
        "conductance_S", "pulse_voltage_set_V", "pulse_voltage_actual_V",
        "pulse_current_A", "pulse_relative_time_s", "i_pred_A", "r_floor_Ohm",
        "pulse_device_voltage_V",
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in records:
            out = dict(row)
            for key in float_keys:
                value = out.get(key, "")
                if isinstance(value, float):
                    out[key] = (
                        f"{value:.8e}" if np.isfinite(value)
                        else ("nan" if np.isnan(value) else "inf")
                    )
            writer.writerow(out)
    print(f"[CSV] Saved -> {csv_path}")


def save_metadata(args, meta_path: str, extra: dict):
    payload = vars(args).copy()
    payload["created_at"] = datetime.now().isoformat(timespec="seconds")
    payload["note"] = "Closed-loop SET/RESET with Icc voltage-window guard."
    payload["max_safe_current_A"] = MAX_SAFE_CURRENT_A
    payload.update(extra)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[JSON] Saved -> {meta_path}")


def plot_records(records, img_path: str, title: str):
    if not records:
        return
    x = np.array([r["global_pulse"] for r in records], dtype=float)
    resistances = np.array([r["resistance_Ohm"] for r in records], dtype=float)
    phases = [r["phase"] for r in records]
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    ax.tick_params(which="both", colors="#FFFFFF", labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#FFFFFF")
    ax.grid(True, color="#3A3A3A", lw=0.5)
    colors = {
        "baseline": "#B0BEC5",
        "set": "#4DB6AC",
        "reset": "#FFB74D",
        "guard_reset": "#EF9A9A",
        "guard_set": "#90CAF9",
    }
    for phase in sorted(set(phases)):
        mask = np.array([p == phase for p in phases]) & np.isfinite(resistances)
        ax.plot(
            x[mask], resistances[mask], "o-", ms=4, lw=1.2,
            color=colors.get(phase, "#90CAF9"), label=phase,
        )
    if np.any(np.isfinite(resistances)):
        ax.set_yscale("log")
    ax.set_xlabel("Pulse number", color="#FFFFFF")
    ax.set_ylabel("Resistance (Ohm)", color="#FFFFFF")
    ax.set_title(title, color="#FFFFFF", fontsize=11)
    ax.legend(facecolor="#000000", edgecolor="#FFFFFF", labelcolor="#FFFFFF", fontsize=8)
    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[PNG] Saved -> {img_path}")
    plt.show()


def series_suffix(r_series: float) -> str:
    """RS tag inserted before the timestamp in file names; empty with no series resistor."""
    tag = series_tag(r_series)
    return f"{tag}_" if tag else ""


def make_run_dir(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{args.dev}_{args.mode}_SET{args.v_set:+.2f}_RST{args.v_reset:+.2f}_"
        f"icc{args.icc*1e6:.0f}uA_{series_suffix(args.r_series)}{ts}"
    )
    run_dir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, ts


def apply_one_write(inst, args, voltage: float):
    pulse = apply_voltage_pulse(
        inst, mode=args.pulse_mode, voltage_v=voltage,
        width_s=args.t_pulse, relax_s=args.t_relax,
        current_limit_a=args.icc,
    )
    overcurrent = (
        np.isfinite(pulse.current_a)
        and abs(pulse.current_a) > args.icc * 1.02
    )
    lost_v = pulse_lost_voltage(pulse, voltage, args.v_drop_ratio)
    if overcurrent or (not pulse.completed and not pulse.compliance):
        current = resistance = conductance = float("nan")
    else:
        current, resistance, conductance = read_resistance(
            inst, args.v_read, args.icc, args.n_read_avg, args.read_settle,
            args.r_series,
        )
    return pulse, current, resistance, conductance, overcurrent, lost_v


def run_closed_loop(inst, args, records, t_zero, r_min: float, r_max: float):
    current, resistance, conductance = read_resistance(
        inst, args.v_read, args.icc, args.n_read_avg, args.read_settle,
        args.r_series,
    )
    add_record(
        records, 0, "baseline", 0, 0, time.time() - t_zero,
        0.0, current, resistance, conductance,
        action="read", planned="baseline",
        r_floor=r_min,
    )
    print(f"\n[Baseline] R={fmt_resistance(resistance)}  G={conductance*1e6:.3f} uS")
    print(f"[Guard] R_min={fmt_resistance(r_min)}  R_max={fmt_resistance(r_max)}  "
          f"on-limit={args.on_limit}")

    if args.mode == "set":
        plan = [("set", args.n_set)]
        cycles = 1
    elif args.mode == "reset":
        plan = [("reset", args.n_reset)]
        cycles = 1
    elif args.mode == "hold":
        plan = [("hold", args.n_hold)]
        cycles = 1
    else:
        plan = [("set", args.n_set), ("reset", args.n_reset)]
        cycles = args.cycles

    global_pulse = 0
    guard_used = 0
    stop_reason = ""

    def do_pulse(cycle, planned, idx, resistance):
        nonlocal global_pulse, guard_used, stop_reason
        action, reason = decide_next_action(
            planned, resistance, args.v_set, args.v_reset, args.icc,
            on_limit=args.on_limit, guard_ratio=args.guard_ratio,
            margin=args.r_margin, r_min=r_min, r_max=r_max,
            r_series=args.r_series,
        )
        v_for_pred = args.v_set if planned == "set" else args.v_reset
        i_pred = predicted_write_current(resistance, v_for_pred, args.r_series)
        if action == "stop":
            stop_reason = reason
            print(f"[Guard stop] {reason}  R={fmt_resistance(resistance)}  "
                  f"I_pred={i_pred*1e6:.2f} uA")
            return resistance, True

        guarded = action != planned
        if guarded:
            guard_used += 1
            if guard_used > args.max_guard:
                stop_reason = "max_guard_pulses"
                print(f"[Guard stop] reversed {args.max_guard} times, stopping.")
                return resistance, True
            print(f"[Guard] planned {planned} -> {action} ({reason})")

        voltage = args.v_set if action == "set" else args.v_reset
        pulse, current, new_r, conductance, overcurrent, lost_v = apply_one_write(
            inst, args, voltage
        )
        global_pulse += 1
        phase = action if not guarded else f"guard_{action}"
        add_record(
            records, cycle, phase, idx, global_pulse, time.time() - t_zero,
            voltage, current, new_r, conductance, pulse,
            planned=planned, action=action, guard_reason=reason,
            i_pred=i_pred, r_floor=r_min, r_series=args.r_series,
        )
        note = ""
        if pulse.compliance or lost_v:
            note = "  [compliance/Vdrop]"
        print(
            f"  pulse={global_pulse:04d}  {phase:>12} #{idx:03d}  "
            f"R={fmt_resistance(new_r):>14}  "
            f"Vact={pulse.voltage_actual_v:+.3f} V  "
            f"I={pulse.current_a*1e6:7.2f} uA{note}"
        )
        if overcurrent or (not pulse.completed and not pulse.compliance):
            stop_reason = pulse.stopped_reason or "pulse_incomplete"
            safe_shutdown(inst)
            print(f"[Safety stop] {stop_reason}")
            return new_r, True

        if action == "reset" and (pulse.compliance or lost_v):
            stop_reason = "reset_voltage_clamped"
            print("[Guard stop] RESET did not reach full voltage. Stop to avoid thicker filament.")
            return new_r, True

        if action == "set" and (pulse.compliance or lost_v or (
            math.isfinite(new_r) and new_r <= r_min
        )):
            if args.on_limit == "reverse" and guard_used < args.max_guard:
                print("[Guard] SET hit Icc or R_min; applying one RESET")
                return do_pulse(cycle, "reset", idx, new_r if math.isfinite(new_r) else resistance)
            stop_reason = "set_hit_icc"
            print("[Guard stop] SET hit compliance / R_min")
            return new_r, True

        return new_r if math.isfinite(new_r) else resistance, False

    try:
        for cycle in range(1, cycles + 1):
            for planned, count in plan:
                if planned == "hold":
                    for idx in range(1, count + 1):
                        if math.isfinite(resistance) and r_min < resistance < r_max:
                            print(f"[Hold] R={fmt_resistance(resistance)} inside window, stop.")
                            stop_reason = "in_window"
                            return stop_reason, global_pulse
                        hold_plan = "reset" if (
                            not math.isfinite(resistance) or resistance <= r_min
                        ) else "set"
                        resistance, stopped = do_pulse(cycle, hold_plan, idx, resistance)
                        if stopped:
                            return stop_reason, global_pulse
                    continue
                print(
                    f"\n[Cycle {cycle}] planned {planned}: "
                    f"V={(args.v_set if planned == 'set' else args.v_reset):+.4g} V, "
                    f"count={count}"
                )
                for idx in range(1, count + 1):
                    resistance, stopped = do_pulse(cycle, planned, idx, resistance)
                    if stopped:
                        return stop_reason, global_pulse
    except KeyboardInterrupt:
        print("\n[Stop] Ctrl+C received, saving collected data.")
        stop_reason = stop_reason or "keyboard"
    return stop_reason, global_pulse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Closed-loop memristor SET/RESET with Icc voltage-window guard"
    )
    parser.add_argument("--address", default=DEFAULT_VISA_ADDRESS)
    parser.add_argument("--dev", default="DEV")
    parser.add_argument("--mode", choices=("cycle", "set", "reset", "hold"), default="cycle",
                        help="cycle=SET then RESET; set/reset=one direction; hold=keep R in window")
    parser.add_argument("--v-read", type=float, default=V_READ)
    parser.add_argument("--v-set", type=float, default=V_SET)
    parser.add_argument("--v-reset", type=float, default=V_RESET)
    parser.add_argument("--t-pulse", type=float, default=T_PULSE)
    parser.add_argument("--t-relax", type=float, default=T_RELAX)
    parser.add_argument("--n-set", type=int, default=N_SET)
    parser.add_argument("--n-reset", type=int, default=N_RESET)
    parser.add_argument("--n-hold", type=int, default=8, help="Max pulses in hold mode")
    parser.add_argument("--cycles", type=int, default=N_CYCLES)
    parser.add_argument("--icc", type=float, default=I_CC)
    parser.add_argument("--nplc", type=float, default=NPLC)
    parser.add_argument("--n-read-avg", type=int, default=N_READ_AVG)
    parser.add_argument("--read-settle", type=float, default=READ_SETTLE)
    parser.add_argument("--pulse-mode", choices=("software", "hardware"), default="software")
    parser.add_argument("--on-limit", choices=("reverse", "stop"), default="reverse",
                        help="When SET would exceed Icc: apply RESET, or stop")
    parser.add_argument("--guard-ratio", type=float, default=GUARD_RATIO,
                        help="Treat I_pred >= ratio*Icc as about to clip")
    parser.add_argument("--r-margin", type=float, default=R_MARGIN,
                        help="R_min = |V|/Icc * margin")
    parser.add_argument("--r-min", type=float, default=None,
                        help="Stop SET below this 0.1 V resistance (Ohm)")
    parser.add_argument("--r-max", type=float, default=1e6,
                        help="Stop RESET above this 0.1 V resistance (Ohm)")
    parser.add_argument("--v-drop-ratio", type=float, default=V_DROP_RATIO)
    parser.add_argument("--max-guard", type=int, default=MAX_GUARD)
    parser.add_argument("--no-plot", action="store_true")
    add_series_argument(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    validate_series(args.r_series)
    for voltage in (args.v_read, args.v_set, args.v_reset):
        validate_safety_limits(voltage, args.icc)
    if args.v_set <= 0:
        raise ValueError("--v-set must be > 0 (positive SET)")
    if args.v_reset >= 0:
        raise ValueError("--v-reset must be < 0 (negative RESET)")
    if args.t_pulse <= 0:
        raise ValueError("--t-pulse must be > 0")
    if args.pulse_mode == "software" and args.t_pulse < 20e-3:
        raise ValueError("software pulse width must be >= 0.02 s")
    if args.t_relax < 0:
        raise ValueError("--t-relax must be >= 0")
    if min(args.n_set, args.n_reset, args.n_hold, args.cycles, args.n_read_avg, args.max_guard) <= 0:
        raise ValueError("counts must be > 0")
    if not (0 < args.guard_ratio <= 1):
        raise ValueError("--guard-ratio must be in (0, 1]")
    if args.r_margin < 1:
        raise ValueError("--r-margin must be >= 1")

    v_write = max(abs(args.v_set), abs(args.v_reset))
    auto_min = r_floor_ohm(v_write, args.icc, args.r_margin)
    # The automatic limit is for the whole loop; subtract Rs (not below 0) for the device.
    auto_min = max(auto_min - args.r_series, 0.0)
    r_min = auto_min if args.r_min is None else args.r_min
    r_max = args.r_max
    if r_max <= r_min:
        raise ValueError("--r-max must be > R_min")

    run_dir, ts = make_run_dir(args)
    rs_sfx = series_suffix(args.r_series)
    csv_path = os.path.join(run_dir, f"pulse_set_reset_{rs_sfx}{ts}.csv")
    img_path = os.path.join(run_dir, f"pulse_set_reset_{rs_sfx}{ts}.png")
    meta_path = os.path.join(run_dir, f"pulse_set_reset_{rs_sfx}{ts}.json")

    print("=" * 68)
    print("  Closed-loop Pulse SET / RESET")
    print("=" * 68)
    print(f"  Device       : {args.dev}")
    print(f"  Mode         : {args.mode}")
    print(f"  SET          : {args.v_set:+.4g} V x {args.n_set}")
    print(f"  RESET        : {args.v_reset:+.4g} V x {args.n_reset}")
    print(f"  Pulse width  : {args.t_pulse:.4g} s")
    print(f"  Read         : {args.v_read:+.4g} V, avg={args.n_read_avg}")
    print(f"  Compliance   : {args.icc:.3e} A")
    if args.r_series > 0:
        print(f"  Series R     : {fmt_resistance(args.r_series)} (R = device only)")
    print(f"  R window     : {fmt_resistance(r_min)} .. {fmt_resistance(r_max)}")
    print(f"  On Icc limit : {args.on_limit}")
    print(f"  Pulse mode   : {args.pulse_mode}")
    print(f"  Output       : {run_dir}")
    print("=" * 68)
    print("  SET approaching Icc -> RESET or stop. RESET voltage droop -> stop.")
    print("=" * 68)
    ans = input("\nConfirm start? (Enter / q to quit): ").strip().lower()
    if ans == "q":
        return

    records = []
    inst = connect(args.address)
    t_zero = time.time()
    stop_reason = ""
    try:
        init_smu(inst, args.icc, args.nplc)
        stop_reason, _ = run_closed_loop(inst, args, records, t_zero, r_min, r_max)
    except KeyboardInterrupt:
        print("\n[Stop] Ctrl+C received, saving collected data.")
        stop_reason = stop_reason or "keyboard"
    finally:
        safe_shutdown(inst)
        print("[Safe] Output OFF, voltage returned to 0 V.")
        inst.close()

    save_csv(records, csv_path)
    save_metadata(args, meta_path, {
        "r_min_Ohm": r_min, "stop_reason": stop_reason, "r_series_Ohm": args.r_series,
    })
    finite_r = np.array(
        [r["resistance_Ohm"] for r in records if np.isfinite(r["resistance_Ohm"])],
        dtype=float,
    )
    if len(finite_r):
        print(f"[Summary] R range: {fmt_resistance(float(np.min(finite_r)))} "
              f"to {fmt_resistance(float(np.max(finite_r)))}")
        if np.min(finite_r) > 0:
            print(f"[Summary] Rmax/Rmin: {float(np.max(finite_r) / np.min(finite_r)):.3f}x")
    if stop_reason:
        print(f"[Summary] Stop reason: {stop_reason}")

    if not args.no_plot:
        title = f"{args.dev} SET {args.v_set:+.2f} V / RESET {args.v_reset:+.2f} V"
        if args.r_series > 0:
            title += f"  Rs={args.r_series / 1e3:g} kOhm"
        plot_records(records, img_path, title)


if __name__ == "__main__":
    main()
