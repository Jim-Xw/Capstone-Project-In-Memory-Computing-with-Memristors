"""One-direction, write-verify resistance control at a fixed pulse amplitude.

Default operation only reads at +0.1 V. Explicit --direction down/up enables
positive SET/negative RESET. The controller stops on drift, an opposite trend,
a large jump, compliance, invalid readings, or a conservative resistance floor.
It cannot guarantee monotonic device physics or prevent a single-pulse overshoot.
See pulse_trend.md for commands and the measured limitations of this device.
--r-series <Ohm> subtracts a series HI-line resistor so resistances describe the device.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import time

try:
    from . import keithley_smu as smu
except ImportError:
    import keithley_smu as smu


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=("read", "diagnose", "down", "up"), default="read",
                        help="diagnose: signed current at 0, +0.1, 0 V; never issues write pulses")
    parser.add_argument("--address", default="auto")
    parser.add_argument("--dev", default="DEV")
    parser.add_argument("--v-set", type=float, default=1.0)
    parser.add_argument("--v-reset", type=float, default=-1.0)
    parser.add_argument("--icc", type=float, default=50e-6)
    parser.add_argument("--pulse-mode", choices=("software", "hardware"), default="software")
    parser.add_argument("--t-pulse", type=float, default=0.02)
    parser.add_argument("--max-pulses", type=int, default=3)
    parser.add_argument("--target", type=float, help="Stop after crossing this resistance (Ohm)")
    parser.add_argument("--r-min", type=float, default=24e3,
                        help="Lower bound in Ohm; effective floor also includes 1.2*max(|V|)/Icc")
    parser.add_argument("--r-max", type=float, default=1e6, help="Stop RESET at this upper bound")
    parser.add_argument("--verify-delay", type=float, default=1.0,
                        help="Output-off wait before and between verification bursts (s)")
    parser.add_argument("--verify-timeout", type=float, default=15.0,
                        help="Maximum settling observation time per state (s)")
    parser.add_argument("--stable-reads", type=int, default=3)
    parser.add_argument("--stable-tolerance", type=float, default=0.02,
                        help="Maximum relative resistance spread for settling")
    parser.add_argument("--min-change", type=float, default=0.05,
                        help="Minimum fractional change counted as directional progress")
    parser.add_argument("--max-step", type=float, default=0.20,
                        help="Stop after a pulse changes resistance by more than this fraction")
    parser.add_argument("--max-no-change", type=int, default=2)
    parser.add_argument("--n-read-avg", type=int, default=10)
    parser.add_argument("--nplc", type=float, default=0.1)
    parser.add_argument("--read-settle", type=float, default=0.02)
    smu.add_series_argument(parser)
    return parser.parse_args(argv)


def validate_args(args):
    smu.validate_series(args.r_series)
    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    for voltage in (0.1, args.v_set, args.v_reset):
        smu.validate_safety_limits(voltage, args.icc)
    if args.v_set <= 0 or args.v_reset >= 0:
        raise ValueError("SET must be positive and RESET negative")
    minimum_width = (smu.SOFTWARE_MIN_PULSE_S if args.pulse_mode == "software"
                     else smu.HARDWARE_MIN_PULSE_S)
    if args.t_pulse < minimum_width:
        raise ValueError(f"--t-pulse must be >= {minimum_width} s in this mode")
    if min(args.max_pulses, args.max_no_change, args.n_read_avg) < 1 or args.stable_reads < 2:
        raise ValueError("Counts must be positive; --stable-reads must be >= 2")
    if args.verify_delay <= 0 or args.verify_timeout <= args.verify_delay * args.stable_reads:
        raise ValueError("Verification timeout must exceed stable-reads * verify-delay > 0")
    if args.nplc <= 0 or args.read_settle < 0:
        raise ValueError("NPLC must be positive and read-settle nonnegative")
    if not 0 < args.stable_tolerance < args.min_change < args.max_step < 1:
        raise ValueError("Require 0 < stable-tolerance < min-change < max-step < 1")
    # The Icc term bounds the whole path (device + series resistor); the series
    # resistor also limits write current, so the device-only floor is reduced by it.
    floor = max(args.r_min,
                1.2 * max(abs(args.v_set), abs(args.v_reset)) / args.icc - args.r_series)
    if args.r_min <= 0 or args.r_max <= floor:
        raise ValueError("Require 0 < R_min and R_max above the effective resistance floor")
    if args.target is not None and not floor < args.target <= args.r_max:
        raise ValueError("Target must be above the effective floor and <= R_max")
    return floor


def relative_spread(values):
    return (max(values) - min(values)) / statistics.median(values)


def set_start_floor(args, floor):
    """Reserve for an assumed next SET drop and the configured read spread.

    max_step is an experimental allowance, not a physical bound on the device.
    This screening rule cannot guarantee the post-pulse resistance.
    """
    return floor / ((1 - args.max_step) * (1 - args.stable_tolerance))


def device_sample_resistance(sample, r_series):
    """Resistance of the device alone for one read sample (nan if invalid)."""
    measured = sample.resistance_ohm
    return measured - r_series if math.isfinite(measured) else measured


def stable_resistance(inst, args, emit, pulse_index):
    """Require consecutive quiet bursts, with zero bias between bursts."""
    deadline = time.monotonic() + args.verify_timeout
    window = []
    while time.monotonic() < deadline:
        time.sleep(args.verify_delay)
        if time.monotonic() >= deadline:
            break
        try:
            readings = smu.read_resistance_samples(
                inst, current_limit_a=args.icc, count=args.n_read_avg,
                settle_s=args.read_settle,
            )
        except smu.ResistanceReadError as exc:
            for sample_index, sample in enumerate(exc.readings, 1):
                rejected = sample_index == exc.sample_index
                emit({"event": "read_sample", "pulse_index": pulse_index,
                      "sample_index": sample_index, **asdict(sample),
                      "sample_valid": not rejected, "burst_valid": False,
                      "read_issues": ";".join(exc.issues) if rejected else "",
                      "resistance_ohm": (None if rejected else
                                         device_sample_resistance(sample, args.r_series)),
                      "device_voltage_V": smu.device_voltage(
                          sample.voltage_actual_v, sample.current_a, args.r_series)})
            emit({"event": "read_error", "pulse_index": pulse_index,
                  "sample_index": exc.sample_index, "read_issues": ";".join(exc.issues),
                  "reason": str(exc)})
            raise
        for sample_index, sample in enumerate(readings, 1):
            emit({"event": "read_sample", "pulse_index": pulse_index,
                  "sample_index": sample_index, **asdict(sample),
                  "sample_valid": True, "burst_valid": True,
                  "resistance_ohm": device_sample_resistance(sample, args.r_series),
                  "device_voltage_V": smu.device_voltage(
                      sample.voltage_actual_v, sample.current_a, args.r_series)})
        voltage = statistics.mean(s.voltage_actual_v for s in readings)
        current = statistics.mean(s.current_a for s in readings)
        resistance = voltage / current - args.r_series
        noise = relative_spread([device_sample_resistance(s, args.r_series) for s in readings])
        if noise <= args.stable_tolerance:
            window.append(resistance)
        else:
            window.clear()
        window = window[-args.stable_reads:]
        emit({"event": "read_burst", "pulse_index": pulse_index,
              "resistance_ohm": resistance, "voltage_actual_v": voltage,
              "current_a": current, "relative_spread": noise,
              "device_voltage_V": smu.device_voltage(voltage, current, args.r_series)})
        # Never accept a result that arrived after the observation deadline.
        if (time.monotonic() <= deadline and len(window) == args.stable_reads
                and relative_spread(window) <= args.stable_tolerance):
            value = statistics.median(window)
            emit({"event": "stable_read", "pulse_index": pulse_index,
                  "resistance_ohm": value, "relative_spread": relative_spread(window)})
            return value
    raise RuntimeError("Resistance did not settle within the verification timeout")


def boundary_reason(resistance, args, floor):
    if not math.isfinite(resistance) or resistance <= 0:
        return "invalid_resistance"
    if resistance <= floor:
        return "resistance_floor"
    if args.direction == "up" and resistance >= args.r_max:
        return "resistance_ceiling"
    if args.target is not None:
        reached = resistance <= args.target if args.direction == "down" else resistance >= args.target
        if reached:
            return "target_crossed"
    return ""


def change_reason(before, after, args):
    change = (after - before) / before
    if abs(change) > args.max_step:
        return "step_too_large", change
    progress = -change if args.direction == "down" else change
    if progress < -args.min_change:
        return "opposite_trend", change
    return ("no_clear_change" if progress < args.min_change else "progress"), change


def pulse_failure(pulse, commanded, icc):
    if not math.isfinite(pulse.current_a) or not math.isfinite(pulse.voltage_actual_v):
        return "invalid_pulse_read"
    if abs(pulse.current_a) > icc:
        return "overcurrent"
    if abs(pulse.current_a) == icc:
        return "current_limit_reached"
    if pulse.compliance:
        return "pulse_compliance"
    if not pulse.completed:
        return pulse.stopped_reason or "pulse_incomplete"
    if abs(pulse.voltage_actual_v - commanded) > abs(commanded) * 0.1:
        return "pulse_voltage_mismatch"
    return ""


def diagnose_current(inst, args, emit):
    """Observe offsets and drift without deriving a programming resistance."""
    means = []
    try:
        for phase, voltage in (("zero_before", 0.0), ("read_bias", 0.1), ("zero_after", 0.0)):
            time.sleep(args.verify_delay)
            try:
                samples = smu.read_diagnostic_current_samples(
                    inst, current_limit_a=args.icc, count=args.n_read_avg,
                    settle_s=args.read_settle, voltage_v=voltage,
                )
            except smu.ResistanceReadError as exc:
                for index, sample in enumerate(exc.readings, 1):
                    emit({"event": "diagnostic_sample", "phase": phase,
                          "sample_index": index, **asdict(sample), "burst_valid": False,
                          "sample_valid": index != exc.sample_index,
                          "device_voltage_V": smu.device_voltage(
                              sample.voltage_actual_v, sample.current_a, args.r_series)})
                emit({"event": "read_error", "phase": phase,
                      "read_issues": ";".join(exc.issues), "reason": str(exc)})
                raise
            currents = [sample.current_a for sample in samples]
            for index, sample in enumerate(samples, 1):
                emit({"event": "diagnostic_sample", "phase": phase,
                      "sample_index": index, **asdict(sample), "burst_valid": True,
                      "sample_valid": True,
                      "device_voltage_V": smu.device_voltage(
                          sample.voltage_actual_v, sample.current_a, args.r_series)})
            mean = statistics.mean(currents)
            std = statistics.stdev(currents) if len(currents) > 1 else None
            means.append(mean)
            emit({"event": "diagnostic_summary", "phase": phase,
                  "voltage_set_v": voltage, "current_mean_a": mean,
                  "current_std_a": std, "current_min_a": min(currents),
                  "current_max_a": max(currents), "sample_count": len(currents)})
            std_text = f"{std*1e12:.3f}" if std is not None else "unavailable"
            print(f"[Diagnostic] {phase}: mean={mean*1e12:+.3f} pA, "
                  f"std={std_text} pA, range={min(currents)*1e12:+.3f}..{max(currents)*1e12:+.3f} pA")
        offset = (means[0] + means[2]) / 2
        delta = means[1] - offset
        drift = means[2] - means[0]
        emit({"event": "diagnostic_comparison", "mean_zero_current_a": offset,
              "bias_minus_zero_current_a": delta, "zero_drift_a": drift,
              "write_authorized": False})
        print(f"[Diagnostic] Bias minus mean zero current={delta*1e12:+.3f} pA; "
              f"zero-current drift={drift*1e12:+.3f} pA")
        print("[Diagnostic] No resistance state accepted. Zero-bias current can include relaxation as well as offset.")
        return "diagnostic_only"
    finally:
        smu.safe_shutdown(inst)


def run_trend(inst, args, emit, floor):
    try:
        if args.direction == "diagnose":
            return diagnose_current(inst, args, emit)
        resistance = stable_resistance(inst, args, emit, 0)
        print(f"[Baseline] R={resistance:.6g} Ohm")
        if args.direction == "read":
            return "read_only"
        no_change = 0
        for index in range(1, args.max_pulses + 1):
            reason = boundary_reason(resistance, args, floor)
            if reason:
                return reason
            if args.direction == "down" and resistance <= set_start_floor(args, floor):
                emit({"event": "set_guard", "resistance_ohm": resistance,
                      "pulse_index": index - 1, "r_floor_ohm": floor,
                      "set_start_floor_ohm": set_start_floor(args, floor),
                      "reason": "insufficient_set_reserve"})
                return "insufficient_set_reserve"
            voltage = args.v_set if args.direction == "down" else args.v_reset
            started = time.monotonic()
            try:
                pulse = smu.apply_voltage_pulse(
                    inst, mode=args.pulse_mode, voltage_v=voltage,
                    width_s=args.t_pulse, relax_s=0, current_limit_a=args.icc,
                )
            finally:
                smu.safe_shutdown(inst)
            v_dev = (pulse.voltage_actual_v if math.isfinite(pulse.voltage_actual_v)
                     else pulse.voltage_set_v)
            emit({"event": "pulse", "pulse_index": index, **asdict(pulse),
                  "requested_width_s": args.t_pulse,
                  "host_call_elapsed_s": time.monotonic() - started,
                  "pulse_device_voltage_V": smu.device_voltage(
                      v_dev, pulse.current_a, args.r_series)})
            reason = pulse_failure(pulse, voltage, args.icc)
            if reason:
                return reason
            updated = stable_resistance(inst, args, emit, index)
            reason, delta = change_reason(resistance, updated, args)
            emit({"event": "change", "pulse_index": index,
                  "resistance_ohm": updated, "previous_resistance_ohm": resistance,
                  "fractional_change": delta, "reason": reason})
            print(f"[Pulse {index}] R={updated:.6g} Ohm, change={delta:+.1%}, {reason}")
            if reason in ("step_too_large", "opposite_trend"):
                return reason
            boundary = boundary_reason(updated, args, floor)
            if boundary:
                return boundary
            no_change = no_change + 1 if reason == "no_clear_change" else 0
            if no_change >= args.max_no_change:
                return "no_clear_change"
            resistance = updated
        return "pulse_budget"
    finally:
        smu.safe_shutdown(inst)


def main(argv=None):
    args = parse_args(argv)
    floor = validate_args(args)
    print(f"Direction={args.direction}, Icc={args.icc*1e6:g} uA, floor={floor:g} Ohm")
    print(f"SET requires R > {set_start_floor(args, floor):.6g} Ohm before each pulse")
    if args.r_series > 0:
        print(f"Series resistor={args.r_series:g} Ohm; resistances are device-only")
    print("Connect FRONT FORCE HI/LO. Verification reads use +0.1 V.")
    if args.direction == "diagnose":
        print("Diagnostic only: signed current at 0 V, +0.1 V, 0 V; no SET/RESET pulses.")
    if args.direction in ("down", "up"):
        voltage = args.v_set if args.direction == "down" else args.v_reset
        print(f"Up to {args.max_pulses} pulses at {voltage:+g} V, requested width={args.t_pulse:g} s")
        print("Stops after unexpected changes; a single pulse may still overshoot.")
        if args.pulse_mode == "software":
            print("Software width includes blocking read/communication overhead; it is not a precise pulse width.")
    if input("Confirm start? (Enter / q to quit): ").strip().lower() == "q":
        return 0
    label = "".join(c if c.isalnum() or c in "_-" else "_" for c in args.dev) or "DEV"
    tag = smu.series_tag(args.r_series)
    tag = f"{tag}_" if tag else ""
    folder = Path(__file__).resolve().parent / "Pulse_Trend" / (
        f"{label}_{args.direction}_{tag}{datetime.now():%Y%m%d_%H%M%S_%f}")
    folder.mkdir(parents=True)
    metadata = {**vars(args), "effective_r_min_ohm": floor,
                "set_start_floor_ohm": set_start_floor(args, floor),
                "read_voltage_v": 0.1, "created_at": datetime.now().isoformat(),
                "r_series_Ohm": args.r_series,
                "timing_note": "host_call_elapsed_s is not measured physical pulse width"}
    records = []
    inst = None
    reason = "not_started"
    exit_code = 0
    started = time.monotonic()
    with (folder / "events.jsonl").open("w", encoding="utf-8") as stream:
        def emit(row):
            row = {"elapsed_s": time.monotonic() - started, **row}
            # Keep JSON valid even for an invalid instrument result.
            row = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                   for k, v in row.items()}
            records.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
        try:
            inst = smu.connect_smu(args.address)
            metadata["instrument_idn"] = getattr(inst, "_memristor_idn", "")
            if args.direction in ("down", "up") and args.pulse_mode == "hardware" and smu.instrument_model(inst) != "2461":
                raise ValueError("Hardware pulses require Keithley 2461")
            actual_icc = smu.configure_voltage_source(inst, args.icc, args.nplc)
            metadata["actual_icc_a"] = actual_icc
            if not math.isfinite(actual_icc) or not math.isclose(actual_icc, args.icc, rel_tol=0.01):
                raise RuntimeError("Instrument current limit differs from requested Icc")
            reason = run_trend(inst, args, emit, floor)
        except KeyboardInterrupt:
            reason, exit_code = "keyboard_interrupt", 130
        except Exception as exc:
            reason, exit_code = "error", 1
            metadata["error"] = str(exc)
            print(f"[Error] {exc}")
        finally:
            if inst is not None:
                smu.safe_shutdown(inst)
                try:
                    inst.close()
                except Exception as exc:
                    metadata["close_error"] = str(exc)
            emit({"event": "stop", "reason": reason})
            metadata["stop_reason"] = reason
            (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            with (folder / "events.csv").open("w", newline="", encoding="utf-8") as output:
                fields = list(dict.fromkeys(key for row in records for key in row))
                # Series-resistor columns go after all pre-existing columns.
                added = [k for k in ("device_voltage_V", "pulse_device_voltage_V") if k in fields]
                fields = [k for k in fields if k not in added] + added
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)
    print(f"[Stop] {reason}\n[Saved] {folder}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
