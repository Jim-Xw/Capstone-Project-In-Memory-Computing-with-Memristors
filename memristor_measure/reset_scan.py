"""Find a switching voltage safely: staircase of single pulses, read after each.

Each step applies one pulse at the next amplitude (v_start, v_start+v_step,
... up to v_stop), waits --t-relax, then reads R at +0.1 V. The scan stops at
the FIRST sign of change, so the device is never driven further than needed:

  switched      R moved the wanted way by --success   (the switching voltage)
  wrong_way     R moved the other way by --wrong      (stop: do not go higher)
  limit_hit     a pulse reached --icc-pulse
  v_stop        no change up to --v-stop

--goal up (default) looks for RESET (R rises); --goal down looks for SET.
With a series resistor give --r-series: reads report the device alone
(R_measured - R_series) and each pulse logs the voltage left across the device
(V - I*R_series). Pulse amplitudes are the applied (instrument) voltages.

R0 is the read before the first pulse. After stopping, R is read again after
--settle seconds, because RESET on these devices keeps rising for seconds.
Nothing is ever applied in the opposite polarity. Example:
``python reset_scan.py --dev Device11-1 --v-start 0.30 --v-stop 0.60 --t-pulse 0.1``
``python reset_scan.py --dev Device11-1 --r-series 10e3 --goal down --polarity negative --v-start 1.0 --v-stop 2.0``
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

try:
    from . import keithley_smu as smu
except ImportError:
    import keithley_smu as smu

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reset_Scan")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stop-on-change RESET voltage scan")
    p.add_argument("--address", default=smu.AUTO_ADDRESS)
    p.add_argument("--dev", default="DEV")
    p.add_argument("--polarity", choices=("positive", "negative"), default="positive",
                   help="Polarity of the pulses (default positive)")
    p.add_argument("--goal", choices=("up", "down"), default="up",
                   help="up = look for RESET (R rises), down = look for SET (R falls)")
    p.add_argument("--r-series", type=float, default=0.0, help="Series resistor in the HI line, Ohm")
    p.add_argument("--v-start", type=float, default=0.30, help="First pulse amplitude, V (magnitude)")
    p.add_argument("--v-stop", type=float, default=0.60, help="Never exceed this amplitude, V (magnitude)")
    p.add_argument("--v-step", type=float, default=0.01)
    p.add_argument("--t-pulse", type=float, default=0.1, help="Pulse width, s")
    p.add_argument("--repeats", type=int, default=1, help="Pulses per amplitude before stepping up")
    p.add_argument("--icc-pulse", type=float, default=2e-4, help="Pulse current limit, A; reaching it stops the scan")
    p.add_argument("--success", "--rise", dest="success", type=float, default=0.30,
                   help="Relative change in the wanted direction that counts as switching")
    p.add_argument("--wrong", "--drop", dest="wrong", type=float, default=0.05,
                   help="Relative change in the other direction that stops the scan")
    p.add_argument("--t-relax", type=float, default=1.0, help="Wait after each pulse before reading, s")
    p.add_argument("--settle", type=float, default=5.0, help="Wait before the final confirming read, s")
    p.add_argument("--v-read", type=float, default=0.1)
    p.add_argument("--icc", type=float, default=1e-4, help="Read current limit, A")
    p.add_argument("--n-read-avg", type=int, default=10)
    p.add_argument("--nplc", type=float, default=1.0)
    p.add_argument("--pulse-mode", choices=("software", "hardware"), default="software")
    args = p.parse_args(argv)
    if not 0 < args.v_start <= args.v_stop:
        raise ValueError("Need 0 < --v-start <= --v-stop")
    if args.v_step <= 0 or args.repeats < 1:
        raise ValueError("--v-step must be > 0 and --repeats >= 1")
    if not (0 < args.wrong < 1 and 0 < args.success < 1):
        raise ValueError("Need 0 < --wrong < 1 and 0 < --success < 1")
    if args.r_series < 0:
        raise ValueError("--r-series must be >= 0")
    smu.validate_safety_limits(args.v_stop, args.icc_pulse)
    smu.validate_safety_limits(args.v_read, args.icc)
    min_width = smu.SOFTWARE_MIN_PULSE_S if args.pulse_mode == "software" else smu.HARDWARE_MIN_PULSE_S
    if args.t_pulse < min_width:
        raise ValueError(f"--t-pulse must be >= {min_width} s in {args.pulse_mode} mode")
    return args


def read_r(inst, args) -> float:
    smu.set_current_limit(inst, args.icc)
    readings = smu.read_resistance_samples(
        inst, current_limit_a=args.icc, count=args.n_read_avg,
        settle_s=0.02, voltage_v=args.v_read,
    )
    v = float(np.mean([r.voltage_actual_v for r in readings]))
    i = float(np.mean([r.current_a for r in readings]))
    return abs(v / i) - args.r_series


def amplitudes(args) -> list[float]:
    n = int(round((args.v_stop - args.v_start) / args.v_step))
    return [round(args.v_start + k * args.v_step, 6) for k in range(n + 1)]


def scan(inst, args, log) -> tuple[str, float, float, float]:
    """Returns (stop reason, voltage at stop, R0, R at stop)."""
    sign = 1.0 if args.polarity == "positive" else -1.0
    r0 = read_r(inst, args)
    log("start", 0.0, None, r0, r0)
    for amp in amplitudes(args):
        for _ in range(args.repeats):
            pulse = smu.apply_voltage_pulse(
                inst, mode=args.pulse_mode, voltage_v=sign * amp, width_s=args.t_pulse,
                relax_s=args.t_relax, current_limit_a=args.icc_pulse,
            )
            hit = pulse.compliance or (np.isfinite(pulse.current_a)
                                       and abs(pulse.current_a) >= args.icc_pulse * 0.985)
            if not pulse.completed and not hit:
                log("abort", amp, pulse, float("nan"), r0)
                return f"pulse_failed:{pulse.stopped_reason}", amp, r0, float("nan")
            r = read_r(inst, args)
            log("pulse", amp, pulse, r, r0)
            if hit:
                return "limit_hit", amp, r0, r
            up_ok, down_ok = r >= r0 * (1 + args.success), r <= r0 * (1 - args.success)
            up_bad, down_bad = r >= r0 * (1 + args.wrong), r <= r0 * (1 - args.wrong)
            if (args.goal == "up" and up_ok) or (args.goal == "down" and down_ok):
                return "switched", amp, r0, r
            if (args.goal == "up" and down_bad) or (args.goal == "down" and up_bad):
                return "wrong_way", amp, r0, r
    return "v_stop", args.v_stop, r0, r


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = smu.series_tag(args.r_series)
    ts = f"{tag}_{ts}" if tag else ts  # e.g. RS10k_20261009_170000
    sgn = "+" if args.polarity == "positive" else "-"
    run_dir = os.path.join(OUTPUT_DIR, f"{args.dev}_{sgn}{args.v_start:.2f}to{sgn}{args.v_stop:.2f}V_"
                                       f"{args.t_pulse*1000:g}ms_{ts}")
    amps = amplitudes(args)
    print("=" * 68)
    print("  Stop-on-Change Switching Scan")
    print("=" * 68)
    print(f"  Device       : {args.dev}  (looking for {'RESET, R up' if args.goal == 'up' else 'SET, R down'}, "
          f"{args.polarity} pulses)")
    if args.r_series:
        print(f"  Series R     : {args.r_series/1e3:g} kOhm (subtracted from reads; device voltage logged)")
    print(f"  Pulses       : {sgn}{args.v_start:.3f} -> {sgn}{args.v_stop:.3f} V in {args.v_step*1000:g} mV steps "
          f"({len(amps)} steps x {args.repeats}), {args.t_pulse:g} s, limit {args.icc_pulse:.1e} A")
    want = 1 + args.success if args.goal == "up" else 1 - args.success
    bad = 1 - args.wrong if args.goal == "up" else 1 + args.wrong
    print(f"  Stops when   : R reaches R0 x {want:.2f} (switched)  |  R reaches R0 x {bad:.2f} (wrong way)"
          f"  |  pulse at limit")
    print(f"  Read         : {args.v_read:+.2f} V, {args.t_relax:g} s after each pulse, limit {args.icc:.1e} A")
    print(f"  Output       : {run_dir}")
    print("=" * 68)
    if input("\nConfirm start? (Enter / q to quit): ").strip().lower() == "q":
        return

    try:
        inst = smu.connect_smu(args.address)
    except smu.InstrumentConnectionError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    rows, t0 = [], time.time()

    def log(kind, amp, pulse, r, r0):
        change = (r - r0) / r0 * 100 if np.isfinite(r) else float("nan")
        rows.append({
            "step": len(rows), "time_s": round(time.time() - t0, 3), "kind": kind,
            "pulse_amplitude_V": amp,
            "pulse_voltage_actual_V": float("nan") if pulse is None else pulse.voltage_actual_v,
            "pulse_current_A": float("nan") if pulse is None else pulse.current_a,
            "device_voltage_V": float("nan") if pulse is None
            else pulse.voltage_actual_v - pulse.current_a * args.r_series,
            "compliance": False if pulse is None else pulse.compliance,
            "resistance_Ohm": r, "change_from_R0_pct": round(change, 2),
        })
        i_txt = "" if pulse is None else f"  I={abs(pulse.current_a)*1e6:7.1f} uA"
        if pulse is not None and args.r_series:
            i_txt += f"  Vdev={pulse.voltage_actual_v - pulse.current_a * args.r_series:+.3f} V"
        print(f"  {kind:>6}  {sgn}{amp:.3f} V{i_txt}  R={r/1e3:9.3f} kOhm  ({change:+6.1f}%)")

    reason, v_at, r0, r_at, r_settled = "error", float("nan"), float("nan"), float("nan"), float("nan")
    try:
        smu.configure_voltage_source(inst, args.icc, args.nplc)
        reason, v_at, r0, r_at = scan(inst, args, log)
        time.sleep(args.settle)
        r_settled = read_r(inst, args)
        log("settled", v_at, None, r_settled, r0)
    except smu.ResistanceReadError as exc:
        reason = f"read_failed: {exc}"
    except KeyboardInterrupt:
        reason = "interrupted"
    finally:
        smu.safe_shutdown(inst)
        inst.close()
        print("[Safe] Output OFF, voltage returned to 0 V.")

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, f"reset_scan_{ts}.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["step"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {"stop_reason": reason, "stop_amplitude_V": v_at, "R0_Ohm": r0,
               "R_at_stop_Ohm": r_at, "R_settled_Ohm": r_settled}
    with open(os.path.join(run_dir, f"reset_scan_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "created_at": datetime.now().isoformat(timespec="seconds"),
                   "result": summary}, f, indent=2)
    meaning = {
        "switched": "switching voltage found (" + ("RESET" if args.goal == "up" else "SET") + ")",
        "wrong_way": "R moved the WRONG way; do not go higher",
        "limit_hit": "pulse reached its current limit; stopped",
        "v_stop": "no change up to --v-stop",
    }.get(reason, reason)
    print(f"\n[Result] {meaning}  at {sgn}{v_at:.3f} V")
    print(f"[Result] R0 {r0/1e3:.2f} kOhm -> {r_at/1e3:.2f} kOhm, after {args.settle:g} s {r_settled/1e3:.2f} kOhm")
    print(f"[Saved] {run_dir}")


if __name__ == "__main__":
    main()
