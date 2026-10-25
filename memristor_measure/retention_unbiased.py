"""Unbiased (floating) retention: program a state, then read it sparsely.

Between reads the SMU output is switched off in high-impedance mode, so the
device terminals float as if disconnected. Reads follow a logarithmic
schedule (dense at first, then up to --max-interval apart), so the device is
under the +0.1 V read bias for well under 1% of the time. Each read is
appended to the CSV immediately, so a crash or power cut keeps the data.

With --target the state is first programmed by resistance_tune.Tuner (all
tuner options, including --recipe, are passed through; the recipe also sets
the read settings and band); without it the present state is measured. Retention is judged against the first read after
programming (R0): the run reports when R first leaves R0*(1 +/- tol).

Wiring: FRONT FORCE HI/LO, probes left landed for the whole run. Disable PC
sleep for long runs. Example:
``python retention_unbiased.py --recipe Device6-1 --target 25e3 --duration 86400``
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
    from . import resistance_tune as rt
except ImportError:
    import keithley_smu as smu
    import resistance_tune as rt

import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Retention_Unbiased")
OFF_MODES = {"himp": "HIMP", "normal": "NORM"}
COLUMNS = ["index", "t_since_program_s", "wall_time", "resistance_Ohm", "drift_pct",
           "in_band", "bias_time_total_s", "note"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--address", default=smu.AUTO_ADDRESS)
    # Shared with the tuner: None means "use the recipe or the tuner default".
    p.add_argument("--dev", default=None)
    p.add_argument("--target", type=float, default=None, help="Program this resistance first (Ohm); omit to measure the present state")
    p.add_argument("--tol", type=float, default=None, help="Retention band around R0 (and tuning band); default from recipe or 0.05")
    p.add_argument("--duration", type=float, default=3600.0, help="Total retention time, s")
    p.add_argument("--first", type=float, default=1.0, help="First read after programming, s")
    p.add_argument("--ppd", type=int, default=10, help="Reads per decade of time")
    p.add_argument("--max-interval", type=float, default=300.0, help="Longest gap between reads, s")
    p.add_argument("--off-mode", choices=tuple(OFF_MODES), default="himp",
                   help="Output-off state between reads: himp floats the terminals, normal holds 0 V")
    p.add_argument("--v-read", type=float, default=None)
    p.add_argument("--icc", type=float, default=None)
    p.add_argument("--n-read-avg", type=int, default=None)
    p.add_argument("--read-settle", type=float, default=None)
    p.add_argument("--nplc", type=float, default=None)
    p.add_argument("--r-series", type=float, default=None,
                   help="Series resistor in the HI line, Ohm (default from recipe or 0); "
                        "reads report the device alone and output names get an RS tag")
    p.add_argument("--no-plot", action="store_true")
    args, tuner_extra = p.parse_known_args(argv)
    if args.duration <= args.first or args.first <= 0:
        raise ValueError("Need 0 < --first < --duration")
    if args.ppd <= 0 or args.max_interval <= 0:
        raise ValueError("--ppd and --max-interval must be > 0")
    return args, tuner_extra


def read_schedule(first: float, duration: float, ppd: int, max_interval: float) -> list[float]:
    """Times after programming: 0, then log-spaced, with gaps capped at max_interval."""
    factor = 10 ** (1 / ppd)
    times, t = [0.0], first
    while t <= duration:
        times.append(t)
        t = min(t * factor, t + max_interval)
    if times[-1] < duration:
        times.append(duration)
    return times


def set_output_off_mode(inst, mode: str) -> str:
    """Select the output-off state; returns the mode actually in effect."""
    for cmd in (f":OUTP:VOLT:SMOD {mode}", f":OUTP:SMOD {mode}"):
        inst.write(cmd)
        if smu.error_is_clear(smu.query_error(inst)):
            return mode
    print(f"[Warning] Instrument rejected output-off mode {mode}; using its default (NORMAL, 0 V).")
    return "NORM"


SHARED = ("dev", "tol", "v_read", "icc", "n_read_avg", "read_settle", "nplc", "r_series")


def tuner_args(args, extra):
    """Tuner settings (recipe + overrides); also used for reads. Without --target
    a dummy target is passed and never tuned."""
    argv = ["--target", str(args.target if args.target is not None else 1e4), "--no-plot"]
    for key in SHARED:
        value = getattr(args, key)
        if value is not None:
            argv += ["--" + key.replace("_", "-"), str(value)]
    return rt.parse_args(argv + extra)


def plot(rows, args, r0, img_path):
    t = np.array([r["t_since_program_s"] for r in rows], dtype=float)
    res = np.array([r["resistance_Ohm"] for r in rows], dtype=float) / 1e3
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    ax.tick_params(colors="#FFFFFF", which="both")
    for spine in ax.spines.values():
        spine.set_color("#FFFFFF")
    ax.grid(True, color="#3A3A3A", lw=0.5, which="both")
    t_plot = np.where(t > 0, t, args.first / 2)
    ax.axhspan(r0 * (1 - args.tol) / 1e3, r0 * (1 + args.tol) / 1e3, color="#4DB6AC", alpha=0.25, lw=0,
               label=f"R0 +/-{args.tol*100:g}%")
    ax.plot(t_plot, res, "o-", ms=4, lw=1, color="#90CAF9", label="R at +0.1 V")
    ax.set_xscale("log")
    ax.set_xlabel("Time since programming (s)", color="#FFFFFF")
    ax.set_ylabel("Resistance (kOhm)", color="#FFFFFF")
    ax.set_title(f"{args.dev} unbiased retention, R0={r0/1e3:.2f} kOhm, output-off={args.off_mode}",
                 color="#FFFFFF")
    ax.legend(facecolor="#000000", edgecolor="#FFFFFF", labelcolor="#FFFFFF", fontsize=8)
    plt.tight_layout()
    plt.savefig(img_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"[PNG] Saved -> {img_path}")
    plt.show()


def main():
    args, extra = parse_args()
    t_args = tuner_args(args, extra)
    args.r_series = t_args.r_series
    for key in SHARED:  # effective values: command line > recipe > tuner default
        setattr(args, key, getattr(t_args, key))
    schedule = read_schedule(args.first, args.duration, args.ppd, args.max_interval)
    read_time = args.read_settle + args.n_read_avg * max(args.nplc / 50, 0.005) + 0.05
    duty = len(schedule) * read_time / args.duration * 100

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = smu.series_tag(args.r_series)
    ts = f"{tag}_{ts}" if tag else ts  # e.g. RS10k_20261009_170000
    state = f"T{args.target/1e3:g}k" if args.target else "present"
    run_dir = os.path.join(OUTPUT_DIR, f"{args.dev}_{state}_dur{args.duration:g}s_{args.off_mode}_{ts}")
    print("=" * 68)
    print("  Unbiased Retention")
    print("=" * 68)
    if t_args.recipe_data is not None:
        print(f"  Recipe       : {t_args.recipe_data['path']}  (device {args.dev})")
    print(f"  State        : {'program ' + format(args.target / 1e3, 'g') + ' kOhm' if args.target else 'measure present state'}")
    print(f"  Duration     : {args.duration:g} s ({args.duration/3600:.2f} h), {len(schedule)} reads, "
          f"gaps up to {args.max_interval:g} s")
    print(f"  Between reads: output OFF, mode {args.off_mode} "
          f"({'terminals floating' if args.off_mode == 'himp' else 'terminals held at 0 V'})")
    print(f"  Read         : {args.v_read:+.2f} V, avg {args.n_read_avg}, est. bias duty {duty:.3f}%")
    print(f"  Band         : R0 +/-{args.tol*100:g}%")
    if args.r_series:
        print(f"  Series R     : {args.r_series/1e3:g} kOhm subtracted from every read")
    print(f"  Output       : {run_dir}")
    print("=" * 68)
    print("  Keep probes landed, avoid touching the setup, and disable PC sleep.")
    if input("\nConfirm start? (Enter / q to quit): ").strip().lower() == "q":
        return

    try:
        inst = smu.connect_smu(args.address)
    except smu.InstrumentConnectionError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, f"retention_unbiased_{ts}.csv")
    meta = {k: v for k, v in vars(args).items()}
    meta.update({"created_at": datetime.now().isoformat(timespec="seconds"),
                 "schedule_s": schedule, "tuner_extra_args": extra,
                 "tuner_settings": {k: getattr(t_args, k) for k in rt.RECIPE_KEYS if k != "set_map"}
                 | {"set_map": rt.format_set_map(t_args.set_map)},
                 "recipe_used": None if t_args.recipe_data is None else t_args.recipe_data["path"]})
    rows, r0, t_exit, tune_result, tuner = [], None, None, None, None
    bias_total = 0.0

    try:
        smu.configure_voltage_source(inst, args.icc, args.nplc)  # *RST resets the off mode
        meta["off_mode_in_effect"] = set_output_off_mode(inst, OFF_MODES[args.off_mode])
        if args.target is not None:
            tuner = rt.Tuner(inst, t_args)
            tune_result = tuner.tune(args.target)
            meta["tuning"] = tune_result
            if not tune_result["success"]:
                raise rt.StopRun("target not reached; retention not started")
        reader = rt.Tuner(inst, t_args)
        t_prog = time.time()
        meta["programmed_at"] = datetime.now().isoformat(timespec="seconds")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            for k, t_sched in enumerate(schedule):
                delay = t_prog + t_sched - time.time()
                if delay > 0:
                    time.sleep(delay)
                t_read = time.time()
                note = ""
                try:
                    r = reader.read()
                except rt.StopRun as exc:
                    r, note = float("nan"), str(exc)[:200]
                    smu.safe_shutdown(inst)
                bias_total += time.time() - t_read
                t_rel = t_read - t_prog
                if r0 is None and np.isfinite(r):
                    r0 = r
                drift = (r - r0) / r0 * 100 if r0 and np.isfinite(r) else float("nan")
                in_band = bool(np.isfinite(drift) and abs(drift) <= args.tol * 100)
                if t_exit is None and np.isfinite(drift) and not in_band:
                    t_exit = t_rel
                row = {"index": k, "t_since_program_s": round(t_rel, 3),
                       "wall_time": datetime.now().isoformat(timespec="seconds"),
                       "resistance_Ohm": r, "drift_pct": round(drift, 3) if np.isfinite(drift) else "nan",
                       "in_band": in_band, "bias_time_total_s": round(bias_total, 3), "note": note}
                writer.writerow(row)
                f.flush()
                rows.append(row)
                print(f"  {k:4d}  t={t_rel:10.1f} s  R={r/1e3:9.3f} kOhm  drift={drift:+6.2f}%"
                      f"{'' if in_band else '  OUT OF BAND'}{'  ' + note if note else ''}")
    except rt.StopRun as exc:
        print(f"[Safety stop] {exc}")
    except KeyboardInterrupt:
        print("\n[Stop] Ctrl+C received; data so far is already saved.")
    finally:
        smu.safe_shutdown(inst)
        print("[Safe] Output OFF, voltage returned to 0 V.")
        inst.close()

    if tuner is not None:
        meta["tuning_steps"] = tuner.records
    valid = [r for r in rows if np.isfinite(r["resistance_Ohm"])]
    meta["R0_Ohm"] = r0
    meta["first_exit_s"] = t_exit
    meta["bias_time_total_s"] = bias_total
    if len(valid) >= 4 and r0:
        t = np.array([r["t_since_program_s"] for r in valid[1:]], dtype=float)
        res = np.array([r["resistance_Ohm"] for r in valid[1:]], dtype=float)
        ok = (t > 0) & (res > 0)
        if ok.sum() >= 3 and np.ptp(t[ok]) > 0:
            meta["loglog_slope"] = float(np.polyfit(np.log10(t[ok]), np.log10(res[ok]), 1)[0])
    with open(os.path.join(run_dir, f"retention_unbiased_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    if not valid:
        return
    last = valid[-1]
    print(f"\n[Summary] R0={r0/1e3:.3f} kOhm, last R={last['resistance_Ohm']/1e3:.3f} kOhm "
          f"({last['drift_pct']}%) at t={last['t_since_program_s']:.0f} s")
    print(f"[Summary] {'Left the +/-' + format(args.tol*100, 'g') + '% band at t=' + format(t_exit, '.0f') + ' s' if t_exit else 'Stayed inside the band for the whole run'}")
    print(f"[Summary] Total time under read bias: {bias_total:.1f} s "
          f"({bias_total / max(last['t_since_program_s'], 1e-9) * 100:.3f}%)")
    if "loglog_slope" in meta:
        print(f"[Summary] Drift exponent d(log R)/d(log t) = {meta['loglog_slope']:+.4f}")
    print(f"[CSV] Saved -> {csv_path}")
    if not args.no_plot:
        plot(valid, args, r0, os.path.join(run_dir, f"retention_unbiased_{ts}.png"))


if __name__ == "__main__":
    main()
