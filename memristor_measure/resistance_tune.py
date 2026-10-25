"""Write-verify resistance tuning for bipolar memristors (either SET polarity).

Strategy (measured on Ni/AlOx/TiOx/Ti, 2026-09-24): SET is abrupt and its
low-resistance level is set by the SET compliance; RESET is gradual at low
amplitude. So each target is approached from below:

1. If R is above the band, apply one SET whose compliance is chosen from
   --set-map so the expected LRS lands below the band.
2. Apply short RESET pulses with an incrementing amplitude (ISPP): the
   amplitude rises by --v-reset-step only while pulses produce no clear change.
3. Stop when R is inside target*(1 +/- tol) and still inside after
   --verify-delay. An overshoot triggers a new SET and a lower RESET amplitude.

Reads use +0.1 V at --icc. Wiring: FRONT FORCE HI/LO. software mode needs
pulses >= 20 ms; hardware mode needs a 2461. Safety: |V| <= 2 V, RESET amplitude
never exceeds --v-reset-max, and an overcurrent or failed pulse/read stops
the run and returns the output to 0 V. Example:
``python resistance_tune.py --target 10e3 20e3 30e3 --tol 0.05 --hold 60``

Polarity: the sign of --v-set selects it. Negative --v-set means positive
RESET pulses (Devices 3-1/6-1); positive --v-set means negative RESET pulses
(Device 13-1). --v-reset-start/--v-reset-max are magnitudes. Setting them equal
gives a fixed-amplitude RESET that is simply repeated; --max-stalls pulses in a
row without clear progress end that target.

Wear test inside a safe range: tune alternating or shuffled targets for many
--rounds, e.g. ``--target 45e3 75e3 --rounds 200 --r-safe 30e3 100e3
--max-drop 0.3 --stop-after-fails 3``. Every write ends at a chosen value, any
read outside --r-safe or a RESET pulse that drops R by more than --max-drop
stops the run, and N missed targets among the last 2N end it as worn out.

Many targets: ``--target-range 75e3 125e3 60`` tunes 60 log-spaced targets in
ascending order; a summary plot of achieved vs target is saved.

Device recipes: ``--recipe Device6-1`` loads recipes/Device6-1.json (or any
path), whose "params" set the defaults; command-line options still override
them. ``--save-recipe NAME`` writes the effective settings to a recipe file
and exits without connecting. Every run's JSON records the recipe it used.
"""

from __future__ import annotations

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
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

try:
    from . import keithley_smu as smu
except ImportError:
    import keithley_smu as smu

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Resistance_Tune")

# SET compliance (A) -> typical LRS read at +0.1 V (Ohm), from the
# 2026-09-24 17:11 runs: 200 uA gave 15.8-18.9 kOhm, 300 uA gave 6.5-9.4 kOhm.
DEFAULT_SET_MAP = "3e-4:7.9e3,2e-4:17.7e3"

RECIPE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes")
# Settings a recipe may hold (argparse dest names). Targets, hold time and the
# instrument address are per-run choices, so they stay on the command line.
RECIPE_KEYS = (
    "tol", "v_read", "icc", "v_set", "t_set", "set_map", "set_spread",
    "v_reset_start", "v_reset_step", "v_reset_max", "t_reset", "icc_reset",
    "min_change", "max_pulses", "max_sets", "r_floor", "verify_delay",
    "t_relax", "n_read_avg", "read_settle", "nplc", "pulse_mode", "r_series",
    "max_stalls", "max_drop",
)


class StopRun(RuntimeError):
    pass


def parse_set_map(text: str) -> list[tuple[float, float]]:
    pairs = []
    for item in text.split(","):
        icc, r = item.split(":")
        pairs.append((float(icc), float(r)))
    if not pairs:
        raise ValueError("--set-map is empty")
    return sorted(pairs, key=lambda p: p[1])


def recipe_path(name: str) -> str:
    """A bare name maps to recipes/<name>.json; anything else is used as a path."""
    if os.path.dirname(name) or name.lower().endswith(".json"):
        return name
    return os.path.join(RECIPE_DIR, f"{name}.json")


def format_set_map(pairs) -> str:
    return ",".join(f"{icc:g}:{r:g}" for icc, r in pairs)


def load_recipe(name: str) -> dict:
    path = recipe_path(name)
    with open(path, encoding="utf-8") as f:
        recipe = json.load(f)
    params = dict(recipe.get("params", {}))
    unknown = sorted(set(params) - set(RECIPE_KEYS))
    if unknown:
        raise ValueError(f"Recipe {path} has unknown params: {', '.join(unknown)}")
    if "set_map" in params:
        sm = params["set_map"]
        params["set_map"] = (parse_set_map(sm) if isinstance(sm, str)
                             else sorted(((float(e["icc_A"]), float(e["typical_lrs_Ohm"])) for e in sm),
                                         key=lambda p: p[1]))
    recipe["params"] = params
    recipe["path"] = os.path.abspath(path)
    return recipe


def save_recipe(args, name: str) -> str:
    """Write the effective settings; keeps notes/measurements already in the file."""
    path = recipe_path(name)
    recipe = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            recipe = json.load(f)
    params = {k: getattr(args, k) for k in RECIPE_KEYS}
    params["set_map"] = format_set_map(args.set_map)
    recipe.setdefault("device", args.dev)
    recipe.setdefault("notes", "")
    recipe["updated_at"] = datetime.now().isoformat(timespec="seconds")
    recipe["params"] = params
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2)
    return path


def choose_set_icc(args, lo: float) -> tuple[float, float]:
    """Highest-LRS entry whose expected spread still lands below the band."""
    candidates = [p for p in args.set_map if p[1] * (1 + args.set_spread) < lo]
    if candidates:
        return candidates[-1]
    return args.set_map[0]


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--recipe")
    known, _ = pre.parse_known_args(argv)
    recipe = load_recipe(known.recipe) if known.recipe else None

    p = argparse.ArgumentParser(description="Write-verify resistance tuning")
    p.add_argument("--recipe", help="Device recipe: name in recipes/ or path to a JSON file")
    p.add_argument("--save-recipe", metavar="NAME",
                   help="Write the effective settings to this recipe (name or path) and exit")
    p.add_argument("--address", default=smu.AUTO_ADDRESS)
    p.add_argument("--dev", default="DEV")
    p.add_argument("--target", type=float, nargs="+", help="Target resistance(s) in Ohm, tuned in order")
    p.add_argument("--target-range", type=float, nargs=3, metavar=("START", "STOP", "N"),
                   help="Generate N targets from START to STOP (Ohm), tuned in ascending order")
    p.add_argument("--target-spacing", choices=("log", "lin"), default="log",
                   help="Spacing for --target-range (default log)")
    p.add_argument("--shuffle", action="store_true",
                   help="Tune the targets in random order, so each one is written from an unrelated state")
    p.add_argument("--seed", type=int, default=0, help="Random seed for --shuffle (recorded in the JSON)")
    p.add_argument("--rounds", type=int, default=1,
                   help="Tune the whole target list this many times (reshuffled each round with --shuffle)")
    p.add_argument("--tol", type=float, default=0.05, help="Relative tolerance band, e.g. 0.05 = +/-5%%")
    p.add_argument("--v-read", type=float, default=0.1)
    p.add_argument("--icc", type=float, default=3e-4, help="Read compliance in A")
    p.add_argument("--v-set", type=float, default=-1.0,
                   help="SET pulse voltage; its sign selects polarity (RESET uses the opposite sign)")
    p.add_argument("--t-set", type=float, default=0.5)
    p.add_argument("--set-map", type=parse_set_map, default=parse_set_map(DEFAULT_SET_MAP),
                   help="SET compliance:typical LRS pairs, e.g. 3e-4:7.9e3,2e-4:17.7e3")
    p.add_argument("--set-spread", type=float, default=0.15, help="Assumed relative LRS spread when choosing SET compliance")
    p.add_argument("--v-reset-start", type=float, default=0.75)
    p.add_argument("--v-reset-step", type=float, default=0.01)
    p.add_argument("--v-reset-max", type=float, default=0.90)
    p.add_argument("--t-reset", type=float, default=0.02)
    p.add_argument("--icc-reset", type=float, default=8e-4)
    p.add_argument("--min-change", type=float, default=0.02, help="Relative R rise below which RESET amplitude is increased")
    p.add_argument("--max-pulses", type=int, default=60, help="Max write pulses per target")
    p.add_argument("--max-sets", type=int, default=6, help="Max SET pulses per target")
    p.add_argument("--max-drop", type=float, default=0.5,
                   help="Stop if a RESET pulse lowers R by more than this fraction (wrong-way switching)")
    p.add_argument("--r-safe", type=float, nargs=2, metavar=("LO", "HI"),
                   help="Safe device-R window (Ohm): any read outside it stops the run")
    p.add_argument("--stop-after-fails", type=int, default=0,
                   help="End the run when N of the last 2N targets were missed (0 = never); "
                        "catches both N misses in a row and one level failing every round")
    p.add_argument("--max-stalls", type=int, default=3,
                   help="RESET pulses in a row at --v-reset-max without clear progress before giving up")
    p.add_argument("--r-floor", type=float, default=1e3,
                   help="Refuse to pulse if R reads below this (Ohm): likely a short or a probe problem")
    p.add_argument("--verify-delay", type=float, default=2.0, help="Wait before the confirming read, s")
    p.add_argument("--hold", type=float, default=0.0, help="Read-only hold after each target, s")
    p.add_argument("--hold-interval", type=float, default=2.0)
    p.add_argument("--t-relax", type=float, default=0.1)
    p.add_argument("--n-read-avg", type=int, default=10)
    p.add_argument("--read-settle", type=float, default=0.02)
    p.add_argument("--nplc", type=float, default=1.0)
    p.add_argument("--pulse-mode", choices=("software", "hardware"), default="software")
    p.add_argument("--r-series", type=float, default=0.0,
                   help="Series resistor in the HI line, Ohm; subtracted from every read")
    p.add_argument("--no-plot", action="store_true")
    if recipe is not None:
        p.set_defaults(**recipe["params"])
        if recipe.get("device"):
            p.set_defaults(dev=recipe["device"])
    args = p.parse_args(argv)
    args.recipe_data = recipe
    if args.target_range and not args.target:
        start, stop, n = args.target_range
        if not (0 < start < stop and n >= 1 and float(n).is_integer()):
            raise ValueError("--target-range needs 0 < START < STOP and integer N >= 1")
        gen = np.geomspace if args.target_spacing == "log" else np.linspace
        args.target = [float(round(t, 1)) for t in gen(start, stop, int(n))]
    if not args.save_recipe and not args.target:
        p.error("--target or --target-range is required (unless only saving a recipe)")
    if args.target:
        import random
        if args.rounds < 1:
            raise ValueError("--rounds must be >= 1")
        base, rounds = list(args.target), []
        for k in range(args.rounds):
            order = list(base)
            if args.shuffle:
                random.Random(args.seed + k).shuffle(order)
            rounds.extend(order)
        args.target = rounds

    if args.v_set == 0 or args.v_reset_start <= 0:
        raise ValueError("--v-set must be non-zero; --v-reset-start/--v-reset-max are magnitudes > 0")
    if args.max_stalls < 1:
        raise ValueError("--max-stalls must be >= 1")
    args.reset_sign = -1.0 if args.v_set > 0 else 1.0  # RESET uses the opposite polarity to SET
    if not 0 < args.max_drop < 1:
        raise ValueError("--max-drop must be in (0, 1)")
    if args.r_safe:
        lo_s, hi_s = args.r_safe
        if not 0 <= lo_s < hi_s:
            raise ValueError("--r-safe needs 0 <= LO < HI")
        outside = [t for t in args.target or [] if not (lo_s < t * (1 - args.tol) and t * (1 + args.tol) < hi_s)]
        if outside:
            raise ValueError(f"Targets with their band must lie inside --r-safe: {sorted(set(outside))}")
    if not 0 < args.v_reset_start <= args.v_reset_max:
        raise ValueError("Need 0 < --v-reset-start <= --v-reset-max")
    if args.v_reset_step <= 0:
        raise ValueError("--v-reset-step must be > 0")
    if not 0 < args.tol < 0.5:
        raise ValueError("--tol must be in (0, 0.5)")
    if any(t <= 0 for t in args.target or []):
        raise ValueError("--target values must be > 0")
    smu.validate_safety_limits(args.v_read, args.icc)
    smu.validate_safety_limits(args.v_reset_max, args.icc_reset)
    for icc, _ in args.set_map:
        smu.validate_safety_limits(args.v_set, icc)
    min_width = smu.SOFTWARE_MIN_PULSE_S if args.pulse_mode == "software" else smu.HARDWARE_MIN_PULSE_S
    if min(args.t_set, args.t_reset) < min_width:
        raise ValueError(f"Pulse widths must be >= {min_width} s in {args.pulse_mode} mode")
    return args


class Tuner:
    def __init__(self, inst, args):
        self.inst = inst
        self.args = args
        self.records = []
        self.t_zero = time.time()
        self.step = 0
        self.v_effective = None  # last RESET amplitude that moved R

    def read(self) -> float:
        a = self.args
        smu.set_current_limit(self.inst, a.icc)
        try:
            readings = smu.read_resistance_samples(
                self.inst, current_limit_a=a.icc, count=a.n_read_avg,
                settle_s=a.read_settle, voltage_v=a.v_read,
            )
        except smu.ResistanceReadError as exc:
            raise StopRun(f"read failed: {exc}") from exc
        v = float(np.mean([r.voltage_actual_v for r in readings]))
        i = float(np.mean([r.current_a for r in readings]))
        r = abs(v / i) - a.r_series  # device alone
        if a.r_safe and not (a.r_safe[0] <= r <= a.r_safe[1]):
            raise StopRun(f"R={r/1e3:.2f} kOhm left the safe window "
                          f"{a.r_safe[0]/1e3:g}-{a.r_safe[1]/1e3:g} kOhm")
        return r

    def pulse(self, voltage: float, width: float, icc: float):
        result = smu.apply_voltage_pulse(
            self.inst, mode=self.args.pulse_mode, voltage_v=voltage,
            width_s=width, relax_s=self.args.t_relax, current_limit_a=icc,
        )
        overcurrent = np.isfinite(result.current_a) and abs(result.current_a) > icc * 1.02
        if overcurrent or (not result.completed and not result.compliance):
            self.log("abort", voltage, width, icc, result, float("nan"), float("nan"))
            raise StopRun(f"pulse failed: {result.stopped_reason or 'overcurrent'}")
        return result

    def log(self, action, voltage, width, icc, pulse, r, target):
        self.step += 1
        lo, hi = target * (1 - self.args.tol), target * (1 + self.args.tol)
        in_band = bool(np.isfinite(r) and lo <= r <= hi)
        self.records.append({
            "step": self.step,
            "time_s": time.time() - self.t_zero,
            "target_Ohm": target,
            "action": action,
            "pulse_voltage_V": voltage,
            "pulse_width_s": width,
            "pulse_icc_A": icc,
            "pulse_voltage_actual_V": float("nan") if pulse is None else pulse.voltage_actual_v,
            "pulse_current_A": float("nan") if pulse is None else pulse.current_a,
            "compliance": False if pulse is None else pulse.compliance,
            "resistance_Ohm": r,
            "in_band": in_band,
        })
        v_txt = "" if not np.isfinite(voltage) else f"V={voltage:+.3f} "
        tag = "  [compliance]" if pulse is not None and pulse.compliance else ""
        print(f"  {self.step:4d}  {action:>9}  {v_txt:<10} R={r/1e3:8.3f} kOhm"
              f"{'  in band' if in_band else ''}{tag}")
        return in_band

    def tune(self, target: float) -> dict:
        a = self.args
        lo, hi = target * (1 - a.tol), target * (1 + a.tol)
        icc_set, r_expected = choose_set_icc(a, lo)
        print(f"\n[Target] {target/1e3:.3f} kOhm  band {lo/1e3:.3f}-{hi/1e3:.3f} kOhm  "
              f"SET Icc={icc_set:.2e} A (typ. {r_expected/1e3:.1f} kOhm)")
        if r_expected * (1 + a.set_spread) >= lo:
            print("  [Warning] Target is at or below the lowest calibrated LRS; SET may land above the band.")

        r = self.read()
        self.log("read", float("nan"), float("nan"), float("nan"), None, r, target)
        if r < a.r_floor:
            raise StopRun(f"R={r:.0f} Ohm is below --r-floor {a.r_floor:.0f} Ohm before any pulse; "
                          "check probe landing / shorts before writing")
        v_reset = a.v_reset_start
        if self.v_effective is not None:
            v_reset = max(a.v_reset_start, self.v_effective - a.v_reset_step)
        pulses = sets = stalls_at_max = 0
        while pulses < a.max_pulses:
            if lo <= r <= hi:
                time.sleep(a.verify_delay)
                r = self.read()
                if self.log("verify", float("nan"), float("nan"), float("nan"), None, r, target):
                    return {"target_Ohm": target, "success": True, "final_Ohm": r,
                            "pulses": pulses, "sets": sets}
                continue
            if r > hi:
                if sets >= a.max_sets:
                    break
                result = self.pulse(a.v_set, a.t_set, icc_set)
                pulses += 1
                sets += 1
                r = self.read()
                self.log("set", a.v_set, a.t_set, icc_set, result, r, target)
                continue
            # Below the band: gradual RESET (amplitude stepped up only while stalled).
            v_pulse = a.reset_sign * v_reset
            result = self.pulse(v_pulse, a.t_reset, a.icc_reset)
            pulses += 1
            r_new = self.read()
            self.log("reset", v_pulse, a.t_reset, a.icc_reset, result, r_new, target)
            # A RESET pulse should never reach its limit or collapse R; either means
            # it is acting as a SET (or the wiring is reversed).
            if result.compliance:
                raise StopRun(f"RESET pulse at {v_pulse:+.3f} V reached its {a.icc_reset:.1e} A "
                              f"limit (R {r/1e3:.2f} -> {r_new/1e3:.2f} kOhm); check polarity")
            if r_new < (1 - a.max_drop) * r:
                raise StopRun(f"RESET pulse at {v_pulse:+.3f} V lowered R "
                              f"{r/1e3:.2f} -> {r_new/1e3:.2f} kOhm; check polarity")
            if r_new > hi:
                v_reset = max(a.v_reset_start, v_reset - 2 * a.v_reset_step)
                self.v_effective = v_reset
                stalls_at_max = 0
            elif (r_new - r) / r < a.min_change:
                if v_reset >= a.v_reset_max - 1e-9:
                    stalls_at_max += 1
                    if stalls_at_max >= a.max_stalls:
                        print("  [Stop] No progress at --v-reset-max.")
                        r = r_new
                        break
                v_reset = min(v_reset + a.v_reset_step, a.v_reset_max)
            else:
                self.v_effective = v_reset
                stalls_at_max = 0
            r = r_new
        return {"target_Ohm": target, "success": False, "final_Ohm": r,
                "pulses": pulses, "sets": sets}

    def hold(self, target: float):
        a = self.args
        if a.hold <= 0:
            return
        print(f"  [Hold] {a.hold:.0f} s, every {a.hold_interval:.1f} s")
        end = time.time() + a.hold
        while time.time() < end:
            t0 = time.time()
            r = self.read()
            self.log("hold", float("nan"), float("nan"), float("nan"), None, r, target)
            time.sleep(max(0.0, a.hold_interval - (time.time() - t0)))


def save(records, summary, args, run_dir, ts, quiet=False):
    csv_path = os.path.join(run_dir, f"resistance_tune_{ts}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else ["step"])
        writer.writeheader()
        writer.writerows(records)
    meta = {k: v for k, v in vars(args).items() if k not in ("set_map", "recipe_data")}
    if args.recipe_data is not None:
        meta["recipe_used"] = {**args.recipe_data, "params": {
            **args.recipe_data["params"], "set_map": format_set_map(args.recipe_data["params"].get("set_map", args.set_map))}}
    meta["set_map"] = [{"icc_A": i, "typical_lrs_Ohm": r} for i, r in args.set_map]
    meta["created_at"] = datetime.now().isoformat(timespec="seconds")
    meta["results"] = summary
    with open(os.path.join(run_dir, f"resistance_tune_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if not quiet:
        print(f"[CSV] Saved -> {csv_path}")
    return csv_path


def plot(records, args, img_path, show: bool):
    if not records:
        return
    steps = np.array([r["step"] for r in records])
    res = np.array([r["resistance_Ohm"] for r in records], dtype=float) / 1e3
    actions = [r["action"] for r in records]
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    ax.tick_params(colors="#FFFFFF")
    for spine in ax.spines.values():
        spine.set_color("#FFFFFF")
    ax.grid(True, color="#3A3A3A", lw=0.5, which="both")
    for target in dict.fromkeys(r["target_Ohm"] for r in records):
        idx = steps[[r["target_Ohm"] == target for r in records]]
        ax.fill_between([idx.min() - 0.5, idx.max() + 0.5],
                        target * (1 - args.tol) / 1e3, target * (1 + args.tol) / 1e3,
                        color="#4DB6AC", alpha=0.25, lw=0)
    ax.plot(steps, res, "-", color="#90CAF9", lw=0.8)
    colors = {"read": "#B0BEC5", "set": "#EF5350", "reset": "#FFB74D",
              "verify": "#66BB6A", "hold": "#4DB6AC", "abort": "#FFFFFF"}
    for action, color in colors.items():
        m = np.array([a == action for a in actions])
        if m.any():
            ax.plot(steps[m], res[m], "o", ms=4, color=color, label=action)
    ax.set_yscale("log")
    ax.set_xlabel("Step", color="#FFFFFF")
    ax.set_ylabel("Resistance at +0.1 V (kOhm)", color="#FFFFFF")
    ax.set_title(f"{args.dev} write-verify tuning, tol +/-{args.tol*100:.0f}%", color="#FFFFFF")
    ax.legend(facecolor="#000000", edgecolor="#FFFFFF", labelcolor="#FFFFFF", fontsize=8)
    plt.tight_layout()
    plt.savefig(img_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"[PNG] Saved -> {img_path}")
    if show:
        plt.show()


def plot_summary(summary, args, img_path):
    """Achieved vs target for many targets: shows range, accuracy and failures."""
    t = np.array([r["target_Ohm"] for r in summary]) / 1e3
    f = np.array([r["final_Ohm"] for r in summary], dtype=float) / 1e3
    ok = np.array([r["success"] for r in summary])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#000000")
    for ax in (ax1, ax2):
        ax.set_facecolor("#000000")
        ax.tick_params(colors="#FFFFFF")
        for spine in ax.spines.values():
            spine.set_color("#FFFFFF")
        ax.grid(True, color="#3A3A3A", lw=0.5)
    lim = np.array([min(t.min(), np.nanmin(f)) * 0.97, max(t.max(), np.nanmax(f)) * 1.03])
    ax1.plot(lim, lim, color="#888888", lw=0.8)
    ax1.fill_between(lim, lim * (1 - args.tol), lim * (1 + args.tol), color="#4DB6AC", alpha=0.2, lw=0)
    ax1.plot(t[ok], f[ok], "o", ms=4, color="#66BB6A", label="reached")
    if (~ok).any():
        ax1.plot(t[~ok], f[~ok], "x", ms=6, color="#EF5350", label="not reached")
    ax1.set_xlabel("Target (kOhm)", color="#FFFFFF")
    ax1.set_ylabel("Final device R (kOhm)", color="#FFFFFF")
    ax1.legend(facecolor="#000000", edgecolor="#FFFFFF", labelcolor="#FFFFFF", fontsize=8)
    err = (f - t) / t * 100
    idx = np.arange(1, len(t) + 1)
    ax2.axhspan(-args.tol * 100, args.tol * 100, color="#4DB6AC", alpha=0.2, lw=0)
    ax2.plot(idx[ok], err[ok], "o", ms=4, color="#66BB6A")
    if (~ok).any():
        ax2.plot(idx[~ok], err[~ok], "x", ms=6, color="#EF5350")
    ax2.set_xlabel("Target number", color="#FFFFFF")
    ax2.set_ylabel("Error (%)", color="#FFFFFF")
    rs = f", Rs={args.r_series/1e3:g} kOhm" if args.r_series else ""
    fig.suptitle(f"{args.dev}: {ok.sum()}/{len(t)} targets within +/-{args.tol*100:g}%{rs}", color="#FFFFFF")
    plt.tight_layout()
    plt.savefig(img_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"[PNG] Saved -> {img_path}")


def main():
    args = parse_args()
    if args.save_recipe:
        path = save_recipe(args, args.save_recipe)
        print(f"[Recipe] Saved effective settings for {args.dev} -> {path}")
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = smu.series_tag(args.r_series)
    ts = f"{tag}_{ts}" if tag else ts  # e.g. RS10k_20261009_170000
    if len(args.target) <= 4:
        targets_txt = "_".join(f"{t/1e3:g}k" for t in args.target)
    else:  # keep the folder name short for long target lists
        targets_txt = f"{min(args.target)/1e3:g}-{max(args.target)/1e3:g}k_x{len(args.target)}"
        if args.shuffle:
            targets_txt += "_shuffled"
        if args.rounds > 1:
            targets_txt += f"_{args.rounds}rounds"
    run_dir = os.path.join(OUTPUT_DIR, f"{args.dev}_T{targets_txt}_tol{args.tol*100:g}pct_{ts}")

    print("=" * 68)
    print("  Write-Verify Resistance Tuning")
    print("=" * 68)
    if args.recipe_data is not None:
        print(f"  Recipe       : {args.recipe_data['path']}  (device {args.dev})")
    if len(args.target) <= 8:
        print(f"  Targets      : {', '.join(f'{t/1e3:g} kOhm' for t in args.target)}  (+/-{args.tol*100:g}%)")
    else:
        print(f"  Targets      : {len(args.target)} from {args.target[0]/1e3:.2f} to {args.target[-1]/1e3:.2f} kOhm "
              f"(+/-{args.tol*100:g}%), {'random order' if args.shuffle else 'in the order given'}")
    print(f"  SET          : {args.v_set:+.3g} V, {args.t_set:g} s, Icc from map "
          + ", ".join(f"{i:.1e} A->{r/1e3:.1f} kOhm" for i, r in args.set_map))
    sgn = "-" if args.reset_sign < 0 else "+"
    print(f"  RESET        : {sgn}{args.v_reset_start:.3f} to {sgn}{args.v_reset_max:.3f} V step {args.v_reset_step:.3f} V, "
          f"{args.t_reset:g} s, Icc {args.icc_reset:.1e} A")
    print(f"  Read         : {args.v_read:+.2f} V, Icc {args.icc:.1e} A, avg {args.n_read_avg}")
    if args.r_series:
        print(f"  Series R     : {args.r_series/1e3:g} kOhm subtracted; pulse voltages are applied values")
    print(f"  Limits       : {args.max_pulses} pulses / {args.max_sets} SETs per target, hold {args.hold:g} s")
    print(f"  Output       : {run_dir}")
    print("=" * 68)
    if input("\nConfirm start? (Enter / q to quit): ").strip().lower() == "q":
        return

    try:
        inst = smu.connect_smu(args.address)
    except smu.InstrumentConnectionError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    os.makedirs(run_dir, exist_ok=True)
    tuner = Tuner(inst, args)
    summary = []
    try:
        smu.configure_voltage_source(inst, args.icc, args.nplc)
        n_fail = args.stop_after_fails
        for target in args.target:
            result = tuner.tune(target)
            summary.append(result)
            recent = summary[-2 * n_fail:] if n_fail else []
            misses = sum(not r["success"] for r in recent)
            if n_fail and misses >= n_fail:
                save(tuner.records, summary, args, run_dir, ts, quiet=True)
                raise StopRun(f"{misses} of the last {len(recent)} targets missed: "
                              "treating the device as no longer tunable")
            status = "OK" if result["success"] else "NOT REACHED"
            print(f"  [{status}] {result['final_Ohm']/1e3:.3f} kOhm after "
                  f"{result['pulses']} pulses ({result['sets']} SET)")
            if result["success"]:
                tuner.hold(target)
            save(tuner.records, summary, args, run_dir, ts, quiet=True)  # keep data if a long run dies
    except StopRun as exc:
        print(f"[Safety stop] {exc}")
    except KeyboardInterrupt:
        print("\n[Stop] Ctrl+C received, saving collected data.")
    except Exception as exc:  # e.g. a VISA timeout late in a long run: still save what we have
        print(f"[Error] {type(exc).__name__}: {exc}")
    finally:
        smu.safe_shutdown(inst)
        print("[Safe] Output OFF, voltage returned to 0 V.")
        inst.close()

    save(tuner.records, summary, args, run_dir, ts)
    print("\n[Summary]")
    for r in summary:
        err = (r["final_Ohm"] - r["target_Ohm"]) / r["target_Ohm"] * 100
        print(f"  target {r['target_Ohm']/1e3:7.2f} kOhm -> {r['final_Ohm']/1e3:7.2f} kOhm "
              f"({err:+5.1f}%)  {'OK' if r['success'] else 'NOT REACHED'}  pulses={r['pulses']}")
    ok = [r for r in summary if r["success"]]
    if summary:
        errs = np.array([(r["final_Ohm"] - r["target_Ohm"]) / r["target_Ohm"] * 100 for r in ok])
        print(f"  Reached {len(ok)}/{len(summary)} targets"
              + (f"; error mean {errs.mean():+.2f}%, |error| max {np.abs(errs).max():.2f}%" if len(ok) else ""))
    if args.rounds > 1 and summary:
        n_list = len(args.target) // args.rounds
        print("  Per round (reached / pulses):")
        for k in range(0, len(summary), n_list):
            chunk = summary[k:k + n_list]
            print(f"    round {k // n_list + 1:3d}: {sum(r['success'] for r in chunk):2d}/{len(chunk)}  "
                  f"pulses {sum(r['pulses'] for r in chunk):4d}")
    if not args.no_plot:
        if len(summary) >= 5:
            plot_summary(summary, args, os.path.join(run_dir, f"resistance_tune_summary_{ts}.png"))
        plot(tuner.records, args, os.path.join(run_dir, f"resistance_tune_{ts}.png"), show=True)


if __name__ == "__main__":
    main()
