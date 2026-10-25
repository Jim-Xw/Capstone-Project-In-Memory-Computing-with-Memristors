"""
replot_report_4_1.py - redraw the Section 4.1 figures (T7 Ti/AlOx/TiOx/Ni) in
the white report style defined in report_style.py.

Each figure is saved in the run folder its data came from, next to the
original plot, as Fig4-x?_<name>.png (600 dpi) and .svg. Original files are
not modified.

Run from the repository root or from memristor_measure/:
    python memristor_measure/replot_report_4_1.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report_style as rs  # noqa: E402

T7 = HERE.parent / "Measure_Result_Docs" / "T7_Ti_AlOx_TiOx_Ni"
D31 = T7 / "Device 3-1（Good Behaviour）"
D61 = T7 / "Device 6-1 (Good Behaviour, broken after 1 hour)"
D131 = T7 / "Device 13-1(Good behaviour dead after 2 hours)"
D142 = T7 / "Device 14-2 (Good behaviour dead in the endurance test)"

R_LABEL = "Resistance at 0.1 V (kΩ)"


def one_csv(folder, pattern="*.csv"):
    files = sorted(Path(folder).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} in {folder}")
    return files[0]


# ------------------------------------------------------------ Figure 4.1
def fig_4_1a():
    run = D31 / "IV-Sweep" / "V-1.0_1.0_cyc1_pts100_SET300uA_RST0mA_dur120s_20260924_151212"
    df = pd.read_csv(run / "iv_sweep_20260924_151212.csv", comment="#")
    v = df["voltage_V"].to_numpy()
    i = np.abs(df["current_A"].to_numpy())
    i[i <= 0] = np.nan

    fig, ax = rs.figure("half")
    ax.plot(v, i, label="Cycle 1", **rs.series(0, marker=False))
    rs.ref_line(ax, y=300e-6)
    rs.note(ax, r"$I_{\mathrm{CC}}$ = 300 µA", xy=(0.12, 340e-6), va="bottom",
            xytext=None)
    n = len(v)
    rs.direction_arrows(ax, v, i, [int(n * f) for f in (0.10, 0.36, 0.60, 0.86)],
                        color=rs.BLUE, step=3)
    # SET and RESET markers taken from the data
    neg = np.where((v < 0) & (np.arange(n) > n // 2) & (np.arange(n) < 3 * n // 4))[0]
    k_set = neg[np.nanargmax(np.diff(np.log10(i[neg]), prepend=np.nan))]
    rs.note(ax, f"SET\n{v[k_set]:.2f} V".replace("-", "\u2212"), xy=(v[k_set], 1.3e-4), xytext=(-0.62, 1.5e-6),
            ha="center")
    pos = np.where((v > 0) & (np.arange(n) < n // 2))[0]
    k_rst = pos[np.nanargmax(i[pos])]
    rs.note(ax, f"RESET\nfrom +{v[k_rst]:.1f} V", xy=(v[k_rst], i[k_rst]),
            xytext=(0.62, 1.5e-6), ha="center")
    rs.log_axis(ax, "y", labels=(1,))
    ax.set_ylim(1e-9, 1e-3)
    ax.set_xlim(-1.1, 1.1)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("|Current| (A)")
    ax.yaxis.set_major_formatter(rs.mpl.ticker.LogFormatterMathtext())
    rs.legend(ax, loc="lower right")
    rs.panel_label(ax, "a")
    return rs.save(fig, run / "Fig4-1a_IV_3-1_report")


def fig_4_1b():
    run = (D142 / "IV_sweep" /
           "V-1.7_1.8_cyc3_pts50_SET150uA_RST150uA_NPLC0.5_RS10k_20261016_154450")
    df = pd.read_csv(run / "iv_sweep_RS10k_20261016_154450.csv")
    fig, ax = rs.figure("half")
    for c in sorted(df["cycle"].unique()):
        d = df[df["cycle"] == c]
        v = d["device_voltage_V"].to_numpy()
        i = np.abs(d["current_A"].to_numpy())
        i[i <= 0] = np.nan
        ax.plot(v, i, label=f"Cycle {c + 1}", **rs.series(int(c), marker=False))
        if c == 0:
            n = len(v)
            rs.direction_arrows(ax, v, i, [int(n * f) for f in (0.10, 0.36, 0.60, 0.86)],
                                color=rs.BLUE, step=3)
    rs.ref_line(ax, y=150e-6)
    rs.note(ax, r"$I_{\mathrm{CC}}$ = 150 µA", xy=(-0.80, 170e-6), va="bottom")
    rs.note(ax, "SET\n+1.0 V", xy=(0.90, 7.7e-5), xytext=(0.88, 1.5e-6), ha="center")
    rs.note(ax, "RESET\nfrom \u22120.7 V", xy=(-0.745, 6.5e-5), xytext=(-0.55, 1.5e-6),
            ha="center")
    rs.log_axis(ax, "y", labels=(1,))
    ax.yaxis.set_major_formatter(rs.mpl.ticker.LogFormatterMathtext())
    ax.set_ylim(1e-9, 1e-3)
    ax.set_xlim(-0.85, 1.1)
    ax.set_xlabel("Device voltage (V)")
    ax.set_ylabel("|Current| (A)")
    rs.legend(ax, loc="lower right")
    rs.panel_label(ax, "b")
    return rs.save(fig, run / "Fig4-1b_IV_14-2_report")


# ------------------------------------------------------------ Figure 4.2
def fig_4_2a():
    run = (D61 / "Pulse_Modulation" /
           "Device6-1_POT-1.00_DEP+1.00_pulse100x100ms_N1x1_cyc5_20261008_132312")
    df = pd.read_csv(one_csv(run))
    df["R"] = df["resistance_Ohm"] / 1e3
    fig, ax = rs.figure("half")
    ax.plot(df["global_pulse"], df["R"], **rs.connector())
    b = df[df["phase"] == "baseline"]
    s = df[df["phase"] == "potentiation"]
    r = df[df["phase"] == "depression"]
    ax.plot(b["global_pulse"], b["R"], label="Initial read", **rs.role("read"))
    ax.plot(s["global_pulse"], s["R"], label="After SET (−1.0 V, 0.1 s)", **rs.role("set"))
    ax.plot(r["global_pulse"], r["R"], label="After RESET (+1.0 V, 0.1 s)", **rs.role("reset"))
    rs.log_axis(ax, "y")
    ax.set_ylim(5, 200)
    ax.set_xlim(-0.5, df["global_pulse"].max() + 0.5)
    ax.xaxis.set_major_locator(rs.mpl.ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(rs.mpl.ticker.MultipleLocator(1))
    ax.set_xlabel("Pulse number")
    ax.set_ylabel(R_LABEL)
    rs.legend(ax, loc="upper left", ncol=1)
    rs.panel_label(ax, "a")
    return rs.save(fig, run / "Fig4-2a_pulse_cycles_6-1_report")


def fig_4_2b():
    run13 = (D131 / "Pulse_Modulation" /
             "Device13-1_POT+2.00_DEP-2.00_pulse500x500ms_N0x20_cyc1_RS10k_20261016_131018")
    run14 = (D142 / "Pulse_Modulation" /
             "Device14-2_POT+2.00_DEP-1.80_pulse500x200ms_N0x15_cyc1_RS10k_20261016_155217")
    fig, ax = rs.figure("half")
    # Both curves are RESET responses, so both carry the RESET colour and are
    # told apart by marker and line style.
    styles = [
        dict(marker="o", linestyle="-", markerfacecolor=rs.BLUE, markeredgecolor="white"),
        dict(marker="s", linestyle="--", markerfacecolor="white", markeredgecolor=rs.BLUE,
             markeredgewidth=0.8),
    ]
    labels = ["13-1 (−2.0 V, 0.5 s)", "14-2 (−1.8 V, 0.2 s)"]
    for run, st, lab in zip((run13, run14), styles, labels):
        df = pd.read_csv(one_csv(run))
        ax.plot(df["pulse_index"], df["resistance_Ohm"] / 1e3, color=rs.BLUE,
                label=lab, **st)
    ax.set_xlim(-0.8, 20.8)
    ax.set_ylim(20, 140)
    ax.xaxis.set_major_locator(rs.mpl.ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(rs.mpl.ticker.MultipleLocator(1))
    ax.set_xlabel("RESET pulse number")
    ax.set_ylabel(R_LABEL)
    rs.legend(ax, loc="lower right", title="Device (applied pulse)", title_fontsize=7.5)
    rs.panel_label(ax, "b")
    return rs.save(fig, run13 / "Fig4-2b_reset_staircase_13-1_14-2_report")


# ------------------------------------------------------------ Figure 4.3
def fig_4_3a():
    run = (D142 / "Resistance_Tune" /
           "Device14-2_T60-110k_x10_tol3pct_RS10k_20261016_155551")
    df = pd.read_csv(one_csv(run))
    df["R"] = df["resistance_Ohm"] / 1e3
    fig, ax = rs.figure("full", height_cm=7.0)
    tol = 0.03
    for tgt, g in df.groupby("target_Ohm", sort=False):
        lo, hi = g["step"].min() - 0.5, g["step"].max() + 0.5
        t = tgt / 1e3
        rs.band(ax, t * (1 - tol), t * (1 + tol), xmin=lo, xmax=hi)
        ax.plot([lo, hi], [t, t], color=rs.GREY, lw=0.5, zorder=0.5)
        ax.text((lo + hi) / 2, t * (1 + tol) + 1.2, f"{t:.0f}", ha="center", va="bottom",
                fontsize=7, color=rs.GREY)
    ax.plot(df["step"], df["R"], **rs.connector())
    groups = [("read", ["read"], "Read"),
              ("set", ["set"], "After SET"),
              ("reset", ["reset"], "After RESET"),
              ("verify", ["verify", "hold"], "Verify and hold reads")]
    for rname, acts, lab in groups:
        d = df[df["action"].isin(acts)]
        ax.plot(d["step"], d["R"], label=lab, **rs.role(rname))
    rs.note(ax, "103 and 110 kΩ\nnot reached", xy=(366, 83.5), xytext=(352, 70),
            ha="center")
    ax.set_xlim(0, df["step"].max() + 3)
    ax.set_ylim(30, 125)
    ax.set_xlabel("Step (pulse or read)")
    ax.set_ylabel(R_LABEL)
    rs.legend(ax, loc="lower right", ncol=4, columnspacing=1.2)
    rs.panel_label(ax, "a", dx_pt=-30)
    return rs.save(fig, run / "Fig4-3a_tune_10_levels_14-2_report")


def fig_4_3b():
    run = (D142 / "Resistance_Tune" /
           "Device14-2_T40-95k_x24_shuffled_2rounds_tol3pct_RS10k_20261016_161040")
    meta = json.loads(one_csv(run, "*.json").read_text(encoding="utf-8"))
    res = pd.DataFrame(meta["results"])
    n_per_round = len(res) // int(meta.get("rounds", 2))
    res["round"] = np.where(res.index < n_per_round, 1, 2)
    res["t"] = res["target_Ohm"] / 1e3
    res["f"] = res["final_Ohm"] / 1e3
    tol = float(meta.get("tol", 0.03))

    fig, ax = rs.figure("half", height_cm=7.0)
    x = np.linspace(35, 100, 50)
    ax.fill_between(x, x * (1 - tol), x * (1 + tol), color=rs.BAND, lw=0, zorder=0)
    ax.plot(x, x, color=rs.GREY, lw=0.6, zorder=0.5)
    shapes = {1: "o", 2: "s"}
    for rnd in (1, 2):
        for ok in (True, False):
            d = res[(res["round"] == rnd) & (res["success"] == ok)]
            if ok:
                kw = dict(markerfacecolor=rs.GREEN, markeredgecolor="white",
                          markeredgewidth=0.4, color=rs.GREEN)
            else:
                kw = dict(markerfacecolor="white", markeredgecolor=rs.INK,
                          markeredgewidth=0.8, color=rs.INK)
            n_ok = int(((res["round"] == rnd) & res["success"]).sum())
            lab = (f"Round {rnd}, reached ({n_ok}/{n_per_round})" if ok
                   else f"Round {rnd}, not reached")
            ax.plot(d["t"], d["f"], linestyle="none", marker=shapes[rnd],
                    markersize=4.5 if rnd == 1 else 4.0, label=lab, **kw)
    rs.note(ax, "Line: final = target\nShading: ±3% of target", xy=(37.5, 97.5),
            ha="left", va="top")
    ax.set_xlim(35, 100)
    ax.set_ylim(35, 100)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Target resistance (kΩ)")
    ax.set_ylabel("Final resistance (kΩ)")
    rs.legend(ax, loc="lower right")
    rs.panel_label(ax, "b")
    return rs.save(fig, run / "Fig4-3b_shuffled_targets_14-2_report")


# ------------------------------------------------------------ Figure 4.4
RET = D131 / "Retention_Unbiased"


def _ret(stamp, dur):
    run = RET / f"Device13-1_present_dur{dur}s_himp_RS10k_20261016_{stamp}"
    df = pd.read_csv(one_csv(run))
    df = df[df["t_since_program_s"] > 0]
    return run, df["t_since_program_s"].to_numpy(), df["drift_pct"].to_numpy()


def fig_4_4a():
    sets = [("131117", 3600, "63.6 kΩ (3600 s)"),
            ("120242", 120, "73.6 kΩ (120 s)"),
            ("122149", 120, "90.9 kΩ (120 s)")]
    fig, ax = rs.figure("half")
    rs.band(ax, -5, 5)
    rs.note(ax, "±5% band", xy=(1.2, 4.0), ha="left", va="center")
    out_run = None
    for k, (stamp, dur, lab) in enumerate(sets):
        run, t, d = _ret(stamp, dur)
        out_run = out_run or run
        ax.plot(t, d, label=lab, **rs.series(k))
    ax.axhline(0, color=rs.GREY, lw=0.5, zorder=0.5)
    rs.log_axis(ax, "x", labels=(1,))
    ax.xaxis.set_major_formatter(rs.mpl.ticker.FuncFormatter(lambda v, p: f"{v:g}"))
    ax.set_xlim(0.8, 5000)
    ax.set_ylim(-6, 6)
    ax.set_xlabel("Time after programming (s)")
    ax.set_ylabel(r"Change from $R_0$ (%)")
    rs.legend(ax, loc="lower left", title=r"$R_0$ (hold time)", title_fontsize=7.5)
    rs.panel_label(ax, "a")
    return rs.save(fig, out_run / "Fig4-4a_retention_stable_13-1_report")


def fig_4_4b():
    sets = [("123544", 3600, "70.6 kΩ, low state"),
            ("115042", 600, "41.0 kΩ, 0.1 s SET"),
            ("142147", 3600, "129.5 kΩ, high state")]
    fig, ax = rs.figure("half")
    rs.band(ax, -5, 5)
    out_run = None
    for k, (stamp, dur, lab) in enumerate(sets):
        run, t, d = _ret(stamp, dur)
        if stamp == "142147":
            out_run = run
        ax.plot(t, d, label=lab, **rs.series(k))
    ax.axhline(0, color=rs.GREY, lw=0.5, zorder=0.5)
    rs.log_axis(ax, "x", labels=(1,))
    ax.xaxis.set_major_formatter(rs.mpl.ticker.FuncFormatter(lambda v, p: f"{v:g}"))
    ax.set_xlim(0.8, 5000)
    ax.set_ylim(-100, 250)
    ax.yaxis.set_major_locator(rs.mpl.ticker.MultipleLocator(50))
    ax.set_xlabel("Time after programming (s)")
    ax.set_ylabel(r"Change from $R_0$ (%)")
    rs.legend(ax, loc="upper right", title=r"$R_0$, state", title_fontsize=7.5)
    rs.panel_label(ax, "b")
    return rs.save(fig, out_run / "Fig4-4b_retention_unstable_13-1_report")


# ------------------------------------------------------------ Figure 4.5
def fig_4_5():
    run = (D142 / "Pulse_Modulation" /
           "Device14-2_POT+2.00_DEP-2.00_pulse500x200ms_N1x10_cyc100_RS10k_20261016_162523")
    df = pd.read_csv(one_csv(run))
    df = df[df["cycle"] > 0].copy()
    df["R"] = df["resistance_Ohm"] / 1e3
    n_per = df.groupby("cycle").size().max()
    df["x"] = df["cycle"] - 1 + df.groupby("cycle").cumcount() / n_per + 1 / n_per

    s = df[df["phase"] == "potentiation"].groupby("cycle").first()
    r = df[df["phase"] == "depression"].groupby("cycle").last()

    fig, ax = rs.figure("full", height_cm=6.5)
    ax.plot(df["x"], df["R"], label="All reads", color=rs.LIGHT, lw=0.6, zorder=1)
    ax.plot(s["x"], s["R"], label="After SET (+2.0 V, 0.5 s)", **rs.role("set", markersize=3.6))
    ax.plot(r["x"], r["R"], label="After 10th RESET (−2.0 V, 0.2 s)",
            **rs.role("reset", markersize=3.6))
    for c in (31, 56, 73):
        rs.ref_line(ax, x=c - 0.5, lw=0.6)
    rs.note(ax, "Drops during RESET\n(cycles 31, 32)", xy=(29.7, 3.0), ha="right",
            va="center", xytext=None)
    rs.note(ax, "Drop during RESET\n(cycle 56)", xy=(54.7, 2.0), ha="right", va="center",
            xytext=None)
    rs.note(ax, "Permanent short,\n0.92 kΩ (cycle 73)", xy=(73.3, 2.0), ha="left",
            va="center", xytext=None)
    rs.log_axis(ax, "y")
    ax.set_ylim(0.5, 150)
    ax.set_xlim(0, 100.5)
    ax.xaxis.set_major_locator(rs.mpl.ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(rs.mpl.ticker.MultipleLocator(2))
    ax.set_xlabel("Cycle number")
    ax.set_ylabel(R_LABEL)
    rs.legend(ax, loc="lower left", ncol=1)
    return rs.save(fig, run / "Fig4-5_endurance_14-2_report")


ALL = [fig_4_1a, fig_4_1b, fig_4_2a, fig_4_2b, fig_4_3a, fig_4_3b,
       fig_4_4a, fig_4_4b, fig_4_5]

if __name__ == "__main__":
    rs.apply()
    only = set(sys.argv[1:])
    for f in ALL:
        if only and f.__name__ not in only:
            continue
        for p in f():
            print(p.relative_to(T7))
