"""
plot_write_verify_flowchart.py - flowchart of the write-verify tuning loop in
resistance_tune.py (Tuner.tune), drawn in the white report style.

Output: Report_Figures/Fig3-x_write_verify_flowchart.png (600 dpi) and .svg
in the repository root. Run: python memristor_measure/plot_write_verify_flowchart.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report_style as rs  # noqa: E402

OUT = HERE.parent / "Report_Figures"
if len(sys.argv) > 1:
    OUT = Path(sys.argv[1])

FS = 8.0      # box text (pt)
FS_EDGE = 7.5  # yes / no labels
LW = 0.8

L, M, R = 2.55, 7.4, 12.75   # column centres (cm)
X_RET = 15.6                 # return line on the right
Y_JOIN = 11.45               # junction above the first decision

TINT = {
    "read": ("#f1f0ec", rs.MUTED),
    "set": ("#fde8df", rs.ORANGE),
    "reset": ("#e3eefb", rs.BLUE),
    "verify": ("#ddf2ea", rs.GREEN),
    "stop": ("#e8e7e2", rs.GREY),
    "plain": ("white", rs.INK),
}


def box(ax, x, y, w, h, lines, role="plain", rounded=0.12, bold_first=False,
        dashed=False, align="center"):
    face, edge = TINT[role]
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rounded}",
                       fc=face, ec=edge, lw=LW, ls=(0, (3, 2)) if dashed else "-",
                       zorder=2)
    ax.add_patch(p)
    n = len(lines)
    dy = 0.36
    for k, t in enumerate(lines):
        yy = y + (n - 1) / 2 * dy - k * dy
        xx = x if align == "center" else x - w / 2 + 0.25
        ax.text(xx, yy, t, ha=align if align != "center" else "center", va="center",
                fontsize=FS, color=rs.INK, zorder=3,
                fontweight="bold" if (bold_first and k == 0) else "normal")
    return dict(l=x - w / 2, r=x + w / 2, t=y + h / 2, b=y - h / 2, x=x, y=y)


def diamond(ax, x, y, w, h, lines):
    pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    ax.add_patch(Polygon(pts, closed=True, fc="white", ec=rs.INK, lw=LW, zorder=2))
    n = len(lines)
    dy = 0.34
    for k, t in enumerate(lines):
        ax.text(x, y + (n - 1) / 2 * dy - k * dy, t, ha="center", va="center",
                fontsize=FS, color=rs.INK, zorder=3)
    return dict(l=x - w / 2, r=x + w / 2, t=y + h / 2, b=y - h / 2, x=x, y=y)


def arrow(ax, pts, head=True):
    """Polyline through pts with an arrowhead on the last segment."""
    xs, ys = zip(*pts)
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=rs.INK, lw=LW, solid_capstyle="butt", zorder=1)
    if head:
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>", color=rs.INK, lw=LW,
                                    mutation_scale=8, shrinkA=0, shrinkB=0), zorder=1)
    else:
        ax.plot(xs[-2:], ys[-2:], color=rs.INK, lw=LW, zorder=1)


def label(ax, x, y, t, ha="left", va="center"):
    ax.text(x, y, t, ha=ha, va=va, fontsize=FS_EDGE, color=rs.GREY, zorder=3)


def main():
    rs.apply()
    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(16 * rs.CM, 12.6 * rs.CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(1.75, 14.35)
    ax.set_aspect("equal")
    ax.axis("off")

    start = box(ax, M, 13.5, 5.6, 0.75,
                [r"Target $T$, tolerance band $T(1 \pm \mathrm{tol})$"], rounded=0.37)
    read = box(ax, M, 12.3, 5.6, 0.75,
               ["Read $R$ at +0.1 V (mean of 10 samples)"], role="read")
    d1 = diamond(ax, M, 10.4, 3.4, 1.3, ["$R$ in band?"])
    wait = box(ax, R, 10.4, 4.6, 0.75, ["Wait 2 s, read again"], role="verify")
    d2 = diamond(ax, R, 8.6, 3.4, 1.3, ["Still in band?"])
    done = box(ax, R, 6.9, 4.6, 0.9, ["Target reached", "(optional hold reads)"],
               role="verify", rounded=0.37, bold_first=True)
    d3 = diamond(ax, M, 8.6, 3.4, 1.3, ["$R$ above band?"])
    setb = box(ax, L, 8.6, 4.5, 1.25,
               ["SET pulse (0.5 s), then read",
                r"$I_\mathrm{CC}$ from calibration table,",
                "so the LRS lands below the band"], role="set")
    reset = box(ax, M, 6.9, 5.2, 0.95,
                [r"RESET pulse (20 ms) at $V_\mathrm{R}$, then read",
                 r"$V_\mathrm{R}$ starts low (0.75 V by default)"], role="reset")
    d4 = diamond(ax, M, 5.2, 4.4, 1.5, ["Compliance hit, or", "$R$ fell by > 50%?"])
    stop = box(ax, R, 5.2, 4.6, 0.9, ["Stop run", "(reversed polarity or unwanted SET)"],
               role="stop", rounded=0.37, bold_first=True)
    adj = box(ax, M, 3.15, 5.0, 1.5,
              [r"Set $V_\mathrm{R}$ for the next RESET:",
               r"overshoot above band: $V_\mathrm{R}$ $-$ 20 mV",
               r"rise < 2%: $V_\mathrm{R}$ + 10 mV (up to $V_\mathrm{max}$)",
               r"otherwise: keep $V_\mathrm{R}$"], role="reset")
    box(ax, L, 5.35, 4.3, 1.65,
        ["Limits per target", "60 pulses and 6 SETs at most",
         r"3 stalls at $V_\mathrm{max}$: not reached",
         "first read < 1 kΩ: stop (short)"],
        role="plain", dashed=True, bold_first=True, align="left")

    # main column
    arrow(ax, [(M, start["b"]), (M, read["t"])])
    arrow(ax, [(M, read["b"]), (M, Y_JOIN)], head=False)
    arrow(ax, [(M, Y_JOIN), (M, d1["t"])])
    arrow(ax, [(M, d1["b"]), (M, d3["t"])])
    label(ax, M + 0.12, (d1["b"] + d3["t"]) / 2, "No")
    arrow(ax, [(M, d3["b"]), (M, reset["t"])])
    label(ax, M + 0.12, (d3["b"] + reset["t"]) / 2, "No (below band)")
    arrow(ax, [(M, reset["b"]), (M, d4["t"])])
    arrow(ax, [(M, d4["b"]), (M, adj["t"])])
    label(ax, M + 0.12, (d4["b"] + adj["t"]) / 2, "No")

    # verify branch
    arrow(ax, [(d1["r"], d1["y"]), (wait["l"], wait["y"])])
    label(ax, d1["r"] + 0.15, d1["y"] + 0.2, "Yes", va="bottom")
    arrow(ax, [(R, wait["b"]), (R, d2["t"])])
    arrow(ax, [(R, d2["b"]), (R, done["t"])])
    label(ax, R + 0.12, (d2["b"] + done["t"]) / 2, "Yes")
    label(ax, d2["r"] + 0.12, d2["y"] + 0.2, "No", va="bottom")

    # SET branch
    arrow(ax, [(d3["l"], d3["y"]), (setb["r"], setb["y"])])
    label(ax, d3["l"] - 0.15, d3["y"] + 0.2, "Yes", ha="right", va="bottom")
    arrow(ax, [(L, setb["t"]), (L, Y_JOIN), (M - 0.04, Y_JOIN)], head=False)

    # safety stop
    arrow(ax, [(d4["r"], d4["y"]), (stop["l"], stop["y"])])
    label(ax, d4["r"] + 0.15, d4["y"] + 0.2, "Yes", va="bottom")

    # return line on the right: from the V_R adjustment and from a failed confirm
    arrow(ax, [(adj["r"], adj["y"]), (X_RET, adj["y"]), (X_RET, Y_JOIN),
               (M + 0.04, Y_JOIN)], head=False)
    arrow(ax, [(d2["r"], d2["y"]), (X_RET, d2["y"])], head=False)
    for (x, y) in [(M, Y_JOIN), (X_RET, d2["y"])]:
        ax.plot([x], [y], marker="o", ms=2.6, color=rs.INK, zorder=4)
    label(ax, X_RET - 0.12, Y_JOIN + 0.12, "with the new $R$", ha="right", va="bottom")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "Fig3-x_write_verify_flowchart"
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
