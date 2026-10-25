"""
report_style.py - white-background figure style for the EMI Capstone report.

One shared style for every figure with dots and curves (I-V sweeps, pulse
trains, resistance tuning, retention, endurance). Import it in any plotting
script instead of setting colours by hand:

    import report_style as rs
    rs.apply()                                   # set rcParams once
    fig, ax = rs.figure("half")                  # 8.0 x 6.0 cm panel
    ax.plot(v, i, label="Cycle 1", **rs.series(0, marker=False))
    rs.panel_label(ax, "a")
    rs.save(fig, out_dir / "Fig4-1a_IV")         # writes .png (600 dpi) and .svg

Rules of the style
------------------
Canvas
    White figure and axes, black ink, closed axis box, ticks pointing inwards
    on all four sides, minor ticks on. No gridlines and no chart titles: the
    Word caption carries the title, and panels are labelled (a), (b), ...
Text
    Times New Roman (the report body font), 8 pt tick labels, 9 pt axis
    labels, 7.5 pt legends and annotations. Axis labels are written as
    "Quantity (unit)", e.g. "Resistance at 0.1 V (kΩ)".
Size
    "half" = 8.0 x 6.0 cm (two panels side by side), "full" = 16.0 x 6.5 cm.
    Insert figures into Word at 100 % so the text size matches across figures.
Series (identity: cycles, devices, states)
    Fixed order: blue, orange, green. Each slot also has its own marker and
    line style, so the figure still reads when printed in greyscale or by a
    colour-blind reader. Use at most three series per panel; split into
    panels beyond that. Colours are never cycled or re-ordered.
Programming roles (when colour shows what was done to the device)
    SET / low-resistance state  -> orange, downward triangle
    RESET / high-resistance state -> blue, upward triangle
    plain read                    -> grey, open circle
    verified / hold read          -> green, small circle
    Do not mix role colours and series colours in the same figure.
Reference marks
    Compliance limits, identity lines and event markers are dark grey dashed
    0.8 pt lines. Tolerance bands are a light grey fill behind the data.
Marks
    Lines 1.2 pt; markers 3.5 pt with a thin white edge so overlapping points
    stay visible. Dense data (I-V sweeps) are drawn as lines without markers,
    or with markers on every n-th point only.

Colour values: blue, orange and green were checked as a set against a white
background for colour-vision-deficiency separation (all pairs) and contrast
(all >= 3:1). The green is a darker step than a typical screen palette so it
holds up on paper.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

# ---------------------------------------------------------------- colours
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#159a6b"
INK = "#000000"        # axes, ticks and text
GREY = "#52514e"       # reference lines, annotations
MUTED = "#898781"      # plain reads, connecting lines
LIGHT = "#c3c2b7"      # faint traces drawn behind the main data
BAND = "#e8e7e2"       # tolerance-band fill

SERIES = [
    dict(color=BLUE, marker="o", linestyle="-"),
    dict(color=ORANGE, marker="s", linestyle="--"),
    dict(color=GREEN, marker="^", linestyle=":"),
]

ROLES = {
    "set": dict(color=ORANGE, marker="v", markersize=4.0),
    "reset": dict(color=BLUE, marker="^", markersize=4.0),
    "read": dict(color=MUTED, marker="o", markersize=3.5, markerfacecolor="white"),
    "verify": dict(color=GREEN, marker="o", markersize=2.6),
}

SIZES_CM = {"half": (8.0, 6.0), "full": (16.0, 6.5)}
CM = 1 / 2.54

FONT_SERIF = ["Times New Roman", "Liberation Serif", "TeX Gyre Termes",
              "Nimbus Roman", "STIXGeneral", "DejaVu Serif"]


def apply():
    """Set the report rcParams. Call once before creating figures."""
    mpl.rcParams.update({
        # canvas
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        # text
        "font.family": "serif",
        "font.serif": FONT_SERIF,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": True,
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.labelpad": 3,
        # axes and ticks
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.xmargin": 0.03,
        "axes.ymargin": 0.05,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=[s["color"] for s in SERIES]),
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.pad": 3,
        "ytick.major.pad": 3,
        # marks
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "lines.markeredgewidth": 0.4,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "scatter.edgecolors": "white",
        # legend
        "legend.frameon": False,
        "legend.handlelength": 2.0,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.4,
        "legend.labelspacing": 0.3,
        # output
        "svg.fonttype": "path",
        "pdf.fonttype": 42,
    })


def figure(width="half", height_cm=None, nrows=1, ncols=1, **kw):
    """Create a figure at report size. width: 'half', 'full' or a width in cm."""
    if isinstance(width, str):
        w_cm, h_cm = SIZES_CM[width]
    else:
        w_cm, h_cm = float(width), SIZES_CM["half"][1]
    if height_cm is not None:
        h_cm = height_cm
    return plt.subplots(nrows, ncols, figsize=(w_cm * CM, h_cm * CM), **kw)


def series(i, marker=True, line=True, markevery=None, filled=True, **extra):
    """Plot kwargs for series slot i (0, 1, 2) in the fixed order."""
    if i >= len(SERIES):
        raise ValueError("Use at most three series per panel; split into panels.")
    s = SERIES[i]
    kw = dict(color=s["color"])
    kw["linestyle"] = s["linestyle"] if line else "none"
    if marker:
        kw.update(marker=s["marker"], markeredgecolor="white" if filled else s["color"],
                  markerfacecolor=s["color"] if filled else "white")
        if not filled:
            kw["markeredgewidth"] = 0.8
        if markevery is not None:
            kw["markevery"] = markevery
    else:
        kw["marker"] = "none"
    kw.update(extra)
    return kw


def role(name, **extra):
    """Marker-only plot kwargs for a programming role: set, reset, read, verify."""
    r = dict(ROLES[name])
    kw = dict(linestyle="none", markeredgecolor="white", markerfacecolor=r["color"])
    kw.update(r)
    if name == "read":
        kw.update(markeredgecolor=MUTED, markeredgewidth=0.7)
    kw.update(extra)
    return kw


def connector(**extra):
    """Thin grey line that joins consecutive reads behind the role markers."""
    kw = dict(color=LIGHT, linewidth=0.6, linestyle="-", marker="none", zorder=1)
    kw.update(extra)
    return kw


def ref_line(ax, y=None, x=None, label=None, text_kw=None, **extra):
    """Dashed dark-grey reference line (compliance limit, event, identity)."""
    kw = dict(color=GREY, linewidth=0.8, linestyle=(0, (4, 2)), zorder=1.5)
    kw.update(extra)
    if y is not None:
        ln = ax.axhline(y, **kw)
    else:
        ln = ax.axvline(x, **kw)
    if label:
        ln.set_label(label)
    return ln


def band(ax, lo, hi, axis="y", xmin=None, xmax=None, **extra):
    """Light grey tolerance band behind the data."""
    kw = dict(color=BAND, linewidth=0, zorder=0)
    kw.update(extra)
    if axis == "y" and xmin is None:
        return ax.axhspan(lo, hi, **kw)
    if axis == "y":
        return ax.fill_between([xmin, xmax], lo, hi, **kw)
    return ax.axvspan(lo, hi, **kw)


def note(ax, text, xy, xytext=None, arrow=True, **extra):
    """Annotation in grey ink with an optional thin arrow."""
    kw = dict(fontsize=7.5, color=GREY, ha="left", va="center")
    kw.update(extra)
    if xytext is None:
        return ax.annotate(text, xy=xy, **kw)
    arrowprops = dict(arrowstyle="-|>", color=GREY, lw=0.6, mutation_scale=6,
                      shrinkA=1, shrinkB=2) if arrow else None
    return ax.annotate(text, xy=xy, xytext=xytext, arrowprops=arrowprops, **kw)


def direction_arrows(ax, x, y, indices, color=None, step=2, **extra):
    """Small arrowheads along a curve at the given point indices (sweep direction)."""
    color = color or GREY
    for i in indices:
        j = min(i + step, len(x) - 1)
        ax.annotate("", xy=(x[j], y[j]), xytext=(x[i], y[i]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8,
                                    mutation_scale=7, shrinkA=0, shrinkB=0),
                    zorder=4, **extra)


def panel_label(ax, letter, dx_pt=-30, dy_pt=2):
    """Bold (a), (b) ... label above the top-left corner of the axes."""
    return ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction",
                       xytext=(dx_pt, dy_pt), textcoords="offset points",
                       ha="left", va="bottom", fontsize=9, fontweight="bold")


def _plain_number(v, _pos=None):
    if v <= 0:
        return ""
    if v >= 1:
        return f"{v:g}"
    return f"{v:.2g}"


def log_axis(ax, axis="y", labels=(1, 2, 5)):
    """Log scale with plain-number labels at 1, 2, 5 x 10^n (no 10^n notation)."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    getattr(ax, f"set_{axis}scale")("log")
    a.set_major_locator(LogLocator(base=10, subs=labels))
    a.set_major_formatter(FuncFormatter(_plain_number))
    a.set_minor_locator(LogLocator(base=10, subs=[s for s in range(1, 10) if s not in labels]))
    a.set_minor_formatter(NullFormatter())


def legend(ax, **extra):
    kw = dict(loc="best")
    kw.update(extra)
    return ax.legend(**kw)


def save(fig, stem, formats=("png", "svg")):
    """Save the figure next to its data as <stem>.png (600 dpi) and <stem>.svg."""
    stem = Path(stem)
    out = []
    for fmt in formats:
        p = stem.with_suffix(f".{fmt}")
        fig.savefig(p)
        out.append(p)
    plt.close(fig)
    return out
