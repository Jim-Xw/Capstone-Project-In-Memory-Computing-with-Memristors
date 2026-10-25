"""Read a Keithley reading buffer and export the memristor measurement data.

Wiring: this script only reads an existing buffer. If the instrument is still connected
to a device, FORCE HI should go to the top electrode and FORCE LO to the bottom
electrode; never rewire while the output is on. Parameter/command example:
``python buffer_read.py --buffer defbuffer2 --output buffer_read/my_data``;
``--address`` selects the instrument. Output: I-V/resistance PNG and CSV. The original
six CSV columns keep their order, and the set/measured voltage, Icc, compliance, source
status and stop reason are appended. The script first queries SOUR,READ,REL,SOURSTAT and
also supports older buffers/firmware that return only SOUR,READ,REL. Safety warning:
this script does not raise any safety limit, but reading past data does not show that
the wiring or Icc were safe at the time; confirm the output is off before operating the
instrument. Series resistor: with ``--r-series <Ohm>``, the resistance is the device
alone (reading minus Rs) and the file name carries an RS tag before the timestamp.
"""

import argparse
import time
import sys
import os
from datetime import datetime
import numpy as np
import matplotlib
import platform
_fonts = {'Windows': 'Microsoft YaHei', 'Darwin': 'PingFang SC', 'Linux': 'WenQuanYi Micro Hei'}
matplotlib.rcParams['font.family'] = _fonts.get(platform.system(), 'DejaVu Sans')
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams.update({
    "axes.labelsize": 12, "axes.titlesize": 12, "axes.linewidth": 1.2,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "xtick.minor.width": 0.9, "ytick.minor.width": 0.9,
})
matplotlib.use("TkAgg")          # TkAgg with a display; use "Agg" when headless
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

try:
    from .keithley_smu import (
        AUTO_ADDRESS, InstrumentConnectionError, add_series_argument, connect_smu,
        device_voltage, series_tag, validate_series,
    )
except ImportError:
    from keithley_smu import (
        AUTO_ADDRESS, InstrumentConnectionError, add_series_argument, connect_smu,
        device_voltage, series_tag, validate_series,
    )

# ══════════════════════════════════════════════════════
#  Adjustable parameters
# ══════════════════════════════════════════════════════
DEFAULT_VISA_ADDRESS = AUTO_ADDRESS
DEFAULT_BUFFER       = "defbuffer1"   # options: defbuffer1 / defbuffer2 / a custom name
OUTPUT_PREFIX        = "buffer_read/buffer_data"  # output file prefix (no extension)
# ══════════════════════════════════════════════════════


def connect(address: str):
    try:
        return connect_smu(address)
    except InstrumentConnectionError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


def read_buffer(inst, buf_name: str) -> dict:
    """
    Read all data from the given buffer.
    Returns a dict: {voltages, currents, timestamps, n_points}

    2450/2461 SCPI syntax notes:
        - the buffer name must be in double quotes: "defbuffer1"
        - column keywords: SOURce / READing / RELative
        - all columns are queried at once to reduce round trips and avoid timeouts
    """
    # Query the actual number of points in the buffer (name in double quotes)
    n_str = inst.query(f':TRAC:ACT? "{buf_name}"').strip()
    try:
        n = int(float(n_str))
    except ValueError:
        print(f"[ERROR] Cannot parse the point count: '{n_str}'")
        sys.exit(1)

    if n == 0:
        print(f"[WARNING] Buffer '{buf_name}' is empty (0 points)")
        sys.exit(0)

    print(f"[Buffer] '{buf_name}' holds {n} points, reading...")

    # Adjust the timeout to the size: about 50 ms per point, at least 10 s
    inst.timeout = max(10000, n * 50 + 5000)

    query4 = (
        f':TRAC:DATA? 1, {n}, "{buf_name}", '
        "SOUR, READ, REL, SOURSTAT"
    )
    try:
        raw = inst.query(query4).strip()
        vals = [float(x) for x in raw.split(",")]
        columns = 4 if len(vals) >= n * 4 else 3
    except Exception as exc:
        print(f"[Buffer] Four-column query unavailable; using the old three-column format: {exc}")
        raw = inst.query(
            f':TRAC:DATA? 1, {n}, "{buf_name}", SOUR, READ, REL'
        ).strip()
        vals = [float(x) for x in raw.split(",")]
        columns = 3

    if len(vals) != n * columns:
        print(
            f"[WARNING] Returned {len(vals)} values, expected {n*columns}; "
            "truncating"
        )
        n = len(vals) // columns
        vals = vals[:n * columns]

    arr       = np.array(vals).reshape(n, columns)
    voltages  = arr[:, 0]
    currents  = arr[:, 1]
    timestamps = arr[:, 2]
    source_status = arr[:, 3] if columns == 4 else np.full(n, np.nan)

    print(f"[Buffer] Read complete: {n} points")

    # Query instrument errors
    err = inst.query(":SYST:ERR?").strip()
    if not err.startswith("0"):
        print(f"[INSTRUMENT WARNING] {err}")

    return {
        "voltages":   voltages,
        "currents":   currents,
        "timestamps": timestamps,
        "source_status": source_status,
        "n_points":   n,
        "buf_name":   buf_name,
    }


def compute_resistance(voltages: np.ndarray,
                       currents: np.ndarray,
                       r_series: float = 0.0) -> np.ndarray:
    safe = np.abs(currents) > 1e-12
    # Subtract the series resistor to obtain the device resistance (unchanged when r_series=0)
    resistance = np.where(safe,
                          np.abs(voltages / currents) - r_series,
                          np.nan)
    return resistance


def save_csv(data: dict, resistance: np.ndarray, path: str,
             r_series: float = 0.0):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "time_s", "voltage_V",
                    "current_A", "current_uA", "resistance_Ohm",
                    "voltage_set_V", "voltage_actual_V", "icc_A",
                    "compliance", "source_status", "stopped_reason",
                    "device_voltage_V", "r_series_Ohm"])
        for k in range(data["n_points"]):
            r = resistance[k] if np.isfinite(resistance[k]) else ""
            w.writerow([
                k,
                f"{data['timestamps'][k]:.6f}",
                f"{data['voltages'][k]:.6f}",
                f"{data['currents'][k]:.6e}",
                f"{data['currents'][k]*1e6:.4f}",
                f"{r:.4f}" if r != "" else "inf",
                f"{data['voltages'][k]:.6f}",
                f"{data['voltages'][k]:.6f}",
                "",
                "",
                ("" if np.isnan(data["source_status"][k])
                 else int(data["source_status"][k])),
                "",
                f"{device_voltage(data['voltages'][k], data['currents'][k], r_series):.6f}",
                f"{r_series:g}",
            ])
    print(f"[CSV] Saved: {path}")


def plot_and_save(data: dict, resistance: np.ndarray, img_path: str,
                  r_series: float = 0.0):
    v   = data["voltages"]
    i   = data["currents"] * 1e6    # uA
    t   = data["timestamps"]
    n   = data["n_points"]
    idx = np.arange(n)

    # ── Colour gradient (by time/index) ───────────────────────
    norm   = plt.Normalize(0, n - 1)
    cmap   = plt.get_cmap("plasma")
    colors = cmap(norm(idx))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#000000")
    for ax in axes:
        ax.set_facecolor("#000000")
        ax.tick_params(which="both", colors="#FFFFFF", labelsize=11)
        for sp in ax.spines.values():
            sp.set_color("#FFFFFF")

    ax1, ax2 = axes

    # ══ Left panel: I-V curve (colour gradient, direction arrows) ══
    pts  = np.array([v, i]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, cmap="plasma", norm=norm,
                          linewidth=1.5, zorder=2, alpha=0.9)
    lc.set_array(np.arange(len(segs)))
    ax1.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax1, pad=0.02)
    cbar.set_label("Sample Index", color="#FFFFFF", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="#FFFFFF", labelsize=10)
    cbar.outline.set_edgecolor("#FFFFFF")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#FFFFFF")

    # Arrows (8, evenly spaced)
    arrow_positions = np.linspace(0, n - 2, 8, dtype=int)
    for ap in arrow_positions:
        if ap + 1 >= n:
            continue
        dv = v[ap+1] - v[ap]
        di = i[ap+1] - i[ap]
        if abs(dv) + abs(di) < 1e-9:
            continue
        c = cmap(norm(ap))
        ax1.annotate("",
            xy=(v[ap+1], i[ap+1]),
            xytext=(v[ap], i[ap]),
            arrowprops=dict(arrowstyle="-|>", color=c,
                            lw=0.8, mutation_scale=9))

    # Start / end markers
    ax1.scatter([v[0]],  [i[0]],  color="#69F0AE", s=60, zorder=5,
                label=f"Start  ({v[0]:.3f} V, {i[0]:.2f} uA)", edgecolors="#fff", lw=0.5)
    ax1.scatter([v[-1]], [i[-1]], color="#FF4081", s=60, zorder=5,
                label=f"End    ({v[-1]:.3f} V, {i[-1]:.2f} uA)", edgecolors="#fff", lw=0.5)

    ax1.axhline(0, color="#888888", lw=0.5)
    ax1.axvline(0, color="#888888", lw=0.5)

    # Automatic axis limits
    v_pad = max((v.max() - v.min()) * 0.1, 0.1)
    i_pad = max((i.max() - i.min()) * 0.1, 5.0)
    ax1.set_xlim(v.min() - v_pad, v.max() + v_pad)
    ax1.set_ylim(i.min() - i_pad, i.max() + i_pad)

    ax1.set_xlabel("Voltage (V)",    color="#FFFFFF", fontsize=12)
    ax1.set_ylabel("Current (uA)",   color="#FFFFFF", fontsize=12)
    ax1.set_title("I–V Curve  (color = time progression)",
                  color="#FFFFFF", fontsize=11, pad=8)
    ax1.legend(facecolor="#000000", edgecolor="#888888",
               labelcolor="#FFFFFF", fontsize=8, loc="best")
    ax1.grid(True, color="#3A3A3A", lw=0.5, zorder=0)

    # ══ Right panel: resistance vs index (log scale) ══════════════
    r_kohm = resistance / 1e3
    finite = np.isfinite(r_kohm)

    sc = ax2.scatter(idx[finite], r_kohm[finite],
                     c=idx[finite], cmap="plasma", norm=norm,
                     s=4, alpha=0.75, zorder=3)

    # Rolling-median smoothing line (window = 5% of the points, at least 5 points)
    win = max(5, int(n * 0.05))
    if np.sum(finite) >= win:
        from numpy.lib.stride_tricks import sliding_window_view
        r_valid = r_kohm.copy()
        r_valid[~finite] = np.nan
        # Rolling nanmedian
        half = win // 2
        smoothed = np.full(n, np.nan)
        for k in range(n):
            lo = max(0, k - half)
            hi = min(n, k + half + 1)
            chunk = r_kohm[lo:hi]
            valid_chunk = chunk[np.isfinite(chunk)]
            if len(valid_chunk) > 0:
                smoothed[k] = np.nanmedian(valid_chunk)
        ax2.plot(idx, smoothed, color="#FFD54F", lw=1.5,
                 alpha=0.85, label=f"Rolling median (window={win})", zorder=4)

    # Ron / Roff reference lines (5th / 95th percentiles of the finite values)
    if finite.sum() >= 5:
        r_fin = r_kohm[finite]
        r_low = np.percentile(r_fin, 5)
        r_hig = np.percentile(r_fin, 95)
        ax2.axhline(r_low, color="#4DB6AC", lw=1.0,
                    linestyle=":", alpha=0.9,
                    label=f"P5 ≈ {r_low:.2f} kΩ  (LRS reference)")
        ax2.axhline(r_hig, color="#FFB74D", lw=1.0,
                    linestyle=":", alpha=0.9,
                    label=f"P95 ≈ {r_hig:.2f} kΩ  (HRS reference)")
        if r_low > 0:
            ratio = r_hig / r_low
            ax2.text(0.97, 0.05,
                     f"P95/P5 ≈ {ratio:.1f}×",
                     transform=ax2.transAxes,
                     color="#CE93D8", fontsize=9.5, ha="right",
                     bbox=dict(boxstyle="round,pad=0.3",
                               fc="#000000", ec="#9C27B0", alpha=0.85))

    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(ticker.LogFormatter(minor_thresholds=(2, 0.5)))
    ax2.yaxis.set_minor_formatter(ticker.LogFormatter(minor_thresholds=(2, 0.5)))
    ax2.set_xlabel("Sample Index",     color="#FFFFFF", fontsize=12)
    ax2.set_ylabel("Resistance (kΩ)",  color="#FFFFFF", fontsize=12)
    ax2.set_title("Resistance vs. Sample Index",
                  color="#FFFFFF", fontsize=11, pad=8)
    ax2.legend(facecolor="#000000", edgecolor="#888888",
               labelcolor="#FFFFFF", fontsize=8)
    ax2.grid(True, color="#3A3A3A", lw=0.5, which="both", zorder=0)

    # ── Overall title (metadata summary) ──────────────────────
    ts_str  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dur_str = f"{t[-1]:.2f} s" if len(t) > 0 else "—"
    title = (f"Buffer: {data['buf_name']}   |   {n} pts   |   "
             f"Duration: {dur_str}   |   {ts_str}")
    if r_series > 0:
        title += f"   |   Rs={r_series / 1e3:g} kOhm"
    fig.suptitle(title, color="#FFFFFF", fontsize=9, y=1.01)

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[PNG] Saved: {img_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Keithley 2450/2461 buffer readout")
    parser.add_argument("--address", default=DEFAULT_VISA_ADDRESS)
    parser.add_argument("--buffer",  default=DEFAULT_BUFFER,
                        help="buffer name, default defbuffer1")
    parser.add_argument("--output",  default=OUTPUT_PREFIX,
                        help="output file prefix (no extension), default buffer_data")
    add_series_argument(parser)
    args = parser.parse_args()
    validate_series(args.r_series)
    if args.r_series > 0:
        print(f"  Series R    : {args.r_series:g} Ω (resistance is the device alone)")

    # Timestamp suffix avoids overwriting; with a series resistor an RS tag precedes it
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag      = series_tag(args.r_series)
    stem     = f"{args.output}_{tag}" if tag else args.output
    csv_path = f"{stem}_{ts}.csv"
    img_path = f"{stem}_{ts}.png"
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    inst = connect(args.address)
    data = read_buffer(inst, args.buffer)
    inst.close()

    resistance = compute_resistance(data["voltages"], data["currents"], args.r_series)

    # Print a summary in the terminal
    print(f"\n  Points      : {data['n_points']}")
    print(f"  V range     : {data['voltages'].min():.4f} ~ {data['voltages'].max():.4f} V")
    fin = resistance[np.isfinite(resistance)]
    if len(fin):
        print(f"  R range     : {fin.min()/1e3:.3f} ~ {fin.max()/1e3:.3f} kΩ")
        print(f"  Peak current: {np.abs(data['currents']).max()*1e6:.3f} uA")

    save_csv(data, resistance, csv_path, args.r_series)
    plot_and_save(data, resistance, img_path, args.r_series)


if __name__ == "__main__":
    main()