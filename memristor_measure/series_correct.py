"""Correct a saved IV sweep for a series resistor (no instrument needed).

Adds device_voltage_V = V_actual - I*R_series and device_resistance_Ohm =
device_voltage_V / I to the CSV, and plots I against the device voltage next
to the applied voltage. Writes <name>_series_corrected.csv/.png beside it.
Example: ``python series_correct.py IV_sweep/<run>/iv_sweep_<ts>.csv --r-series 10e3``
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def correct(path: str, r_series: float) -> str:
    d = pd.read_csv(path)
    v = d["voltage_actual_V"] if "voltage_actual_V" in d else d["voltage_V"]
    i = d["current_A"]
    d["device_voltage_V"] = v - i * r_series
    with np.errstate(divide="ignore", invalid="ignore"):
        d["device_resistance_Ohm"] = np.where(i.abs() > 1e-12, d["device_voltage_V"] / i, np.nan)
    base = os.path.splitext(path)[0] + "_series_corrected"
    d.to_csv(base + ".csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(v, i * 1e6, lw=0.8, color="#B0BEC5", label="vs applied V")
    ax.plot(d["device_voltage_V"], i * 1e6, lw=1.2, color="#1E88E5", label=f"vs device V (R_series {r_series/1e3:g} kOhm)")
    ax.axhline(0, color="#888888", lw=0.5)
    ax.axvline(0, color="#888888", lw=0.5)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (uA)")
    ax.set_title(os.path.basename(path))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(base + ".png", dpi=150)
    return base


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("csv", nargs="+", help="iv_sweep_*.csv or iv_cycle*_*.csv files")
    p.add_argument("--r-series", type=float, required=True, help="Series resistor, Ohm")
    args = p.parse_args()
    for path in args.csv:
        print(f"[Saved] {correct(path, args.r_series)}.csv / .png")


if __name__ == "__main__":
    main()
