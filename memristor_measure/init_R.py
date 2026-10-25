"""Batch measurement of the initial (pre-forming) resistance distribution of memristors.

Wiring: Keithley FORCE HI to the device top electrode and FORCE LO to the bottom
electrode; always switch the output off before changing devices. Parameter/command
example: ``python init_R.py --n_devices 20 --output initial.csv``; ``--address`` sets
the VISA address. Output: a summary in the terminal and an optional CSV. The CSV keeps
the original column order dev_id/R_mean/R_std/I_mean/V_applied/status and appends the
set/measured voltage, Icc, compliance, source status and stop reason. Safety warning:
the read voltage is limited to ±2 V and Icc is hard-limited to 50 uA; on compliance the
resistance is recorded as nan, the remaining measurements are stopped and the output
returns safely to zero. Never rewire while the output is on.
Series resistor: with ``--r-series <Ohm>``, R is the device alone (reading minus Rs)
and the CSV appends device_voltage_V/r_series_Ohm.
"""

import argparse
import time
import sys
import numpy as np

try:
    from .keithley_smu import (
        AUTO_ADDRESS,
        InstrumentConnectionError,
        add_series_argument,
        connect_smu,
        configure_voltage_source,
        device_voltage,
        error_is_clear,
        read_vi,
        safe_shutdown,
        validate_safety_limits,
        validate_series,
    )
except ImportError:
    from keithley_smu import (
        AUTO_ADDRESS,
        InstrumentConnectionError,
        add_series_argument,
        connect_smu,
        configure_voltage_source,
        device_voltage,
        error_is_clear,
        read_vi,
        safe_shutdown,
        validate_safety_limits,
        validate_series,
    )

# ─────────────────────────────────────────────
# Adjustable parameters (edit these to suit the measurement)
# ─────────────────────────────────────────────
DEFAULT_VISA_ADDRESS = AUTO_ADDRESS
V_READ        = 0.01       # read voltage (V): small signal, well below the forming voltage
I_CC          = 50e-6      # compliance current (A): protective limit against accidental forming
N_SAMPLES     = 10        # samples per device, averaged
SETTLE_TIME   = 0.2       # settling time after V_READ is applied (s)
INTER_DEV_PAUSE = 1.0     # pause between devices (s), for manual reconnection

# Screening criteria
R_SHORT_THRESH = 1e3      # below this value the device is classed as shorted (Ω)
R_OPEN_THRESH  = 1e8      # above this value the device is classed as open (Ω)

NPLC = 1.0                # integration time (number of power line cycles), 1.0 = 16.7 ms @60Hz
# ─────────────────────────────────────────────


def connect(address: str):
    """Open the VISA connection and return the instrument resource object."""
    try:
        return connect_smu(address, timeout_ms=10000)
    except InstrumentConnectionError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


def init_smu(inst, v_read: float, i_cc: float, nplc: float):
    """Configure the 2450/2461 as a small-signal voltage source with current measurement."""
    actual_icc = configure_voltage_source(inst, i_cc, nplc)
    inst.write(f":SOUR:VOLT:LEV {v_read:.4f}")
    time.sleep(0.5)  # wait for the reset to finish
    err = inst.query(":SYST:ERR?").strip()
    if not error_is_clear(err):
        raise RuntimeError(f"Instrument initialisation failed: {err}")
    return actual_icc


def measure_resistance(inst, n_samples: int, settle_time: float,
                       v_read: float, i_cc: float, r_series: float = 0.0) -> dict:
    """
    Per device: set V_READ, output on, take N samples, then return to zero and output off.

    Note: after each device, safe_shutdown() sets the source voltage to 0 V, so the read
    voltage cannot be set only once in init_smu().
    Returns a dict: {R_mean, R_std, I_mean, V_applied, raw_I}
    """
    validate_safety_limits(v_read, i_cc)
    currents = []
    actual_voltages = []
    source_statuses = []
    compliance = False

    try:
        # safe_shutdown() returns to zero after each device, so restore the read voltage here.
        inst.write(f":SOUR:VOLT:LEV {v_read:.9g}")
        err = inst.query(":SYST:ERR?").strip()
        if not error_is_clear(err):
            raise RuntimeError(f"Failed to set the read voltage: {err}")

        inst.write(":OUTP ON")
        time.sleep(settle_time)

        for _ in range(n_samples):
            reading = read_vi(inst, v_read, i_cc)
            currents.append(reading.current_a)
            actual_voltages.append(reading.voltage_actual_v)
            source_statuses.append(reading.source_status)
            if reading.compliance:
                compliance = True
                print("  [SAFETY STOP] Compliance detected; stopping and returning to zero.")
                break
            time.sleep(0.05)
    finally:
        safe_shutdown(inst)

    if not currents:
        raise RuntimeError("The instrument returned no valid V/I reading")

    currents = np.array(currents)
    i_mean = np.nanmean(currents)

    # Resistance R = V / I (guarded against zero current)
    v_applied = v_read
    if compliance:
        r_mean = float("nan")
        r_std = float("nan")
    elif abs(i_mean) < 1e-14:
        r_mean = float("inf")
        r_std = float("nan")
    else:
        # Subtract the series resistor to obtain the device resistance (unchanged when r_series=0)
        r_mean = abs(float(np.nanmean(actual_voltages)) / i_mean) - r_series
        r_vals = np.where(
            np.abs(currents) < 1e-14, np.inf,
            np.abs(np.asarray(actual_voltages) / currents) - r_series,
        )
        r_std = np.nanstd(r_vals[np.isfinite(r_vals)])

    return {
        "R_mean": r_mean,
        "R_std":  r_std,
        "I_mean": i_mean,
        "V_applied": v_applied,
        "raw_I": currents.tolist(),
        "voltage_set_V": v_read,
        "voltage_actual_V": float(np.nanmean(actual_voltages)),
        "icc_A": i_cc,
        "compliance": compliance,
        "source_status": next((s for s in reversed(source_statuses)
                               if s is not None), None),
        "stopped_reason": "compliance" if compliance else "",
        "device_voltage_V": device_voltage(
            float(np.nanmean(actual_voltages)), float(i_mean), r_series),
        "r_series_Ohm": r_series,
    }


def classify(r: float) -> str:
    """Classify the device state."""
    if not np.isfinite(r):
        return "INVALID ⚠"
    if r < R_SHORT_THRESH:
        return "SHORT ⚠"
    if r > R_OPEN_THRESH:
        return "OPEN  ⚠"
    return "OK    ✓"


def format_resistance(r: float) -> str:
    """Format a resistance with an adaptive unit."""
    if not np.isfinite(r):
        return "∞ Ω (OPEN)"
    if r >= 1e9:
        return f"{r/1e9:.3f} GΩ"
    if r >= 1e6:
        return f"{r/1e6:.3f} MΩ"
    if r >= 1e3:
        return f"{r/1e3:.3f} kΩ"
    return f"{r:.2f} Ω"


def print_header():
    print("=" * 65)
    print(f"  {'Device':<8} {'R_mean':>14} {'R_std':>12} {'I_mean':>14} {'Status':<10}")
    print("-" * 65)


def print_result(dev_id: int, res: dict, status: str):
    r_str = format_resistance(res["R_mean"])
    r_std_str = format_resistance(res["R_std"])
    i_str = f"{res['I_mean']*1e6:.3f} uA"
    print(f"  DEV-{dev_id:<4} {r_str:>14} {r_std_str:>12} {i_str:>14}   {status}")


def print_summary(results: list):
    print("\n" + "=" * 65)
    print("  Measurement complete: summary")
    print("=" * 65)

    r_all = [r["R_mean"] for r in results if np.isfinite(r["R_mean"])]
    if not r_all:
        print("  No valid data.")
        return

    r_arr = np.array(r_all)
    n_ok     = sum(1 for r in results if classify(r["R_mean"]) == "OK    ✓")
    n_short  = sum(1 for r in results if classify(r["R_mean"]) == "SHORT ⚠")
    n_open   = sum(1 for r in results if classify(r["R_mean"]) == "OPEN  ⚠")

    print(f"  Total devices : {len(results)}")
    print(f"  Valid (OK)    : {n_ok}")
    print(f"  Short (SHORT) : {n_short}  (R < {format_resistance(R_SHORT_THRESH)})")
    print(f"  Open (OPEN)   : {n_open}  (R > {format_resistance(R_OPEN_THRESH)})")
    print()
    print(f"  R mean        : {format_resistance(np.mean(r_arr))}")
    print(f"  R median      : {format_resistance(np.median(r_arr))}")
    print(f"  R minimum     : {format_resistance(np.min(r_arr))}")
    print(f"  R maximum     : {format_resistance(np.max(r_arr))}")
    print(f"  R std. dev.   : {format_resistance(np.std(r_arr))}")

    # Distribution histogram (ASCII, no matplotlib needed)
    # After the series correction the device resistance can be <= 0; only positive values are binned
    r_pos = r_arr[r_arr > 0]
    if len(r_pos) >= 3:
        print()
        print("  Log resistance distribution (log10 R / Ω):")
        log_r = np.log10(r_pos)
        bins = np.linspace(log_r.min(), log_r.max(), 8)
        counts, edges = np.histogram(log_r, bins=bins)
        max_count = max(counts) if max(counts) > 0 else 1
        bar_width = 30
        for i in range(len(counts)):
            lo = 10 ** edges[i]
            hi = 10 ** edges[i+1]
            bar = "█" * int(counts[i] / max_count * bar_width)
            print(f"  {format_resistance(lo):>10} – {format_resistance(hi):<10} │{bar:<{bar_width}}│ {counts[i]}")

    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Keithley 2450/2461 initial resistance measurement")
    parser.add_argument("--address",   default=DEFAULT_VISA_ADDRESS, help="VISA resource address")
    parser.add_argument("--n_devices", type=int, default=None,       help="number of devices (omit to confirm each one)")
    parser.add_argument("--output",    default=None,                  help="output CSV file path (optional)")
    add_series_argument(parser)
    args = parser.parse_args()
    validate_series(args.r_series)

    validate_safety_limits(V_READ, I_CC)
    inst = connect(args.address)
    actual_icc = init_smu(inst, V_READ, I_CC, NPLC)

    print(f"  Measurement parameters:")
    print(f"    V_read    = {V_READ} V")
    print(f"    I_cc      = {I_CC*1e6:.1f} uA (protection limit)")
    print(f"    Samples   = {N_SAMPLES} per device")
    print(f"    Settling  = {SETTLE_TIME} s")
    if args.r_series > 0:
        print(f"    R_series  = {format_resistance(args.r_series)} (R is the device alone)")
    print()

    results = []
    dev_id  = 1

    print_header()

    try:
        while True:
            # Decide whether to continue with this device
            if args.n_devices is not None:
                if dev_id > args.n_devices:
                    break
                # Automatic mode: prompt the user to reconnect
                input(f"\n  → Connect DEV-{dev_id}, then press Enter to start the measurement...")
            else:
                # Manual mode: ask each time whether to continue
                ans = input(f"\n  → Connect DEV-{dev_id}; Enter to measure, q to quit: ").strip().lower()
                if ans == "q":
                    break

            res = measure_resistance(
                inst, N_SAMPLES, SETTLE_TIME, V_READ, actual_icc, args.r_series
            )
            status = classify(res["R_mean"])
            print_result(dev_id, res, status)
            results.append({"dev_id": dev_id, **res, "status": status})
            if res["compliance"]:
                break

            time.sleep(INTER_DEV_PAUSE)
            dev_id += 1

    except KeyboardInterrupt:
        print("\n\n  [INTERRUPTED] Stopped manually by the user.")

    finally:
        # Make sure the instrument output is off and the voltage is at zero
        try:
            safe_shutdown(inst)
        except Exception:
            pass
        inst.close()

    if results:
        print_summary(results)

        # Optional: save the CSV
        if args.output:
            import csv
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                old_fields = [
                    "dev_id", "R_mean", "R_std", "I_mean",
                    "V_applied", "status",
                ]
                extra_fields = [
                    "voltage_set_V", "voltage_actual_V", "icc_A",
                    "compliance", "source_status", "stopped_reason",
                    "device_voltage_V", "r_series_Ohm",
                ]
                writer = csv.DictWriter(
                    f, fieldnames=old_fields + extra_fields
                )
                writer.writeheader()
                for r in results:
                    row = {k: r[k] for k in old_fields + extra_fields}
                    writer.writerow(row)
            print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()