# Ni/AlOx/TiOx/Ti device — operating notes

Measured on 2026-09-24 with a Keithley 2450 (software pulse mode, minimum pulse width 20 ms). Everything here comes from one device tested over one afternoon, with 3–5 cycles per condition. Re-check these values before relying on them for another device or on a later day.

## Polarity

This device switches with the **opposite polarity to the script defaults**:

| Operation | Polarity | Behaviour |
|---|---|---|
| SET (resistance falls) | **Negative** | Abrupt. Happens on one pulse; the final level is set by the current limit (compliance). |
| RESET (resistance rises) | **Positive** | Gradual at 0.75–0.86 V. Nearly complete in one pulse at 0.9 V. |

Always read at **+0.1 V**. That is well below the RESET threshold, and no read disturbance was seen.

## Working parameters

| Purpose | Voltage | Width | Current limit | Typical result (read at 0.1 V) |
|---|---|---|---|---|
| Full SET | −1.0 V | 0.5 s | 300 µA | 6.5–9.4 kΩ (mean 7.9 kΩ) |
| Higher-R SET | −1.0 V | 0.5 s | 200 µA | 15.8–18.9 kΩ (mean 17.7 kΩ) |
| Full RESET | +0.9 V | 0.1 s | 800 µA | 35–41 kΩ; up to about 62 kΩ with more pulses |
| Fine RESET step | +0.75 to +0.89 V | 20 ms | 800 µA | +2–7% per pulse |

What did **not** work:
- **SET limit of 100 µA:** no SET at all.
- **SET pulses of 0.1 s:** only partial SET, even at −1.0 V.
- **RESET at +0.8 V or below from the low state:** no RESET.
- **RESET with a 300 µA limit:** the current clamps, and the device only sees about 0.77 V.

## Scripts (run from `memristor_measure/`, PowerShell)

Tune to a target resistance (write-verify, then a 60 s hold):

```powershell
python resistance_tune.py --dev NiAlOxTiOx --target 12e3 25e3 --tol 0.05 --hold 60
```

Measured results: 12, 15, 25 and 28 kΩ were each reached within ±5% in 4–28 pulses, and all four stayed inside the band for the 60 s hold. Useful range is about 10–30 kΩ.

Switch fully, then check retention:

```powershell
python resistance_switch_retention.py --dev NiAlOxTiOx --v-set -1.00 --t-set 0.5 --v-reset 0.90 --t-reset 0.1 --retention-time 300 --interval 2 --icc 3e-4 --icc-set 3e-4 --icc-reset 8e-4 --v-read 0.1 --n-read-avg 5 --nplc 1 --cycles 3
```

Pulse trains for calibration:

```powershell
python pulse_resistance_modulation.py --dev NiAlOxTiOx --v-pot -1.00 --v-dep 0.90 --t-pot 0.5 --t-dep 0.1 --n-pot 1 --n-dep 1 --cycles 5 --icc 3e-4 --icc-pot 3e-4 --icc-dep 8e-4 --v-read 0.1 --n-read-avg 10 --nplc 1
```

In `pulse_resistance_modulation.py`, "potentiation" means SET and must use a **negative** `--v-pot`. "Depression" means RESET and must use a **positive** `--v-dep`. Do not use `pulse_trend.py` on this device: it assumes positive SET, and it stops whenever a SET pulse reaches its current limit.

## Retention (at room temperature, read at 0.1 V)

| State | Measured | Change |
|---|---|---|
| Low state, 2.5–5.8 kΩ | 5 min × 3 cycles | Under +1.5% |
| High state, 24–44 kΩ | 5 min × 3 cycles | Rises by 2–11% |
| Full high state, about 67–75 kΩ | 5 min | Never falls; jumps between two levels about 10% apart |
| Tuned states, 12–28 kΩ | 60 s each | Rises by 0.3–4.2%; higher R drifts more |

Resistance only ever drifted **upward**. None of this shows retention beyond 5 minutes.

## Cautions

- **Avoid partial SET states.** Once, a partly switched 17.5 kΩ state fell to 7.0 kΩ over 2 minutes with no pulses applied. Aim for full SET or tuned states.
- **Keep SET and RESET balanced.** Several RESET pulses per cycle combined with a weak SET (−0.85 V, 0.1 s) pushed the device steadily into the high state, until SET stopped working.
- **Positive pulses occasionally lower R.** This happened at +0.7 V early on, and once at +0.88 V during tuning (26.2 → 17.1 kΩ). The tuner recovers from this automatically.
- **The RESET voltage needed rises as R rises.** About 0.75 V moves the device at 12–15 kΩ, about 0.80 V at 19 kΩ, and 0.83–0.86 V at 22–26 kΩ.
- **`pulse_current_A` in the CSVs is sampled at the start of the pulse.** It doesn't show a switch that happens later in the pulse, so judge switching from the read after the pulse.
- **In IV-sweep CSVs, part of `resistance_Ohm` isn't real data.** While the sweep is at its current limit, that column repeats the last value before the limit was hit.

## Data

Every run folder is named `NiAlOxTiOx_*_20260924_*`:
- `Pulse_Modulation/`: calibration runs
- `Switch_Retention/`: `…_162509` (3-cycle retention)
- `Fixed_V_Read/`: `…_161541` (5-minute high-state hold)
- `Resistance_Tune/`: `…_172005` and `…_172542` (write-verify tuning)
