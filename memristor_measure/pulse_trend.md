# Control the resistance trend with one pulse at a time

`pulse_trend.py` supports positive SET (`--direction down`) and negative RESET
(`--direction up`). It checks the resistance at +0.1 V after every pulse and
continues only after the readback settles. It never reverses polarity or
automatically increases voltage, pulse width, or current compliance.

This controls the decision to apply another pulse. It cannot guarantee a smooth
or monotonic physical response, and a single pulse can overshoot. The thresholds
below are initial controller settings, not calibrated properties of this device.

## What the existing measurements establish

Source: `Pulse_Modulation/pulse_modulation_*.csv` inside the dated run folders,
with their matching JSON metadata, 17 September 2026. There are 13 runs: five
baseline-only runs and eight runs containing one negative pulse each. None
contains a positive SET pulse. Compare resistance within each run; the common
label `DEV` does not establish that separate runs used the same device or wiring.

- 15:05:13: commanded -0.8 V, actual -0.552 V, 50 uA compliance;
  11.898 to 11.872 kOhm. Full write voltage was not reached.
- 15:09:01: -0.9 V, Icc 95 uA; 24.727 to 26.733 kOhm (+8.1%).
- 15:10:00: -0.9 V, Icc 95 uA; 25.185 to 25.194 kOhm (+0.034%).
- 15:11:37: -1.0 V, Icc 95 uA; 25.191 to 27.624 kOhm (+9.7%).
- 15:12:31: -1.0 V, Icc 95 uA; 1.265 to 0.794 MOhm (-37.3%).
- 16:55:46: -1.0 V, Icc 95 uA; 5.761 to 6.409 MOhm (+11.2%).

The data show state-dependent responses, including a change opposite the desired
RESET trend. Cross-run changes could involve relaxation or intervening actions;
they are not a continuous retention measurement. The later baseline-only records
are about 6.23 and 6.29 MOhm, but a fresh read is necessary before another write.
Reliable gradual SET/RESET needs repeated, within-run characterization.

## Run from the repository root

First take a fresh baseline. This command only applies the +0.1 V read bias:

```powershell
python memristor_measure/pulse_trend.py --direction read
```

Once the wiring is checked (FRONT FORCE HI/LO), start with one write pulse and
review the result. Run the relevant command for the direction you want:

```powershell
python memristor_measure/pulse_trend.py --direction down --v-set 1.0 --icc 50e-6 --max-pulses 1
python memristor_measure/pulse_trend.py --direction up --v-reset -1.0 --icc 50e-6 --max-pulses 1
```

Every run prompts before connecting. The 50 uA starting limit is conservative;
the -1 V measurements cited above used 95 uA, so their response is not established
at 50 uA. A compliance stop is a result to inspect, not a reason to raise Icc
automatically. RESET also stops if already above the default 1 MOhm ceiling.

After validating small, repeatable changes, use a short sequence, optionally
stopping at a target crossing (target in Ohm):

```powershell
python memristor_measure/pulse_trend.py --direction down --max-pulses 3 --target 100e3
python memristor_measure/pulse_trend.py --direction up --max-pulses 3 --target 300e3
```

These targets are examples, not measured stable levels. A target is a one-sided
stopping boundary, not a precision tolerance band or a promise of retention.

## Verification and stop conditions

- Ten samples per read burst, using measured voltage/current. Any invalid,
  missing, reversed-current, or compliance read stops the run.
- One second with output off before and between bursts. Three consecutive bursts
  must have no more than 2% spread both within each burst and between burst means.
  The accepted state is the median of the three burst resistance estimates.
- Verification times out after 15 seconds; an in-flight VISA operation may take
  longer to return. Settling in this short window does not prove long-term retention.
- A change below 5% is unresolved; stop after two consecutive unresolved changes.
  Stop immediately after a change greater than 20%, or an opposite change greater
  than 5%. Completed valid read bursts are saved even when their trend or
  stability is rejected. A failed sample invalidates its entire burst. The
  rejected sample and prior samples are retained for diagnostics, flagged with
  `sample_valid` and `burst_valid`; they are never used to authorize a write.
- The effective floor is `max(r_min, 1.2 * max(abs(V_SET), abs(V_RESET)) / Icc)`:
  24 kOhm with +/-1 V and 50 uA. This is a conservative screening estimate based
  on the low-bias resistance, not a nonlinear device model. Actual write voltage
  and current are checked independently.
- Before another SET, require `R_read > floor / ((1-max_step)*(1-stable_tolerance))`.
  Default settings require more than 30.61 kOhm. This reserves room for an assumed
  20% next-pulse decrease and 2% read spread. Neither is a proven physical bound;
  a larger single-pulse jump can still cross the floor. Both SET and RESET are
  blocked at/below the floor, with no automatic recovery pulse or Icc increase.
- Stop on pulse compliance, overcurrent, incomplete pulse, or more than 10%
  difference between commanded and measured pulse voltage. The controller does
  not issue another pulse or a recovery RESET after these stops.
- Any measured current at or above Icc stops this controller; there is no 2%
  overcurrent allowance. The shared driver rejects a programmed current-limit
  readback above the requested value. This check verifies the setting, not the
  transient current waveform. Existing open-loop scripts do not have this
  controller's resistance reserve guard; use `pulse_trend.py` for this workflow.
- Output is returned to zero and switched off between operations and on exit.

Settings are exposed through `--help`. Increase observation time when studying
relaxation; do not loosen the stability requirement simply to force more pulses.

`--r-series <Ohm>` (default 0): resistor in series in the HI line; resistances, floor (Icc term minus Rs), target and logs describe the device alone, and the run folder gets an `RS..k` tag.

## Pulse timing matters

The existing software pulse implementation performs a blocking read before
returning to zero. Its requested 20 ms is not a verified physical pulse width;
communication and measurement time can extend it. `host_call_elapsed_s` in the
new log includes overhead and shutdown, and is not a pulse-width measurement.

For a Keithley 2461, `--pulse-mode hardware` uses the project's existing native
pulse implementation. Pulse widths below 20 ms require that mode; the project
accepts down to 150 us, but the practical width also depends on measurement and
instrument settings. Confirm the waveform and instrument errors before treating
short pulses as calibrated. A 2450 supports only software mode in this project.
The shared software pulse routine now skips any remaining hold after a
compliance, at-limit, or invalid sample and always attempts output OFF. It still
must wait for the blocking instrument read before it can react.

### A resistance floor does not guarantee zero current overshoot

For an ohmic estimate, 1 V / 50 uA = 20 kOhm. The margins above are safeguards
for deciding whether to issue another pulse, not proof that instantaneous
current remains below Icc. The 0.1 V resistance is not the resistance at -1 V;
in the 15:11:37 run, a 25.19 kOhm initial read was followed by about 66.5 uA
at -1 V (Icc was 95 uA). This demonstrates why the simple estimate can understate
write current.

Normally the SMU reduces output voltage on reaching compliance, so an
unreachable RESET voltage should lead to a stop rather than increasing Icc.
Fast current overshoot during switching remains possible. Tektronix discusses
the response-time limits of DC current compliance and placing current limiting
close to the device in its
[pulse memory testing application note](https://download.tek.com/document/1KW-60638-1_Non-Volatile_Memory%20Test_4200A-SCS_Application_Note_111320.pdf).
A requirement to stay below a damage-current ceiling needs hardware limiting
with margin, control of interconnect capacitance, and transient waveform
validation on the actual setup. The software does not certify that requirement.

## Output and offline validation

`Pulse_Trend/<device>_<direction>_<timestamp>/` contains `events.csv`,
`events.jsonl`, and `metadata.json`. The JSONL log is flushed after every event.
Records include raw reads, read bursts, accepted settled states, actual pulse
voltage/current, fractional resistance changes, and the stop reason. These files
are saved on measurement errors and Ctrl+C as well as on normal completion.

```powershell
python -m unittest test_keithley_smu test_pulse_set_reset test_pulse_trend
```

Tests use fake instruments. They do not establish device performance or verify
hardware pulse timing.

## If a baseline read fails

An invalid baseline stops the run before any SET/RESET pulse. The error now
prints sample number, measured voltage/current, requested Icc, compliance flag,
source status, and the exact rejected condition. `events.csv` and `events.jsonl`
also retain the rejected sample and any preceding samples. A communication or
parsing failure retains earlier samples and its error text, without inventing
the missing reading. Rejected samples have no accepted resistance estimate.

- `compliance` or `current_at_or_above_limit`: the read reached the current
  guard; do not raise Icc to bypass it.
- `read_voltage_out_of_tolerance`: measured voltage is outside 0.09 to 0.11 V
  for the +0.1 V read. Inspect the printed current and compliance alongside it.
- `nonpositive_or_unresolved_current`: current is negative, zero, or at/below
  1e-14 A; this does not establish overcurrent or device damage.
- `nonfinite_current`, `nonfinite_voltage`, or `read_failed`: no usable complete
  measurement was obtained. Review the saved diagnostic before programming.

For the next diagnostic run, repeat `--direction read`. The older generic
`Invalid resistance read: compliance, current, or voltage` message did not save
the failed sample, so its exact cause cannot be recovered from that run's files.

### Signed-current diagnostic for very small signals

When current is near zero and changes sign, collect a diagnostic without trying
to classify a resistance state. From inside `memristor_measure`:

```powershell
python pulse_trend.py --direction diagnose --icc 50e-6 --nplc 1 --read-settle 0.5 --n-read-avg 30
```

This records three bursts at 0 V, +0.1 V, and 0 V, with output OFF between bursts
and on exit. Each burst prints its signed mean, sample standard deviation and
range, in pA. All raw samples are saved in the usual events files. The final
comparison subtracts the mean of the two zero-bias averages from the +0.1 V
average and reports the change between the zero-bias averages. Zero-bias current
can contain device relaxation as well as instrumental offset; this comparison
is a diagnostic, not a calibrated correction or proof of retention/recovery.

This mode allows small negative/zero currents so they can be examined rather
than discarded. It still rejects compliance, current magnitude at/above Icc,
nonfinite readings and voltage mismatches. It issues no SET/RESET pulses, never
calculates a zero-bias resistance, and cannot authorize a write. Ordinary `read`,
`down` and `up` retain their strict resistance-validation rules.

Write-verify programming is also used in published memristor work:
[4K-memristor analog-grade passive crossbar circuit](https://www.nature.com/articles/s41467-021-25455-0).
That supports the feedback approach, not transfer of another device's pulse
parameters to this sample.
