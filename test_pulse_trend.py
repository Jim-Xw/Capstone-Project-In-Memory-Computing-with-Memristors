"""Instrument-free checks for write-verify decisions and shutdown paths."""

import json
from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import Mock, patch

from memristor_measure import keithley_smu as smu
from memristor_measure import pulse_trend as trend


def reading(resistance=100e3, voltage=0.1, compliance=False):
    return smu.ViReading(0.1, voltage, voltage / resistance, 50e-6, compliance)


def pulse(voltage=1.0, **overrides):
    values = dict(mode="software", voltage_set_v=voltage, voltage_actual_v=voltage,
                  current_a=10e-6, current_limit_a=50e-6, compliance=False, completed=True)
    values.update(overrides)
    return smu.PulseResult(**values)


class ReadVerificationTests(unittest.TestCase):
    def test_diagnostics_accept_signed_current_at_zero_and_read_bias(self):
        for voltage in (0.0, 0.1):
            samples = [smu.ViReading(voltage, voltage, i, 50e-6, False)
                       for i in (-1.6e-10, 0.0, 2.3e-10)]
            inst = Mock()
            with patch.object(smu, "read_vi", side_effect=samples):
                result = smu.read_diagnostic_current_samples(
                    inst, current_limit_a=50e-6, count=3, settle_s=0, voltage_v=voltage)
            self.assertEqual(result, tuple(samples))
            self.assertEqual(inst.write.call_args.args, (":OUTP OFF",))

    def test_diagnostics_retain_current_and_voltage_protection(self):
        for current, actual_v, compliance in ((50e-6, 0.1, False), (-50e-6, 0.1, False),
                                               (1e-6, 0.1, True), (float("nan"), 0.1, False),
                                               (1e-9, 0.05, False)):
            sample = smu.ViReading(0.1, actual_v, current, 50e-6, compliance)
            with self.subTest(sample=sample), patch.object(smu, "read_vi", return_value=sample):
                with self.assertRaises(smu.ResistanceReadError):
                    smu.read_diagnostic_current_samples(Mock(), current_limit_a=50e-6, count=1, settle_s=0)

    def test_diagnostic_voltage_cannot_be_a_programming_voltage(self):
        for voltage in (-0.1, 0.2, 1.0):
            inst = Mock()
            with self.assertRaises(ValueError):
                smu.read_diagnostic_current_samples(inst, current_limit_a=50e-6, voltage_v=voltage)
            inst.write.assert_not_called()

    def test_rejected_sample_reports_values_and_specific_reason(self):
        cases = [
            (reading(compliance=True), "compliance"),
            (reading(voltage=0.05), "read_voltage_out_of_tolerance"),
            (smu.ViReading(0.1, 0.1, -1e-9, 50e-6, False), "nonpositive_or_unresolved_current"),
            (smu.ViReading(0.1, 0.1, 0.0, 50e-6, False), "nonpositive_or_unresolved_current"),
            (smu.ViReading(0.1, 0.1, 50e-6, 50e-6, False), "current_at_or_above_limit"),
            (smu.ViReading(0.1, 0.1, float("nan"), 50e-6, False), "nonfinite_current"),
            (smu.ViReading(0.1, float("nan"), 1e-6, 50e-6, False), "nonfinite_voltage"),
        ]
        for sample, reason in cases:
            with self.subTest(reason=reason):
                inst = Mock()
                with patch.object(smu, "read_vi", side_effect=[reading(), sample]):
                    with self.assertRaises(smu.ResistanceReadError) as caught:
                        smu.read_resistance_samples(inst, current_limit_a=50e-6, count=2, settle_s=0)
                error = caught.exception
                self.assertIn(reason, error.issues)
                self.assertEqual(error.sample_index, 2)
                self.assertEqual(error.readings[-1], sample)
                self.assertIn("Vactual=", str(error))
                self.assertIn("Icc=5e-05 A", str(error))
                self.assertEqual(inst.write.call_args.args, (":OUTP OFF",))

    def test_transport_error_preserves_partial_read_without_inventing_sample(self):
        with patch.object(smu, "read_vi", side_effect=[reading(), OSError("timeout")]):
            with self.assertRaises(smu.ResistanceReadError) as caught:
                smu.read_resistance_samples(Mock(), current_limit_a=50e-6, count=3, settle_s=0)
        self.assertEqual(caught.exception.issues, ("read_failed",))
        self.assertEqual(caught.exception.sample_index, 2)
        self.assertEqual(len(caught.exception.readings), 1)
        self.assertIsInstance(caught.exception.__cause__, OSError)

    def test_uses_actual_voltage_and_turns_off(self):
        inst = Mock()
        with patch.object(smu, "read_vi", return_value=reading(voltage=0.099)):
            samples = smu.read_resistance_samples(inst, current_limit_a=50e-6, count=2, settle_s=0)
        self.assertEqual(len(samples), 2)
        self.assertAlmostEqual(samples[0].resistance_ohm, 100e3)
        self.assertEqual(inst.write.call_args.args, (":OUTP OFF",))

    def test_invalid_or_partial_read_fails_closed(self):
        bad = [reading(compliance=True), reading(voltage=0.05),
               smu.ViReading(0.1, 0.1, float("nan"), 50e-6, False),
               smu.ViReading(0.1, 0.1, -1e-6, 50e-6, False), OSError("timeout")]
        for value in bad:
            with self.subTest(value=value):
                inst = Mock()
                with patch.object(smu, "read_vi", side_effect=[reading(), value]):
                    with self.assertRaises((RuntimeError, OSError)):
                        smu.read_resistance_samples(inst, current_limit_a=50e-6, count=2, settle_s=0)
                self.assertEqual(inst.write.call_args.args, (":OUTP OFF",))


class TrendTests(unittest.TestCase):
    def test_diagnostic_mode_never_writes_or_accepts_resistance(self):
        args = trend.parse_args(["--direction", "diagnose", "--n-read-avg", "3"])
        samples = [smu.ViReading(v, v, i, 50e-6, False)
                   for v, values in ((0, [-2e-10, 0, 2e-10]),
                                     (0.1, [-1e-10, 2e-10, 5e-10]),
                                     (0, [0, 1e-10, 2e-10])) for i in values]
        events, inst = [], Mock()
        with patch.object(smu, "read_vi", side_effect=samples), patch.object(trend.time, "sleep"), \
                patch.object(smu, "apply_voltage_pulse") as write, \
                patch.object(trend, "stable_resistance") as stable:
            reason = trend.run_trend(inst, args, events.append, trend.validate_args(args))
        self.assertEqual(reason, "diagnostic_only")
        write.assert_not_called()
        stable.assert_not_called()
        self.assertEqual(len([e for e in events if e["event"] == "diagnostic_sample"]), 9)
        self.assertFalse(any("resistance_ohm" in e for e in events))
        self.assertAlmostEqual(events[-1]["bias_minus_zero_current_a"], 1.5e-10, delta=1e-20)
        self.assertFalse(events[-1]["write_authorized"])
        self.assertEqual(inst.write.call_args.args, (":OUTP OFF",))

    def test_diagnostic_compliance_stops_remaining_phases_and_saves_sample(self):
        args = trend.parse_args(["--direction", "diagnose"])
        events, inst = [], Mock()
        sample = smu.ViReading(0, 0, 50e-6, 50e-6, True)
        with patch.object(smu, "read_vi", return_value=sample) as read, \
                patch.object(trend.time, "sleep"), patch.object(smu, "apply_voltage_pulse") as write:
            with self.assertRaises(smu.ResistanceReadError):
                trend.run_trend(inst, args, events.append, trend.validate_args(args))
        self.assertEqual(read.call_count, 1)
        write.assert_not_called()
        self.assertEqual(events[-1]["event"], "read_error")
        self.assertEqual(events[0]["phase"], "zero_before")
        self.assertFalse(events[0]["sample_valid"])
        self.assertEqual(inst.write.call_args.args, (":OUTP OFF",))

    def test_failed_baseline_is_logged_and_never_used_for_a_write(self):
        inst, events = Mock(), []
        args = trend.parse_args(["--direction", "down", "--n-read-avg", "2"])
        bad_sample = smu.ViReading(0.1, 0.05, 50e-6, 50e-6, True, source_status=168)
        with patch.object(smu, "read_vi", side_effect=[reading(), bad_sample]), \
                patch.object(trend.time, "sleep"), \
                patch.object(smu, "apply_voltage_pulse") as write:
            with self.assertRaises(smu.ResistanceReadError):
                trend.run_trend(inst, args, events.append, trend.validate_args(args))
        write.assert_not_called()
        self.assertEqual([e["event"] for e in events], ["read_sample", "read_sample", "read_error"])
        self.assertTrue(events[0]["sample_valid"])
        self.assertFalse(events[0]["burst_valid"])
        self.assertFalse(events[1]["sample_valid"])
        self.assertIsNone(events[1]["resistance_ohm"])
        self.assertEqual(events[1]["voltage_actual_v"], 0.05)
        self.assertEqual(events[1]["current_a"], 50e-6)
        self.assertIn("compliance", events[2]["read_issues"])
        self.assertEqual(inst.write.call_args.args, (":OUTP OFF",))

    def execute(self, values, args=None, result=None):
        args = args or trend.parse_args(["--direction", "down"])
        inst, events = Mock(), []
        with patch.object(trend, "stable_resistance", side_effect=values), \
                patch.object(smu, "apply_voltage_pulse", return_value=result or pulse()) as apply:
            reason = trend.run_trend(inst, args, events.append, trend.validate_args(args))
        self.assertEqual(inst.write.call_args.args, (":OUTP OFF",))
        return reason, apply.call_args_list, events

    def test_default_is_read_only(self):
        reason, calls, _ = self.execute([100e3], trend.parse_args([]))
        self.assertEqual((reason, len(calls)), ("read_only", 0))

    def test_gradual_set_stops_at_target_without_reversing(self):
        args = trend.parse_args(["--direction", "down", "--target", "85000"])
        reason, calls, _ = self.execute([100e3, 90e3, 84e3], args)
        self.assertEqual((reason, len(calls)), ("target_crossed", 2))
        self.assertTrue(all(c.kwargs["voltage_v"] == 1 for c in calls))

    def test_reset_uses_negative_polarity(self):
        args = trend.parse_args(["--direction", "up", "--target", "110000"])
        reason, calls, _ = self.execute([100e3, 112e3], args, pulse(-1))
        self.assertEqual(reason, "target_crossed")
        self.assertEqual(calls[0].kwargs["voltage_v"], -1)

    def test_stops_on_opposite_large_or_absent_change(self):
        for values, expected, count in [([100e3, 110e3], "opposite_trend", 1),
                                       ([100e3, 50e3], "step_too_large", 1),
                                       ([100e3, 99e3, 98e3], "no_clear_change", 2)]:
            with self.subTest(expected=expected):
                reason, calls, _ = self.execute(values)
                self.assertEqual((reason, len(calls)), (expected, count))

    def test_low_resistance_blocks_both_polarities(self):
        for direction in ("down", "up"):
            reason, calls, _ = self.execute([12e3], trend.parse_args(["--direction", direction]))
            self.assertEqual((reason, len(calls)), ("resistance_floor", 0))

    def test_set_stops_before_floor_and_does_not_attempt_reset(self):
        reason, calls, events = self.execute([30e3])
        self.assertEqual((reason, len(calls)), ("insufficient_set_reserve", 0))
        self.assertEqual(events[-1]["reason"], "insufficient_set_reserve")

    def test_set_reserve_covers_assumed_drop_and_read_spread(self):
        args = trend.parse_args(["--direction", "down"])
        floor = trend.validate_args(args)
        start = trend.set_start_floor(args, floor)
        self.assertAlmostEqual(floor, 24e3)
        self.assertAlmostEqual(start, 30612.24489795918)
        self.assertAlmostEqual(start * (1 - args.max_step) * (1 - args.stable_tolerance), floor)
        reason, calls, _ = self.execute([start], args)
        self.assertEqual((reason, len(calls)), ("insufficient_set_reserve", 0))

    def test_stronger_reset_voltage_raises_set_floor(self):
        args = trend.parse_args(["--direction", "down", "--v-reset", "-1.5"])
        self.assertAlmostEqual(trend.validate_args(args), 36e3)
        reason, calls, _ = self.execute([40e3], args)
        self.assertEqual((reason, len(calls)), ("insufficient_set_reserve", 0))

    def test_set_stops_before_next_pulse_when_reserve_used(self):
        reason, calls, _ = self.execute([35e3, 30e3])
        self.assertEqual((reason, len(calls)), ("insufficient_set_reserve", 1))

    def test_reset_not_blocked_by_set_only_reserve(self):
        args = trend.parse_args(["--direction", "up", "--max-pulses", "1"])
        reason, calls, _ = self.execute([30e3, 33e3], args, pulse(-1))
        self.assertEqual((reason, len(calls)), ("pulse_budget", 1))

    def test_pulse_budget_is_hard_limit(self):
        reason, calls, _ = self.execute([100e3, 90e3, 81e3, 73e3])
        self.assertEqual((reason, len(calls)), ("pulse_budget", 3))

    def test_bad_pulse_never_triggers_another_write(self):
        for result, expected in [(pulse(compliance=True), "pulse_compliance"),
                                 (pulse(current_a=60e-6), "overcurrent"),
                                 (pulse(current_a=50.1e-6), "overcurrent"),
                                 (pulse(current_a=-50.1e-6), "overcurrent"),
                                 (pulse(current_a=50e-6), "current_limit_reached"),
                                 (pulse(completed=False), "pulse_incomplete"),
                                 (pulse(voltage_actual_v=0.5), "pulse_voltage_mismatch"),
                                 (pulse(voltage_actual_v=float("nan")), "invalid_pulse_read")]:
            with self.subTest(expected=expected):
                reason, calls, _ = self.execute([100e3], result=result)
                self.assertEqual((reason, len(calls)), (expected, 1))

    def test_read_error_after_pulse_shuts_down(self):
        inst, args = Mock(), trend.parse_args(["--direction", "down"])
        with patch.object(trend, "stable_resistance", side_effect=[100e3, OSError("lost")]), \
                patch.object(smu, "apply_voltage_pulse", return_value=pulse()) as apply:
            with self.assertRaises(OSError):
                trend.run_trend(inst, args, lambda row: None, trend.validate_args(args))
        self.assertEqual(apply.call_count, 1)
        self.assertEqual(inst.write.call_args.args, (":OUTP OFF",))

    def test_settling_rejects_drift_and_waits_for_three_quiet_bursts(self):
        args, events = trend.parse_args([]), []
        bursts = [[reading(r)] for r in (100e3, 120e3, 121e3, 120e3)]
        with patch.object(smu, "read_resistance_samples", side_effect=bursts) as read, \
                patch.object(trend.time, "sleep"), patch.object(trend.time, "monotonic", return_value=0):
            result = trend.stable_resistance(Mock(), args, events.append, 0)
        self.assertEqual(result, 120e3)
        self.assertEqual(read.call_count, 4)

    def test_noisy_burst_resets_settling_window(self):
        args = trend.parse_args([])
        bursts = [[reading(100e3)], [reading(100e3)], [reading(90e3), reading(110e3)],
                  [reading(100e3)], [reading(100e3)], [reading(100e3)]]
        with patch.object(smu, "read_resistance_samples", side_effect=bursts) as read, \
                patch.object(trend.time, "sleep"), patch.object(trend.time, "monotonic", return_value=0):
            self.assertAlmostEqual(trend.stable_resistance(Mock(), args, lambda row: None, 0), 100e3)
        self.assertEqual(read.call_count, 6)

    def test_settling_timeout(self):
        with patch.object(trend.time, "monotonic", side_effect=[0, 20]):
            with self.assertRaisesRegex(RuntimeError, "did not settle"):
                trend.stable_resistance(Mock(), trend.parse_args([]), lambda row: None, 0)

    def test_parameters_reject_nan_and_unsafe_targets(self):
        for argv in (["--icc", "nan"], ["--target", "10000"], ["--t-pulse", "0.001"],
                     ["--verify-delay", "0"], ["--stable-reads", "1"], ["--max-step", "inf"]):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                trend.validate_args(trend.parse_args(argv))

    def test_errors_preserve_records_and_close_instrument(self):
        inst = Mock()
        inst._memristor_idn = "fake"
        # Keep the generated/cleaned test directory inside the known workspace.
        workspace = Path(__file__).resolve().parent
        folder = workspace / ("test_pulse_trend_" + uuid.uuid4().hex)
        folder.mkdir()
        try:
            fake_script = Path(folder) / "pulse_trend.py"
            with patch.object(trend, "__file__", str(fake_script)), patch("builtins.input", return_value=""), \
                    patch.object(smu, "connect_smu", return_value=inst), \
                    patch.object(smu, "configure_voltage_source", return_value=50e-6), \
                    patch.object(trend, "stable_resistance", side_effect=OSError("lost read")):
                self.assertEqual(trend.main([]), 1)
            metadata = json.loads(next(Path(folder).rglob("metadata.json")).read_text())
            self.assertEqual(metadata["stop_reason"], "error")
            self.assertTrue(next(Path(folder).rglob("events.csv")).exists())
            inst.close.assert_called_once()
            self.assertEqual(inst.write.call_args.args, (":OUTP OFF",))
        finally:
            assert folder.resolve().parent == workspace
            shutil.rmtree(folder)


if __name__ == "__main__":
    unittest.main()
