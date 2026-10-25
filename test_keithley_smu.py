"""Unit tests for the shared Keithley connection helpers."""

import unittest
from unittest.mock import patch

from memristor_measure.keithley_smu import (
    MAX_SAFE_CURRENT_A,
    InstrumentConnectionError,
    ViReading,
    apply_voltage_pulse,
    configure_voltage_source,
    connect_smu,
    discover_supported_smus,
    error_is_clear,
    is_compliance,
    model_from_idn,
    parse_current_response,
    parse_vi_response,
    safe_shutdown,
    set_current_limit,
    validate_safety_limits,
)


class FakeInstrument:
    def __init__(self, idn):
        self.idn = idn
        self.closed = False
        self.commands = []
        self.timeout = None
        self.write_termination = None
        self.read_termination = None

    def clear(self):
        self.commands.append("clear")

    def write(self, command):
        self.commands.append(command)

    def query(self, command):
        if command == "*IDN?":
            return self.idn
        if command in (":SOUR:VOLT:ILIM?", ":SOUR:VOLT:ILIM:LEV?"):
            return "5.0e-5"
        if command == ":SYST:ERR?":
            return '0,"No error"'
        if command == ":SOUR:VOLT:ILIM:TRIP?":
            return "0"
        if command.startswith(':READ? "defbuffer1"'):
            return "0.1,2.0e-6,0.01,0"
        if command == "*OPC?":
            return "1"
        if command == ':TRAC:ACT? "defbuffer1"':
            return "1"
        if command.startswith(':TRAC:DATA? 1,1,"defbuffer1"'):
            return "0.5,1.0e-5,0.002,0"
        raise AssertionError(f"Unexpected query: {command}")

    def close(self):
        self.closed = True


class FakeResourceManager:
    def __init__(self, resources):
        self.resources = resources
        self.opened = []

    def list_resources(self):
        return tuple(self.resources)

    def open_resource(self, address):
        value = self.resources[address]
        if isinstance(value, Exception):
            raise value
        inst = FakeInstrument(value)
        self.opened.append((address, inst))
        return inst


class KeithleySmuTests(unittest.TestCase):
    def test_model_detection(self):
        self.assertEqual(model_from_idn("KEITHLEY INSTRUMENTS,MODEL 2461,123,1.7"), "2461")
        self.assertEqual(model_from_idn("Keithley Instruments Inc.,2450,123,1.0"), "2450")
        self.assertIsNone(model_from_idn("OTHER,2461,123,1.0"))

    def test_current_response_parsing(self):
        self.assertAlmostEqual(parse_current_response("0.1,2.5e-6,40000,0,0"), 2.5e-6)
        self.assertAlmostEqual(parse_current_response("-3.2e-6\n"), -3.2e-6)
        with self.assertRaises(ValueError):
            parse_current_response("")

    def test_discovery_filters_unreachable_and_unsupported_resources(self):
        rm = FakeResourceManager(
            {
                "USB::2461": "KEITHLEY INSTRUMENTS,MODEL 2461,123,1.7",
                "USB::OTHER": "OTHER,MODEL 1,123,1.0",
                "USB::OFFLINE": OSError("offline"),
            }
        )
        self.assertEqual(
            discover_supported_smus(rm),
            [("USB::2461", "KEITHLEY INSTRUMENTS,MODEL 2461,123,1.7")],
        )

    def test_auto_connect_selects_single_2461(self):
        rm = FakeResourceManager(
            {"USB::2461": "KEITHLEY INSTRUMENTS,MODEL 2461,123,1.7"}
        )
        inst = connect_smu("auto", rm=rm)
        self.assertEqual(inst.timeout, 15000)
        self.assertEqual(inst.write_termination, "\n")
        self.assertIn("*CLS", inst.commands)
        self.assertFalse(inst.closed)

    def test_explicit_address_accepts_2450(self):
        rm = FakeResourceManager(
            {"USB::2450": "KEITHLEY INSTRUMENTS,MODEL 2450,123,1.7"}
        )
        inst = connect_smu("USB::2450", rm=rm)
        self.assertFalse(inst.closed)

    def test_auto_connect_rejects_no_match(self):
        rm = FakeResourceManager({"USB::OTHER": "OTHER,MODEL 1,123,1.0"})
        with self.assertRaisesRegex(InstrumentConnectionError, "No reachable"):
            connect_smu("auto", rm=rm)

    def test_auto_connect_rejects_multiple_matches(self):
        rm = FakeResourceManager(
            {
                "USB::2450": "KEITHLEY INSTRUMENTS,MODEL 2450,A,1.0",
                "USB::2461": "KEITHLEY INSTRUMENTS,MODEL 2461,B,1.0",
            }
        )
        with self.assertRaisesRegex(InstrumentConnectionError, "Multiple"):
            connect_smu("auto", rm=rm)

    def test_explicit_address_rejects_unsupported_model(self):
        rm = FakeResourceManager({"USB::OTHER": "KEITHLEY INSTRUMENTS,MODEL 2400,A,1.0"})
        with self.assertRaisesRegex(InstrumentConnectionError, "not a supported"):
            connect_smu("USB::OTHER", rm=rm)

    def test_error_queue_parser(self):
        self.assertTrue(error_is_clear('0,"No error"'))
        self.assertTrue(error_is_clear("+0,No error"))
        self.assertFalse(error_is_clear('-100,"Command error"'))
        self.assertFalse(error_is_clear("not-a-code"))

    def test_hard_safety_limits(self):
        validate_safety_limits(2.0, MAX_SAFE_CURRENT_A)
        validate_safety_limits(0.1, 80e-6)
        with self.assertRaisesRegex(ValueError, "safety limit"):
            validate_safety_limits(2.01, 50e-6)
        validate_safety_limits(4.0, 50e-6, allow_overvoltage=True)

    def test_vi_response_and_compliance(self):
        reading = parse_vi_response(
            "0.099,2.0e-6,0.01,4",
            voltage_set_v=0.1,
            icc_a=50e-6,
        )
        self.assertAlmostEqual(reading.voltage_actual_v, 0.099)
        self.assertAlmostEqual(reading.resistance_ohm, 49500.0)
        self.assertFalse(reading.compliance)
        limited = parse_vi_response(
            "0.08,4.99e-5,0.02,0",
            voltage_set_v=0.1,
            icc_a=50e-6,
        )
        self.assertTrue(limited.compliance)
        self.assertTrue(is_compliance(49.3e-6, 50e-6))

    def test_configure_voltage_source_enables_readback(self):
        inst = FakeInstrument("KEITHLEY INSTRUMENTS,MODEL 2461,123,1.7")
        actual = configure_voltage_source(inst, 50e-6, 0.1)
        self.assertAlmostEqual(actual, 50e-6)
        self.assertIn(":SOUR:VOLT:READ:BACK ON", inst.commands)
        self.assertIn(":OUTP OFF", inst.commands)

    def test_rejects_current_limit_readback_above_request(self):
        for actual in ("5.01e-5", "nan", "inf", "0", "-1e-6"):
            with self.subTest(actual=actual):
                inst = FakeInstrument("KEITHLEY INSTRUMENTS,MODEL 2461,123,1.7")
                original = inst.query
                with patch.object(inst, "query", side_effect=lambda cmd:
                                  actual if cmd == ":SOUR:VOLT:ILIM?" else original(cmd)):
                    with self.assertRaisesRegex(RuntimeError, "Unsafe current limit"):
                        set_current_limit(inst, 50e-6)
                self.assertNotIn(":OUTP ON", inst.commands)
                self.assertEqual(inst.commands[-1], ":OUTP OFF")

    def test_compliance_or_invalid_software_read_skips_remaining_hold(self):
        for current, limited in ((49.5e-6, True), (50.1e-6, False), (float("nan"), False)):
            with self.subTest(current=current):
                inst = FakeInstrument("KEITHLEY INSTRUMENTS,MODEL 2461,123,1.7")
                reading = ViReading(1.0, 1.0, current, 50e-6, limited)
                with patch("memristor_measure.keithley_smu.read_vi", return_value=reading), \
                        patch("memristor_measure.keithley_smu.time.sleep") as sleep, \
                        patch("memristor_measure.keithley_smu.time.perf_counter", return_value=0):
                    result = apply_voltage_pulse(inst, mode="software", voltage_v=1.0,
                                                 width_s=1.0, relax_s=0, current_limit_a=50e-6)
                sleep.assert_not_called()
                self.assertFalse(result.completed)
                self.assertEqual(inst.commands[-1], ":OUTP OFF")

    def test_software_read_exception_turns_output_off(self):
        inst = FakeInstrument("KEITHLEY INSTRUMENTS,MODEL 2461,123,1.7")
        with patch("memristor_measure.keithley_smu.read_vi", side_effect=OSError("timeout")):
            with self.assertRaises(OSError):
                apply_voltage_pulse(inst, mode="software", voltage_v=1.0,
                                    width_s=0.02, relax_s=0, current_limit_a=50e-6)
        self.assertEqual(inst.commands[-1], ":OUTP OFF")

    def test_software_pulse_records_reading_and_returns_to_zero(self):
        inst = FakeInstrument("KEITHLEY INSTRUMENTS,MODEL 2461,123,1.7")
        result = apply_voltage_pulse(
            inst,
            mode="software",
            voltage_v=0.5,
            width_s=0.02,
            relax_s=0.0,
            current_limit_a=50e-6,
        )
        self.assertTrue(result.completed)
        self.assertEqual(result.mode, "software")
        self.assertIn(":SOUR:VOLT:LEV 0", inst.commands)

    def test_hardware_pulse_rejects_2450(self):
        inst = FakeInstrument("KEITHLEY INSTRUMENTS,MODEL 2450,123,1.7")
        inst._memristor_model = "2450"
        with self.assertRaisesRegex(ValueError, "requires Keithley 2461"):
            apply_voltage_pulse(
                inst,
                mode="hardware",
                voltage_v=0.5,
                width_s=0.02,
                relax_s=0.01,
                current_limit_a=50e-6,
            )

    def test_hardware_pulse_uses_single_safe_2461_train(self):
        inst = FakeInstrument("KEITHLEY INSTRUMENTS,MODEL 2461,123,1.7")
        inst._memristor_model = "2461"
        inst.timeout = 15000
        result = apply_voltage_pulse(
            inst,
            mode="hardware",
            voltage_v=0.5,
            width_s=0.02,
            relax_s=0.01,
            current_limit_a=50e-6,
        )
        pulse_commands = [
            command for command in inst.commands
            if command.startswith(":SOUR:PULS:TR:VOLT")
        ]
        self.assertEqual(len(pulse_commands), 1)
        self.assertIn(",1,ON,", pulse_commands[0])
        self.assertTrue(pulse_commands[0].endswith(",ON"))
        self.assertTrue(result.completed)
        self.assertEqual(result.mode, "hardware")

    def test_safe_shutdown_never_raises(self):
        inst = FakeInstrument("KEITHLEY INSTRUMENTS,MODEL 2461,123,1.7")
        safe_shutdown(inst)
        self.assertEqual(
            inst.commands[-3:],
            [":ABOR", ":SOUR:VOLT:LEV 0", ":OUTP OFF"],
        )

    def test_init_r_restores_read_voltage_for_every_device(self):
        from memristor_measure import init_R

        inst = FakeInstrument("KEITHLEY INSTRUMENTS,MODEL 2461,123,1.7")
        reading = ViReading(
            voltage_set_v=0.1,
            voltage_actual_v=0.1,
            current_a=1e-6,
            icc_a=50e-6,
            compliance=False,
        )
        with patch.object(init_R, "read_vi", return_value=reading):
            init_R.measure_resistance(inst, 1, 0.0, 0.1, 50e-6)
            init_R.measure_resistance(inst, 1, 0.0, 0.1, 50e-6)

        self.assertEqual(inst.commands.count(":SOUR:VOLT:LEV 0.1"), 2)


if __name__ == "__main__":
    unittest.main()
