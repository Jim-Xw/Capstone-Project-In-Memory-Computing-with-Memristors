"""Unit tests for closed-loop SET/RESET guard decisions."""

import unittest

from memristor_measure.pulse_set_reset import (
    decide_next_action,
    predicted_write_current,
    r_floor_ohm,
    write_would_hit_icc,
)


class PulseSetResetGuardTests(unittest.TestCase):
    def test_r_floor_at_1v_50ua(self):
        self.assertAlmostEqual(r_floor_ohm(1.0, 50e-6, 1.2), 24e3, places=0)

    def test_high_r_allows_set(self):
        self.assertFalse(write_would_hit_icc(200e3, 1.0, 50e-6))
        action, reason = decide_next_action("set", 200e3, 1.0, -1.0, 50e-6)
        self.assertEqual((action, reason), ("set", "ok"))

    def test_near_floor_reverses_set_to_reset(self):
        self.assertTrue(write_would_hit_icc(22e3, 1.0, 50e-6))
        action, reason = decide_next_action("set", 22e3, 1.0, -1.0, 50e-6)
        self.assertEqual(action, "reset")
        self.assertEqual(reason, "set_would_exceed_icc")

    def test_low_r_blocks_set_without_unsafe_reset(self):
        self.assertTrue(write_would_hit_icc(12e3, 1.0, 50e-6))
        action, reason = decide_next_action("set", 12e3, 1.0, -1.0, 50e-6)
        self.assertEqual((action, reason), ("stop", "set_would_exceed_icc"))

    def test_very_low_r_stops_reset(self):
        action, reason = decide_next_action("reset", 2.3e3, 1.0, -1.0, 50e-6)
        self.assertEqual((action, reason), ("stop", "reset_cannot_apply_full_voltage"))

    def test_set_stop_mode_does_not_reverse(self):
        action, reason = decide_next_action(
            "set", 12e3, 1.0, -1.0, 50e-6, on_limit="stop"
        )
        self.assertEqual((action, reason), ("stop", "set_would_exceed_icc"))

    def test_reset_stops_at_r_max(self):
        action, reason = decide_next_action(
            "reset", 1.2e6, 1.0, -1.0, 50e-6, r_max=1e6
        )
        self.assertEqual((action, reason), ("stop", "reached_r_max"))

    def test_predicted_current(self):
        self.assertAlmostEqual(predicted_write_current(20e3, 1.0), 50e-6, places=9)


if __name__ == "__main__":
    unittest.main()
