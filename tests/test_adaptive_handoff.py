from __future__ import annotations

import unittest
import numpy as np

from engine.adaptive_handoff import (
    _monotone_bridge,
    adaptive_masked_handoff,
    nominal_seam_compatible,
)


class TestAdaptiveHandoff(unittest.TestCase):

    def test_monotone_bridge_has_exact_c1_and_no_overshoot_or_reversal(self):
        info = _monotone_bridge(
            y0=0.0,
            y1=-1.0,
            d0=-0.5,
            d1=-0.5,
            span_octaves=0.5,
        )
        self.assertTrue(info["c1_slope_match"])
        self.assertLessEqual(info["overshoot_db"], 1e-8)
        self.assertLessEqual(info["undershoot_db"], 1e-8)
        self.assertFalse(info["slope_reversal"])
        self.assertEqual(info["artificial_extrema_count"], 0)

    def test_nominal_compatible_curve_is_unchanged(self):
        freq = np.geomspace(800.0, 2000.0, 241)
        x = np.log2(freq / 1000.0)
        target = 60.0 + 2.0 * x
        masked = target + 0.01 * x

        ok, diag = nominal_seam_compatible(
            freq, target, masked, nominal_hz=1000.0
        )
        self.assertTrue(ok)
        self.assertTrue(diag["slope_magnitude_match_pass"])
        self.assertTrue(diag["curvature_stable"])

        out, report = adaptive_masked_handoff(
            freq, target, masked,
            nominal_hz=1000.0,
            domain_end_hz=2000.0,
        )
        np.testing.assert_allclose(out, masked, atol=0.0, rtol=0.0)
        self.assertEqual(report["status"], "NORMAL_HANDOFF")

    def test_nominal_slope_magnitude_mismatch_is_rejected(self):
        freq = np.geomspace(800.0, 2000.0, 241)
        x = np.log2(freq / 1000.0)
        target = 60.0 + 2.0 * x
        masked = 60.0 + 2.5 * x

        ok, diag = nominal_seam_compatible(
            freq, target, masked, nominal_hz=1000.0
        )
        self.assertFalse(ok)
        self.assertFalse(diag["slope_magnitude_match_pass"])

    def test_incompatible_nominal_seam_uses_earliest_valid_adaptive_bridge(self):
        freq = np.geomspace(1000.0, 3000.0, 481)
        x = np.log2(freq / 1000.0)
        target = 60.0 + 2.0 * x
        mask = -0.9 * np.exp(-((x - 0.18) / 0.10) ** 2) + 0.25 * x
        masked = target + mask

        out, report = adaptive_masked_handoff(
            freq, target, masked,
            nominal_hz=1000.0,
            domain_end_hz=3000.0,
            min_transition_octaves=0.125,
            max_transition_octaves=0.8,
            stability_window_octaves=0.20,
        )

        self.assertEqual(report["status"], "ADAPTIVE_HANDOFF")
        h = report["actual_handoff_hz"]
        e = report["transition_end_hz"]
        self.assertGreater(h, 1000.0)
        self.assertGreater(e, h)
        self.assertGreaterEqual(np.log2(e / h), 0.125)
        self.assertLessEqual(np.log2(e / h), 0.8)

        low = freq <= h
        high = freq > e
        np.testing.assert_allclose(out[low], target[low], atol=1e-10)
        np.testing.assert_allclose(out[high], masked[high], atol=1e-10)

        self.assertTrue(report["c1_slope_match"])
        self.assertTrue(report["destination_slope_stable"])
        self.assertTrue(report["destination_curvature_stable"])
        self.assertTrue(report["continuity_pass"])
        self.assertTrue(report["monotonicity_pass"])
        self.assertTrue(report["overshoot_pass"])
        self.assertTrue(report["undershoot_pass"])
        self.assertTrue(report["slope_reversal_pass"])

    def test_no_stable_handoff_returns_base_target(self):
        freq = np.geomspace(1000.0, 12000.0, 241)
        x = np.log2(freq / 1000.0)
        target = np.full_like(freq, 60.0)
        masked = target - 0.5 * x
        out, report = adaptive_masked_handoff(
            freq, target, masked,
            nominal_hz=1000.0,
            domain_end_hz=12000.0,
            min_transition_octaves=0.8,
            max_transition_octaves=0.8,
            stability_window_octaves=0.20,
        )

        self.assertIn(report["status"], {"NORMAL_HANDOFF", "NO_STABLE_HANDOFF"})
        if report["status"] == "NO_STABLE_HANDOFF":
            np.testing.assert_allclose(out, target, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()