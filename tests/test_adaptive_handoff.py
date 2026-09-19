from __future__ import annotations

import unittest

import numpy as np

from engine.adaptive_handoff import (
    _monotone_bridge,
    adaptive_masked_handoff,
    nominal_seam_compatible,
)


class TestAdaptiveHandoff(unittest.TestCase):

    def test_monotone_bridge_has_no_overshoot_or_reversal(self):
        info = _monotone_bridge(
            y0=0.0,
            y1=-1.0,
            d0=-0.5,
            d1=-0.5,
            span_octaves=0.5,
        )
        self.assertLessEqual(info["overshoot_db"], 1e-9)
        self.assertLessEqual(info["undershoot_db"], 1e-9)
        self.assertFalse(info["slope_reversal"])
        self.assertEqual(info["artificial_extrema_count"], 0)

    def test_nominal_compatible_curve_is_unchanged(self):
        freq = np.geomspace(800.0, 2000.0, 241)
        target = 60.0 + 2.0 * np.log2(freq / 1000.0)
        masked = target + 0.1 * np.log2(freq / 1000.0)

        ok, diag = nominal_seam_compatible(
            freq, target, masked, nominal_hz=1000.0
        )
        self.assertTrue(ok)
        self.assertEqual(diag["artificial_extrema_count"], 0)

        out, report = adaptive_masked_handoff(
            freq,
            target,
            masked,
            nominal_hz=1000.0,
            domain_end_hz=2000.0,
        )
        np.testing.assert_allclose(out, masked, atol=0.0, rtol=0.0)
        self.assertEqual(report["status"], "NORMAL_HANDOFF")

    def test_incompatible_nominal_seam_uses_adaptive_bridge(self):
        freq = np.geomspace(1000.0, 3000.0, 481)
        x = np.log2(freq / 1000.0)

        # Target rises continuously. The masked curve initially bends
        # downward, then becomes continuously rising again.
        target = 60.0 + 2.0 * x
        mask = -0.9 * np.exp(-((x - 0.18) / 0.10) ** 2) + 0.25 * x
        masked = target + mask

        out, report = adaptive_masked_handoff(
            freq,
            target,
            masked,
            nominal_hz=1000.0,
            domain_end_hz=3000.0,
            min_transition_octaves=0.125,
            max_transition_octaves=0.8,
            stability_window_octaves=0.20,
        )

        self.assertEqual(report["status"], "ADAPTIVE_HANDOFF")
        self.assertGreater(report["actual_handoff_hz"], 1000.0)
        self.assertGreater(report["transition_end_hz"],
                           report["actual_handoff_hz"])
        self.assertTrue(report["continuity_pass"])
        self.assertTrue(report["monotonicity_pass"])
        self.assertTrue(report["overshoot_pass"])
        self.assertTrue(report["undershoot_pass"])
        self.assertTrue(report["slope_reversal_pass"])

        h = report["actual_handoff_hz"]
        e = report["transition_end_hz"]

        # Target remains exact through H.
        low = freq <= h
        np.testing.assert_allclose(out[low], target[low], atol=1e-10)

        # Masked curve is restored after E.
        high = freq > e
        np.testing.assert_allclose(out[high], masked[high], atol=1e-10)

    def test_no_stable_handoff_does_not_invent_a_curve(self):
        freq = np.geomspace(1000.0, 12000.0, 241)
        x = np.log2(freq / 1000.0)

        target = np.full_like(freq, 60.0)
        masked = target - 0.5 * x

        out, report = adaptive_masked_handoff(
            freq,
            target,
            masked,
            nominal_hz=1000.0,
            domain_end_hz=12000.0,
            min_transition_octaves=0.8,
            max_transition_octaves=0.8,
            stability_window_octaves=0.20,
        )

        # The locked failure behavior is to return the BaseTarget rather
        # than inventing an undocumented correction.
        self.assertIn(
            report["status"],
            {"NORMAL_HANDOFF", "NO_STABLE_HANDOFF"},
        )
        if report["status"] == "NO_STABLE_HANDOFF":
            np.testing.assert_allclose(out, target, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
