import tempfile
import unittest
from pathlib import Path

import numpy as np

from engine.earprint_engine import (
    alignment_scenarios,
    build_hybrid_curve,
    discover_txt,
    gaussian_once_strict_domain,
    quarter_octave_sin2_taper,
    retention_and_mask,
)


class EarPrintMathTests(unittest.TestCase):
    def test_three_alignment_scenarios_use_inclusive_band_medians(self):
        f = np.array([200, 300, 500, 700, 1000, 2000], dtype=float)
        reference = np.zeros_like(f)
        curve = np.array([1, 1, 2, 2, 2, 5], dtype=float)
        bands = [(200, 1000), (200, 500), (500, 1000)]

        offsets, scenarios, centre, uncertainty = alignment_scenarios(
            f, curve, reference, bands
        )

        expected_offsets = np.array([2.0, 1.0, 2.0])
        self.assertTrue(np.allclose(offsets, expected_offsets))
        self.assertTrue(np.allclose(scenarios, np.vstack([curve - x for x in expected_offsets])))
        self.assertTrue(np.allclose(centre, np.median(scenarios, axis=0)))
        self.assertTrue(np.all(uncertainty >= 0))

    def test_retention_zero_zero_is_deterministically_zero(self):
        centre = np.array([-2.0, 0.0, 4.0, 0.0])
        uncertainty = np.array([1.0, 0.0, 2.0, 3.0])
        retention, mask = retention_and_mask(centre, uncertainty)
        expected_r = np.array([2 / 3, 0.0, 2 / 3, 0.0])
        expected_m = np.array([-4 / 3, 0.0, 8 / 3, 0.0])
        self.assertTrue(np.allclose(retention, expected_r))
        self.assertTrue(np.allclose(mask, expected_m))

    def test_gaussian_pass_is_limited_to_strict_personal_domain(self):
        f = np.arange(20.0, 20001.0, 20.0)
        y = np.sin(np.log(f))
        out = gaussian_once_strict_domain(y, f, 1000, 12000, 4, 8)
        self.assertTrue(np.allclose(out[f <= 1000], y[f <= 1000]))
        self.assertTrue(np.allclose(out[f >= 12000], y[f >= 12000]))
        self.assertFalse(np.allclose(out[(f > 1000) & (f < 12000)], y[(f > 1000) & (f < 12000)]))

    def test_quarter_octave_taper_has_exact_domain_boundaries(self):
        q = 2 ** 0.25
        f = np.array([1000.0, 1000*q, 2000.0, 12000/q, 12000.0])
        w = quarter_octave_sin2_taper(f, 1000.0, 12000.0)
        self.assertEqual(w[0], 0.0)
        self.assertAlmostEqual(w[1], 1.0, places=12)
        self.assertAlmostEqual(w[2], 1.0, places=12)
        self.assertAlmostEqual(w[3], 1.0, places=12)
        self.assertEqual(w[4], 0.0)

    def test_hybrid_preserves_lower_target_and_is_continuous_at_join(self):
        f = np.array([500.0, 800.0, 1000.0, 1200.0, 2000.0], dtype=float)
        target = np.array([60, 61, 62, 63, 64], dtype=float)
        pure = np.array([59, 60, 61, 70, 71], dtype=float)
        hybrid, shift = build_hybrid_curve(f, target, pure, 1000.0)
        self.assertTrue(np.allclose(hybrid[f <= 1000], target[f <= 1000]))
        self.assertAlmostEqual(shift, 1.0, places=12)
        self.assertAlmostEqual(hybrid[3], pure[3] + 1.0, places=12)

    def test_dynamic_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for name in ("A.txt", "B.txt", "ignore.txt"):
                (d / name).write_text("1 1\n", encoding="utf-8")
            found = discover_txt(d, "*.txt", ["ignore.txt"])
            self.assertEqual([p.name for p in found], ["A.txt", "B.txt"])


if __name__ == "__main__":
    unittest.main()
