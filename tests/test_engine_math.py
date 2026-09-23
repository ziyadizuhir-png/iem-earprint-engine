import tempfile
import unittest
from pathlib import Path

import numpy as np

from engine.earprint_engine import (
    alignment_scenarios,
    build_hybrid_curve,
    discover_txt,
    extend_high_frequency_display,
    gaussian_once_strict_domain,
    huber_consensus,
    sin2_boundary_taper,
)


class EarPrintMathTests(unittest.TestCase):
    def test_high_frequency_extension_anchors_at_exact_boundary(self):
        freq = np.array([1000.0, 9000.0, 15000.0], dtype=float)
        level = np.array([0.0, 9.0, 15.0], dtype=float)
        out = extend_high_frequency_display(freq, level, 12000.0, -6.0)

        boundary = np.interp(np.log(12000.0), np.log(freq), level)
        expected_15k = boundary - 6.0 * np.log2(15000.0 / 12000.0)
        self.assertAlmostEqual(out[2], expected_15k, places=12)
        self.assertAlmostEqual(out[1], level[1], places=12)

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
        self.assertTrue(
            np.allclose(
                scenarios,
                np.vstack(
                    [curve - x for x in expected_offsets]
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                centre,
                np.median(scenarios, axis=0),
            )
        )
        self.assertTrue(np.all(uncertainty >= 0))

    def test_huber_consensus_downweights_large_outlier(self):
        values = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=float,
        )

        centre, scale, weights = huber_consensus(
            values,
            tuning_constant=1.345,
            scale_factor=1.4826,
            scale_floor_db=0.15,
        )

        expected_weight = (1.345 * 0.15) / 10.0
        expected_centre = (
            10.0 * expected_weight
            / (2.0 + expected_weight)
        )

        self.assertTrue(np.allclose(scale, 0.15))
        self.assertTrue(
            np.allclose(
                weights[:2],
                1.0,
            )
        )
        self.assertTrue(
            np.allclose(
                weights[2],
                expected_weight,
            )
        )
        self.assertTrue(
            np.allclose(
                centre,
                expected_centre,
            )
        )

    def test_huber_consensus_preserves_consensus_without_outlier(self):
        values = np.array(
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ],
            dtype=float,
        )

        centre, scale, weights = huber_consensus(values)

        self.assertTrue(
            np.allclose(
                centre,
                [1.0, 2.0, 3.0],
            )
        )
        self.assertTrue(
            np.allclose(
                weights,
                1.0,
            )
        )
        self.assertTrue(
            np.allclose(
                scale,
                0.15,
            )
        )

    def test_gaussian_pass_is_limited_to_strict_personal_domain(self):
        f = np.arange(
            20.0,
            20001.0,
            20.0,
        )
        y = np.sin(np.log(f))
        out = gaussian_once_strict_domain(
            y,
            f,
            1000,
            12000,
            4,
            8,
        )
        self.assertTrue(
            np.allclose(
                out[f <= 1000],
                y[f <= 1000],
            )
        )
        self.assertTrue(
            np.allclose(
                out[f >= 12000],
                y[f >= 12000],
            )
        )
        self.assertFalse(
            np.allclose(
                out[
                    (f > 1000)
                    & (f < 12000)
                ],
                y[
                    (f > 1000)
                    & (f < 12000)
                ],
            )
        )

    def test_sin2_boundary_taper_has_exact_domain_boundaries(self):
        lower_end = 1000.0 * (2 ** (1 / 3))
        upper_start = 12000.0 / (2 ** 0.25)
        f = np.array([
            1000.0,
            lower_end,
            2000.0,
            upper_start,
            12000.0,
        ])
        w = sin2_boundary_taper(
            f,
            1000.0,
            12000.0,
            lower_transition_octaves=1 / 3,
            upper_transition_octaves=0.25,
        )
        self.assertEqual(w[0], 0.0)
        self.assertAlmostEqual(
            w[1],
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            w[2],
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            w[3],
            1.0,
            places=12,
        )
        self.assertEqual(w[4], 0.0)

    def test_hybrid_preserves_lower_target_and_is_continuous_at_join(self):
        f = np.array([
            500.0,
            800.0,
            1000.0,
            1200.0,
            2000.0,
        ])
        target = np.array(
            [60, 61, 62, 63, 64],
            dtype=float,
        )
        pure = np.array(
            [59, 60, 61, 70, 71],
            dtype=float,
        )
        hybrid, shift = build_hybrid_curve(
            f,
            target,
            pure,
            1000.0,
        )
        self.assertTrue(
            np.allclose(
                hybrid[f <= 1000],
                target[f <= 1000],
            )
        )
        self.assertAlmostEqual(
            shift,
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            hybrid[3],
            pure[3] + 1.0,
            places=12,
        )

    def test_dynamic_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for name in (
                "A.txt",
                "B.txt",
                "ignore.txt",
            ):
                (d / name).write_text(
                    "1 1\n",
                    encoding="utf-8",
                )
            found = discover_txt(
                d,
                "*.txt",
                ["ignore.txt"],
            )
            self.assertEqual(
                [p.name for p in found],
                ["A.txt", "B.txt"],
            )


if __name__ == "__main__":
    unittest.main()
