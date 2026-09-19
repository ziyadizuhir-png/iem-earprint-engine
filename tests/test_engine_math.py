import tempfile
import unittest
from pathlib import Path

import numpy as np

from engine.earprint_engine import (
    alignment_scenarios,
    build_hybrid_curve,
    discover_txt,
    gaussian_once_strict_domain,
    retention_and_mask,
    sin2_boundary_taper,
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