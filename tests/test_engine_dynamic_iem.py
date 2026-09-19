import tempfile
import unittest
from pathlib import Path

import numpy as np

from engine.earprint_engine import (
    alignment_scenarios,
    discover_txt,
)


class DynamicIEMRegressionTests(unittest.TestCase):
    """
    Regression tests for the dynamic-IEM contract.

    These tests exercise the same discovery + alignment primitives used by
    the engine. They do not alter or reinterpret the locked EarPrint math.
    """

    def _median_aligned_curve(self, f, curve, reference, bands):
        offsets, _, _, _ = alignment_scenarios(
            f, curve, reference, bands
        )
        return curve - float(np.median(offsets))

    def test_adding_one_iem_changes_equal_vote_centre(self):
        f = np.array([200, 300, 500, 700, 1000, 2000], dtype=float)
        reference = np.zeros_like(f)
        bands = [(200, 1000), (200, 500), (500, 1000)]

        # These IEMs deliberately have different SHAPES, not just different
        # constant level offsets. A pure level offset is removed by the
        # prompt-defined alignment step and therefore cannot test whether an
        # additional independent vote changes the EarPrint.
        curve_a = np.array([1, 1, 2, 2, 2, 5], dtype=float)
        curve_b = np.array([2, 2, 3, 3, 4, 6], dtype=float)
        curve_c = np.array([8, 8, 9, 10, 9, 12], dtype=float)

        a = self._median_aligned_curve(f, curve_a, reference, bands)
        b = self._median_aligned_curve(f, curve_b, reference, bands)
        c = self._median_aligned_curve(f, curve_c, reference, bands)

        centre_two = np.median(np.vstack([a, b]), axis=0)
        centre_three = np.median(np.vstack([a, b, c]), axis=0)

        self.assertFalse(np.allclose(centre_two, centre_three))

    def test_removing_added_iem_restores_previous_vote_centre(self):
        f = np.array([200, 300, 500, 700, 1000, 2000], dtype=float)
        reference = np.zeros_like(f)
        bands = [(200, 1000), (200, 500), (500, 1000)]

        curves = [
            np.array([1, 1, 2, 2, 2, 5], dtype=float),
            np.array([2, 2, 3, 3, 4, 6], dtype=float),
            np.array([8, 8, 9, 10, 9, 12], dtype=float),
        ]

        aligned = [
            self._median_aligned_curve(f, curve, reference, bands)
            for curve in curves
        ]

        baseline = np.median(np.vstack(aligned[:2]), axis=0)
        with_added = np.median(np.vstack(aligned[:3]), axis=0)
        after_removal = np.median(np.vstack(aligned[:2]), axis=0)

        self.assertFalse(np.allclose(baseline, with_added))
        self.assertTrue(np.allclose(baseline, after_removal))

    def test_dynamic_discovery_count_changes_when_iem_is_added(self):
        with tempfile.TemporaryDirectory() as tmp:
            preferred = Path(tmp) / "preferred"
            preferred.mkdir()

            (preferred / "A.txt").write_text("1 1\n", encoding="utf-8")
            (preferred / "B.txt").write_text("1 1\n", encoding="utf-8")

            found_before = discover_txt(preferred, "*.txt", [])
            self.assertEqual(len(found_before), 2)

            (preferred / "C.txt").write_text("1 1\n", encoding="utf-8")

            found_after = discover_txt(preferred, "*.txt", [])
            self.assertEqual(len(found_after), 3)
            self.assertEqual(
                [p.name for p in found_after],
                ["A.txt", "B.txt", "C.txt"],
            )

    def test_dynamic_discovery_count_returns_after_iem_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            preferred = Path(tmp) / "preferred"
            preferred.mkdir()

            for name in ("A.txt", "B.txt", "C.txt"):
                (preferred / name).write_text("1 1\n", encoding="utf-8")

            found_before = discover_txt(preferred, "*.txt", [])
            self.assertEqual(len(found_before), 3)

            (preferred / "C.txt").unlink()

            found_after = discover_txt(preferred, "*.txt", [])
            self.assertEqual(len(found_after), 2)
            self.assertEqual(
                [p.name for p in found_after],
                ["A.txt", "B.txt"],
            )


if __name__ == "__main__":
    unittest.main()
