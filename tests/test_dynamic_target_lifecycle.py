import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

import engine.earprint_engine as engine


class DynamicTargetLifecycleTests(unittest.TestCase):
    """End-to-end regression test for dynamic target discovery and output cleanup."""

    @staticmethod
    def _write_curve(path, freq, level):
        with path.open("w", encoding="utf-8") as f:
            for x, y in zip(freq, level):
                f.write(f"{x:.12g}\t{y:.6f}\n")

    def _make_config(self, root):
        config_dir = root / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        cfg = {
            "specification": "General_Prompt_EarPrint_8_MATH_LOCKED.txt",
            "specification_addendum": "General_Prompt_EarPrint_8_MATH_LOCKED_boundary_addendum.txt",
            "discovery": {
                "preferred_glob": "*.txt",
                "target_glob": "*.txt",
                "ignore_files": [],
            },
            "low_frequency_reference": {"file": "IEF2025.txt"},
            "frequency": {
                "personal_start_hz": 1000,
                "personal_end_hz": 12000,
                "output_start_hz": 20,
                "output_end_hz": 12000,
            },
            "alignment": {
                "bands_hz": [[200, 1000], [200, 500], [500, 1000]],
                "statistic": "median",
            },
            "smoothing": {
                "method": "gaussian",
                "sigma_indices": 1,
                "radius_indices": 2,
                "padding": "nearest",
                "passes": 1,
            },
            "high_frequency_extension": {
                "method": "log_frequency",
                "slope_db_per_octave": -6,
                "display_only": True,
            },
            "repeatability": {
                "enabled": False,
                "method": None,
                "note": "Inactive until valid same-IEM repeats exist and a prompt-defined estimator is supplied.",
            },
            "robust_mask": {
                "enabled": True,
                "method": "one_pass_reweighted_huber",
                "raw_mask_definition": "Huber robust consensus Centre",
                "huber": {
                    "tuning_constant": 1.345,
                    "scale_factor": 1.4826,
                    "scale_floor_db": 0.15,
                    "passes": 1,
                },
                "diagnostics": [
                    "cross_iem_mad",
                    "alignment_uncertainty",
                    "repeatability_floor",
                ],
                "note": "Correction magnitude is not attenuated by a Retention formula.",
                "boundary_taper": {
                    "method": "sin2",
                    "lower_transition_octaves": 1 / 3,
                    "upper_transition_octaves": 0.25,
                },
            },
            "adaptive_handoff": {
                "enabled": True,
                "nominal_anchor_hz": 1000,
                "min_transition_octaves": 0.125,
                "max_transition_octaves": 0.8,
                "stability_window_octaves": 0.2,
                "method": "earliest stable compatible handoff + monotonic cubic Hermite bridge",
            },
            "hybrids": {
                "enabled": True,
                "mode": "on_demand",
                "default_join_hz": 1000,
            },
            "output": {
                "delimiter": "\t",
                "decimals": 6,
                "write_masks": True,
                "write_robust_targets": True,
                "write_pure_earprint": True,
                "write_hybrids": False,
            },
        }

        (config_dir / "project.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False),
            encoding="utf-8",
        )

    def test_add_target_generates_outputs_and_removal_cleans_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preferred_dir = root / "input" / "preferred"
            target_dir = root / "input" / "targets"
            out_dir = root / "output"
            reports_dir = root / "reports"

            preferred_dir.mkdir(parents=True)
            target_dir.mkdir(parents=True)
            self._make_config(root)

            freq = np.geomspace(20.0, 12000.0, 40)
            iem_a = 80 + 3 * np.sin(np.log(freq))
            iem_b = 80 + 4 * np.sin(np.log(freq) + 0.25)

            self._write_curve(preferred_dir / "A.txt", freq, iem_a)
            self._write_curve(preferred_dir / "B.txt", freq, iem_b)

            base = 80 + 2 * np.sin(np.log(freq))
            self._write_curve(target_dir / "IEF2025.txt", freq, base)
            self._write_curve(target_dir / "Target_B.txt", freq, base + 1.0)

            patched = patch.multiple(
                engine,
                CFG=root / "config" / "project.yaml",
                PREFERRED_DIR=preferred_dir,
                TARGET_DIR=target_dir,
                OUT=out_dir,
                REPORTS=reports_dir,
                ROOT=root,
            )

            with patched:
                engine.main()

                self.assertTrue((out_dir / "Target_B__mask.txt").exists())
                self.assertTrue((out_dir / "Target_B__robust_target.txt").exists())

                stats = (reports_dir / "robust_statistics.csv").read_text(
                    encoding="utf-8"
                )
                self.assertIn("huber_tuning_constant", stats)
                self.assertIn("huber_scale_factor", stats)
                self.assertIn("huber_scale_floor_db", stats)
                self.assertNotIn("retention_median", stats)
                self.assertNotIn("total_uncertainty_median_db", stats)

                self._write_curve(target_dir / "Target_C.txt", freq, base - 1.0)
                engine.main()

                self.assertTrue((out_dir / "Target_C__mask.txt").exists())
                self.assertTrue((out_dir / "Target_C__robust_target.txt").exists())

                (target_dir / "Target_C.txt").unlink()
                engine.main()

                self.assertFalse((out_dir / "Target_C__mask.txt").exists())
                self.assertFalse((out_dir / "Target_C__robust_target.txt").exists())
                self.assertTrue((out_dir / "Target_B__mask.txt").exists())
                self.assertTrue((out_dir / "Target_B__robust_target.txt").exists())


if __name__ == "__main__":
    unittest.main()