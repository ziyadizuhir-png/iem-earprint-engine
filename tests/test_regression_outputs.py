"""Dynamic regression checks for generated EarPrint artifacts.

The input directories are the source of truth.

IMPORTANT:
- No IEM/model/target filename is hard-coded here.
- Adding a new input/preferred/*.txt automatically expands the expected
  preferred-vote set.
- Removing an input/preferred/*.txt automatically removes it from the
  expected preferred-vote set.
- Adding/removing input/targets/*.txt does the same for target artifacts.

This prevents dataset changes from requiring test-code edits.
"""

import json
from pathlib import Path

import yaml

from engine.input_hash import calculate_hash, source_files


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config" / "project.yaml"


def _discover(directory: Path) -> list[str]:
    """Return deterministic, filename-only discovery from a data directory."""
    return sorted(
        p.name
        for p in directory.glob("*.txt")
        if p.is_file()
    )


def _manifest() -> dict:
    path = REPORTS / "manifest.json"
    assert path.is_file(), f"Missing generated manifest: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_names(pattern: str, directory: Path) -> list[str]:
    return sorted(
        p.name
        for p in directory.glob(pattern)
        if p.is_file()
    )


def test_generated_manifest_matches_dynamic_input_discovery():
    """Manifest must describe exactly the files currently in input/."""
    cfg = yaml.safe_load(
        CONFIG.read_text(encoding="utf-8")
    )

    preferred = _discover(ROOT / "input" / "preferred")
    targets = _discover(ROOT / "input" / "targets")
    manifest = _manifest()

    assert manifest["dynamic_discovery"] is True

    # These are deliberately derived from the filesystem, never hard-coded.
    assert sorted(manifest["preferred_votes"]) == preferred
    assert sorted(manifest["targets_discovered"]) == targets

    assert manifest["independent_vote_count"] == len(preferred)
    assert manifest["target_count"] == len(targets)

    assert (
        manifest["low_frequency_reference"]
        == cfg["low_frequency_reference"]["file"]
    )


def test_preferred_dataset_is_dynamic():
    """Every preferred TXT is an independent vote; no fixed model list."""
    preferred = _discover(ROOT / "input" / "preferred")
    manifest = _manifest()

    assert len(preferred) >= 1
    assert manifest["independent_vote_count"] == len(preferred)
    assert sorted(manifest["preferred_votes"]) == preferred


def test_generated_outputs_cover_every_discovered_target():
    """Every discovered target must have its complete generated artifact set."""
    targets = _discover(ROOT / "input" / "targets")

    robust = _artifact_names("*__robust_target.txt", OUTPUT)
    masks = _artifact_names("*__mask.txt", OUTPUT)
    handoffs = _artifact_names(
        "*__adaptive_handoff.txt",
        REPORTS,
    )

    assert (OUTPUT / "pure_earprint_dynamic.txt").is_file()

    expected_robust = sorted(
        f"{Path(name).stem}__robust_target.txt"
        for name in targets
    )
    expected_masks = sorted(
        f"{Path(name).stem}__mask.txt"
        for name in targets
    )
    expected_handoffs = sorted(
        f"{Path(name).stem}__adaptive_handoff.txt"
        for name in targets
    )

    assert robust == expected_robust
    assert masks == expected_masks
    assert handoffs == expected_handoffs


def test_no_preferred_filename_is_hard_coded_in_this_regression_contract():
    """Guard the test itself against accidentally reintroducing a fixed list."""
    source = Path(__file__).read_text(encoding="utf-8")

    # The contract should use filesystem discovery rather than naming models.
    forbidden_model_names = (
        "Pudding.txt",
        "Ceramics_Ultra.txt",
        "Galaxy_Buds2_Pro.txt",
        "Svanar_Wireless_Jr.txt",
        "Timeless.txt",
        "WF1000XM4.txt",
    )

    for name in forbidden_model_names:
        assert name not in source


def test_validation_report_is_pass():
    validation_path = REPORTS / "validation.txt"
    assert validation_path.is_file(), (
        f"Missing validation report: {validation_path}"
    )

    validation = validation_path.read_text(
        encoding="utf-8"
    )

    assert "Status: PASS" in validation


def test_handoff_statuses_are_explicit_and_fail_safe():
    allowed = {
        "HANDOFF_ACCEPTED",
        "NO_STABLE_HANDOFF",
    }

    for report in REPORTS.glob("*__adaptive_handoff.txt"):
        status = next(
            (
                line.split(":", 1)[1].strip()
                for line in report.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.startswith("status:")
            ),
            None,
        )

        assert status in allowed, (
            f"Unexpected handoff status in {report.name}: {status}"
        )


def test_build_input_hash_is_deterministic_and_nonempty():
    first = calculate_hash()
    second = calculate_hash()

    assert len(source_files()) > 0
    assert len(first) == 64
    assert first == second


if __name__ == "__main__":
    import unittest

    unittest.main()
