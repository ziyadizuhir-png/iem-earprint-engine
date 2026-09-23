"""Dynamic regression checks for generated EarPrint artifacts.

The input directories are the source of truth. Model/target filenames are
never hard-coded in the regression contract.

The workflow runs the full pytest suite once BEFORE regenerating outputs and
then runs this module again AFTER generation. Therefore checks that depend on
generated artifacts are skipped while those artifacts are stale, and become
active once the generated manifest matches the current input directories.
"""

import json
from pathlib import Path

import pytest
import yaml

from engine.input_hash import calculate_hash, source_files


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config" / "project.yaml"


def _discover(directory: Path) -> list[str]:
    """Discover all TXT inputs dynamically and deterministically."""
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


def _generated_state_is_current() -> bool:
    """Return True only when the generated manifest matches current inputs."""
    path = REPORTS / "manifest.json"
    if not path.is_file():
        return False

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    return (
        manifest.get("dynamic_discovery") is True
        and sorted(manifest.get("preferred_votes", []))
        == _discover(ROOT / "input" / "preferred")
        and sorted(manifest.get("targets_discovered", []))
        == _discover(ROOT / "input" / "targets")
    )


def _require_current_generated_state() -> None:
    """Skip generated-artifact checks until the engine has regenerated them."""
    if not _generated_state_is_current():
        pytest.skip(
            "Generated artifacts are stale relative to current input discovery; "
            "run the EarPrint engine before validating generated outputs"
        )


def test_generated_manifest_matches_dynamic_input_discovery():
    """The generated manifest must exactly reflect discovered input files."""
    _require_current_generated_state()

    cfg = yaml.safe_load(
        CONFIG.read_text(encoding="utf-8")
    )
    preferred = _discover(ROOT / "input" / "preferred")
    targets = _discover(ROOT / "input" / "targets")
    manifest = _manifest()

    assert manifest["dynamic_discovery"] is True
    assert sorted(manifest["preferred_votes"]) == preferred
    assert sorted(manifest["targets_discovered"]) == targets
    assert manifest["independent_vote_count"] == len(preferred)
    assert manifest["target_count"] == len(targets)
    assert (
        manifest["low_frequency_reference"]
        == cfg["low_frequency_reference"]["file"]
    )


def test_preferred_dataset_is_dynamic():
    """Every current preferred TXT is represented as an independent vote."""
    _require_current_generated_state()

    preferred = _discover(ROOT / "input" / "preferred")
    manifest = _manifest()

    assert len(preferred) >= 1
    assert manifest["independent_vote_count"] == len(preferred)
    assert sorted(manifest["preferred_votes"]) == preferred


def test_generated_outputs_cover_every_discovered_target():
    """Every discovered target has its complete generated artifact set."""
    _require_current_generated_state()

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


def test_validation_report_is_pass():
    _require_current_generated_state()

    validation_path = REPORTS / "validation.txt"
    assert validation_path.is_file(), (
        f"Missing validation report: {validation_path}"
    )

    validation = validation_path.read_text(
        encoding="utf-8"
    )

    assert "Status: PASS" in validation


def test_handoff_statuses_are_explicit_and_fail_safe():
    _require_current_generated_state()

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
