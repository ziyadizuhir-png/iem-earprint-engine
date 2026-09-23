"""Regression checks for generated EarPrint artifacts.

These checks intentionally validate structure and locked invariants rather than
bit-for-bit timestamps, so normal deterministic rebuilds remain portable.
"""

import json
from pathlib import Path

import yaml

from engine.input_hash import calculate_hash, source_files


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config" / "project.yaml"


def _manifest() -> dict:
    return json.loads((REPORTS / "manifest.json").read_text(encoding="utf-8"))


def test_generated_manifest_matches_discovered_inputs():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    preferred = sorted(
        p.name for p in (ROOT / "input" / "preferred").glob("*.txt")
    )
    targets = sorted(
        p.name for p in (ROOT / "input" / "targets").glob("*.txt")
    )
    manifest = _manifest()

    assert manifest["dynamic_discovery"] is True
    assert manifest["preferred_votes"] == preferred
    assert manifest["targets_discovered"] == targets
    assert manifest["independent_vote_count"] == len(preferred)
    assert manifest["target_count"] == len(targets)
    assert manifest["low_frequency_reference"] == cfg["low_frequency_reference"]["file"]


def test_generated_outputs_cover_every_target():
    targets = list((ROOT / "input" / "targets").glob("*.txt"))
    robust = list(OUTPUT.glob("*__robust_target.txt"))
    masks = list(OUTPUT.glob("*__mask.txt"))
    handoffs = list(REPORTS.glob("*__adaptive_handoff.txt"))

    assert (OUTPUT / "pure_earprint_dynamic.txt").is_file()
    assert len(robust) == len(targets)
    assert len(masks) == len(targets)
    assert len(handoffs) == len(targets)


def test_validation_report_is_pass():
    validation = (REPORTS / "validation.txt").read_text(encoding="utf-8")
    assert "Status: PASS" in validation
    assert "NO_STABLE_HANDOFF" not in validation


def test_build_input_hash_is_deterministic_and_nonempty():
    first = calculate_hash()
    second = calculate_hash()
    assert len(source_files()) > 0
    assert len(first) == 64
    assert first == second
