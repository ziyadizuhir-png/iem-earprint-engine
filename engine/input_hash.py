"""Print a deterministic hash for all source files that affect EarPrint output."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INCLUDED = (
    "input",
    "config",
    "engine",
    "ARCHITECTURE_LOCK.md",
    "General_Prompt_EarPrint_8_MATH_LOCKED_boundary_addendum.txt",
    "requirements.txt",
)


def source_files() -> list[Path]:
    files: list[Path] = []
    for item in INCLUDED:
        path = ROOT / item
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file())
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def calculate_hash() -> str:
    digest = hashlib.sha256()
    for path in source_files():
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


if __name__ == "__main__":
    print(calculate_hash())
