#!/usr/bin/env python3
"""
IEM EarPrint Engine
Deterministic implementation of the General_Prompt_EarPrint_8 workflow.

DYNAMIC DATA DISCOVERY:
- Every *.txt in input/preferred is one independent IEM vote.
- Every *.txt in input/targets is one independent target.
- Adding/replacing a preferred file or target file automatically changes the build.
- No target list or IEM list is hard-coded in Python.

The only declared semantic choice is the LF reference target in project.yaml,
because Pure EarPrint requires a defined low-frequency reference.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "project.yaml"
PREFERRED_DIR = ROOT / "input" / "preferred"
TARGET_DIR = ROOT / "input" / "targets"
OUT = ROOT / "output"
REPORTS = ROOT / "reports"


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def read_config():
    with CFG.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_txt(directory: Path, glob_pattern: str, ignore_files):
    ignored = set(ignore_files or [])
    files = sorted(
        p for p in directory.glob(glob_pattern)
        if p.is_file() and p.name not in ignored
    )
    if not files:
        fail(f"No input files discovered in {directory} using {glob_pattern!r}")
    return files


def parse_xy(path: Path):
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
        s = raw.strip()
        if not s or s.startswith("#") or s.startswith("//") or s.startswith(";"):
            continue
        parts = re.split(r"[\t,; ]+", s)
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            fail(f"{path.name}: non-finite value at line {line_no}")
        rows.append((x, y))
    if len(rows) < 10:
        fail(f"{path.name}: fewer than 10 numeric data rows found")
    arr = np.asarray(rows, dtype=float)
    if np.any(arr[:, 0] <= 0):
        fail(f"{path.name}: frequency must be > 0")
    if np.any(np.diff(arr[:, 0]) <= 0):
        fail(f"{path.name}: frequencies must be strictly ascending")
    return arr[:, 0], arr[:, 1]


def interp_log(freq_src, level_src, freq_dst):
    if freq_dst.min() < freq_src.min() or freq_dst.max() > freq_src.max():
        fail(
            f"Grid coverage error: source [{freq_src.min():.3f}, {freq_src.max():.3f}] "
            f"does not cover requested [{freq_dst.min():.3f}, {freq_dst.max():.3f}] Hz"
        )
    return np.interp(np.log(freq_dst), np.log(freq_src), level_src)


def median_in_band(freq, values, lo, hi):
    m = (freq >= lo) & (freq <= hi)
    if not np.any(m):
        fail(f"No samples in alignment band {lo}-{hi} Hz")
    return float(np.median(values[m]))


def align_curve(freq, curve, ref_curve, bands):
    """Three alignment scenarios; returns median-offset curve and uncertainty."""
    scenario_offsets = []
    d = curve - ref_curve
    for lo, hi in bands:
        scenario_offsets.append(median_in_band(freq, d, lo, hi))
    offsets = np.asarray(scenario_offsets, dtype=float)
    centre = float(np.median(offsets))
    uncertainty = float(np.max(np.abs(offsets - centre)))
    return curve - centre, centre, uncertainty, offsets


def gaussian_masked(values, freq, start=1000.0, end=12000.0, sigma=16, radius=64):
    """Exactly one Gaussian pass on the strict personal domain."""
    out = values.copy()
    m = (freq > start) & (freq < end)
    idx = np.where(m)[0]
    if len(idx) == 0:
        return out
    a, b = idx[0], idx[-1]
    out[a:b+1] = gaussian_filter1d(
        out[a:b+1], sigma=sigma, radius=radius, mode="nearest"
    )
    return out


def quarter_octave_taper(freq, start=1000.0, end=12000.0):
    """Fixed quarter-octave sin² boundary tapers."""
    w = np.ones_like(freq, dtype=float)
    q = 2 ** 0.25
    low_end = start * q
    high_start = end / q

    low = (freq > start) & (freq < low_end)
    if np.any(low):
        x = np.log2(freq[low] / start) / 0.25
        w[low] = np.sin(0.5 * np.pi * x) ** 2

    high = (freq > high_start) & (freq < end)
    if np.any(high):
        x = np.log2(freq[high] / high_start) / 0.25
        w[high] = np.cos(0.5 * np.pi * x) ** 2

    w[freq <= start] = 0.0
    w[freq >= end] = 0.0
    return w


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE)
    stem = re.sub(r"_+", "_", stem).strip("_.")
    return stem or "target"


def write_xy(path, freq, level, decimals=12):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for x, y in zip(freq, level):
            f.write(f"{x:.12g}\t{y:.{decimals}f}\n")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    cfg = read_config()
    OUT.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    # Clean generated TXT files/reports so removed targets cannot leave stale output.
    for p in OUT.glob("*.txt"):
        p.unlink()
    for p in REPORTS.iterdir():
        if p.is_file():
            p.unlink()

    discovery = cfg["discovery"]
    preferred_files = discover_txt(
        PREFERRED_DIR,
        discovery.get("preferred_glob", "*.txt"),
        discovery.get("ignore_files", []),
    )
    target_files = discover_txt(
        TARGET_DIR,
        discovery.get("target_glob", "*.txt"),
        discovery.get("ignore_files", []),
    )

    bands = [tuple(map(float, x)) for x in cfg["alignment"]["bands_hz"]]
    pstart = float(cfg["frequency"]["personal_start_hz"])
    pend = float(cfg["frequency"]["personal_end_hz"])
    out_start = float(cfg["frequency"]["output_start_hz"])
    out_end = float(cfg["frequency"]["output_end_hz"])
    sigma = int(cfg["smoothing"]["sigma_indices"])
    radius = int(cfg["smoothing"]["radius_indices"])
    decimals = int(cfg["output"]["decimals"])

    preferred = {p.name: parse_xy(p) for p in preferred_files}
    targets = {p.name: parse_xy(p) for p in target_files}

    ref_file = cfg["low_frequency_reference"]["file"]
    if ref_file not in targets:
        fail(
            f"LF reference {ref_file!r} was not discovered in input/targets. "
            "Set low_frequency_reference.file to one of the target filenames."
        )

    master_freq, master_ref = targets[ref_file]

    if master_freq[0] > out_start:
        fail("LF reference target does not cover the configured output start frequency.")
    if master_freq[-1] < min(pend, out_end):
        fail("LF reference target does not cover the required personal/output domain.")

    # Align every preferred curve to the declared LF reference.
    aligned_to_lf = {}
    alignment_report = []
    for path in preferred_files:
        f, y = preferred[path.name]
        yy = interp_log(f, y, master_freq)
        aligned_curve, off, unc, scenarios = align_curve(
            master_freq, yy, master_ref, bands
        )
        aligned_to_lf[path.name] = aligned_curve
        alignment_report.append({
            "iem": path.name,
            "median_offset_db": off,
            "alignment_uncertainty_db": unc,
            "offset_200_1000_db": float(scenarios[0]),
            "offset_200_500_db": float(scenarios[1]),
            "offset_500_1000_db": float(scenarios[2]),
        })

    stack = np.vstack([aligned_to_lf[p.name] for p in preferred_files])
    direct_personal = np.median(stack, axis=0)

    # Pure EarPrint.
    smooth_personal = gaussian_masked(
        direct_personal, master_freq, pstart, pend, sigma=sigma, radius=radius
    )
    idx1 = int(np.argmin(np.abs(master_freq - pstart)))
    join_shift = float(master_ref[idx1] - smooth_personal[idx1])
    personal = smooth_personal + join_shift

    pure = master_ref.copy()
    m_personal = (master_freq > pstart) & (master_freq < pend)
    pure[m_personal] = personal[m_personal]

    # Display-only -6 dB/octave extension.
    ext = master_freq >= pend
    last_idx = np.where(master_freq < pend)[0][-1]
    anchor_f = float(master_freq[last_idx])
    anchor_y = float(pure[last_idx])
    slope = float(cfg["high_frequency_extension"]["slope_db_per_octave"])
    pure[ext] = anchor_y + slope * np.log2(master_freq[ext] / anchor_f)

    out_m = (master_freq >= out_start) & (master_freq <= out_end)
    of = master_freq[out_m]

    if cfg["output"]["write_pure_earprint"]:
        write_xy(OUT / "pure_earprint_dynamic.txt", of, pure[out_m], decimals)

    robust_info = []

    # EVERY discovered target is independently processed.
    for target_path in target_files:
        target_name = target_path.name
        stem = safe_stem(target_name)
        tf, tv = targets[target_name]
        base = interp_log(tf, tv, master_freq)

        centres = []
        per_iem_unc = []

        for iem_path in preferred_files:
            af = aligned_to_lf[iem_path.name]
            delta0 = af - base
            scenario_offsets = np.asarray(
                [median_in_band(master_freq, delta0, lo, hi) for lo, hi in bands],
                dtype=float,
            )
            c = float(np.median(scenario_offsets))
            u = float(np.max(np.abs(scenario_offsets - c)))
            centres.append(af - c)
            per_iem_unc.append(u)

        delta_stack = np.vstack(
            [centres[i] - base for i in range(len(preferred_files))]
        )
        centre = np.median(delta_stack, axis=0)
        cross_mad = np.median(np.abs(delta_stack - centre), axis=0)
        align_unc = float(np.median(per_iem_unc))
        total_unc = np.maximum(cross_mad, align_unc)

        retention = np.zeros_like(centre)
        denom = np.abs(centre) + total_unc
        nz = denom > 0
        retention[nz] = np.abs(centre[nz]) / denom[nz]
        raw_mask = centre * retention

        mask = gaussian_masked(raw_mask, master_freq, pstart, pend, sigma, radius)
        mask *= quarter_octave_taper(master_freq, pstart, pend)
        mask[(master_freq <= pstart) | (master_freq >= pend)] = 0.0

        robust = base + mask
        outside = (master_freq <= pstart) | (master_freq >= pend)
        robust[outside] = base[outside]

        if cfg["output"]["write_masks"]:
            write_xy(OUT / f"{stem}__mask.txt", of, mask[out_m], decimals)
        if cfg["output"]["write_robust_targets"]:
            write_xy(OUT / f"{stem}__robust_target.txt", of, robust[out_m], decimals)

        if cfg.get("hybrids", {}).get("enabled", True) and cfg["output"]["write_hybrids"]:
            hybrid = base.copy()
            hybrid[master_freq > pstart] = pure[master_freq > pstart]
            write_xy(OUT / f"{stem}__plus_earprint.txt", of, hybrid[out_m], decimals)

        robust_info.append({
            "target": target_name,
            "cross_iem_mad_median_db": float(np.median(cross_mad)),
            "alignment_uncertainty_median_db": align_unc,
            "mask_abs_max_db": float(np.max(np.abs(mask))),
            "mask_rms_db": float(np.sqrt(np.mean(mask**2))),
        })

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "IEM EarPrint Engine 1.1.0",
        "dynamic_discovery": True,
        "preferred_glob": discovery.get("preferred_glob", "*.txt"),
        "target_glob": discovery.get("target_glob", "*.txt"),
        "preferred_votes": [p.name for p in preferred_files],
        "independent_vote_count": len(preferred_files),
        "targets_discovered": [p.name for p in target_files],
        "target_count": len(target_files),
        "low_frequency_reference": ref_file,
        "alignment_bands_hz": bands,
        "personal_domain_hz": [pstart, pend],
        "smoothing": {
            "method": "gaussian",
            "sigma_indices": sigma,
            "radius_indices": radius,
            "padding": "nearest",
            "passes": 1,
        },
        "hf_extension": {
            "method": "log_frequency",
            "slope_db_per_octave": slope,
            "display_only": True,
        },
        "grid_points": int(len(master_freq)),
        "output_points": int(len(of)),
        "sha256": {},
    }

    for p in sorted(OUT.glob("*.txt")):
        manifest["sha256"][p.name] = sha256(p)

    (REPORTS / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    with (REPORTS / "alignment_offsets.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(alignment_report[0].keys()))
        w.writeheader()
        w.writerows(alignment_report)

    with (REPORTS / "robust_statistics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(robust_info[0].keys()))
        w.writeheader()
        w.writerows(robust_info)

    prov = [
        "IEM EarPrint Engine validation report",
        f"Dynamic preferred discovery: {discovery.get('preferred_glob', '*.txt')}",
        f"Dynamic target discovery: {discovery.get('target_glob', '*.txt')}",
        f"Independent preferred-response votes discovered: {len(preferred_files)}",
        f"Targets discovered: {len(target_files)}",
        "Vote rule: one IEM file = one vote.",
        "Preferred-response files are consumed directly; PEQ gains are not averaged.",
        f"Master grid: {ref_file}",
        "Alignment scenarios: [200,1000], [200,500], [500,1000] Hz; median used.",
        "Pure EarPrint: LF reference <=1 kHz; one Gaussian pass in strict >1 to <12 kHz; -6 dB/octave display extension >=12 kHz.",
        "Robust masks: independently computed for every discovered target; no universal mask.",
        "Target processing: dynamic; no target filename list is embedded in Python.",
        "Validation status: PASS",
    ]
    (REPORTS / "validation.txt").write_text("\n".join(prov) + "\n", encoding="utf-8")

    print("BUILD PASS")
    print(f"Dynamic preferred votes discovered: {len(preferred_files)}")
    print(f"Dynamic targets discovered: {len(target_files)}")
    print(f"Master grid points: {len(master_freq)}")
    print(f"Outputs: {len(list(OUT.glob('*.txt')))} TXT files")


if __name__ == "__main__":
    main()
