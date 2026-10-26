"""Acquire exact source images, prepare them, and rerun fixed H-Q blind decoding."""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import urllib.request
import zipfile

import cv2
import numpy as np
from tqdm import tqdm

from hq_independent_cpu import prepare_pixels, pixel_hash, task_worker
from verify_release import ROOT, digest, load, verify_inventory

COCO_URL = "https://images.cocodataset.org/zips/val2017.zip"


def download(url, target, expected=None):
    if target.exists():
        if expected is not None and digest(target) != expected:
            raise ValueError("Cached archive checksum mismatch: " + str(target))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
        length = response.headers.get("Content-Length")
        with tqdm(total=int(length) if length else None, unit="B", unit_scale=True,
                  desc=target.name) as progress:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(block)
                progress.update(len(block))
    if expected is not None and digest(partial) != expected:
        raise ValueError("Downloaded archive checksum mismatch: " + str(partial))
    partial.replace(target)


def prepare(dataset_root, raw_dirs, allow_download):
    cohort = load(ROOT / "data" / "hq" / "cohort.json")
    population = load(ROOT / "data" / "hq" / "design.json")["population"]
    archives = {}
    if allow_download:
        for domain, url, checksum in (
            ("real", COCO_URL, None),
            ("ai", population["archive_url"], population["archive_sha256"]),
        ):
            archive = dataset_root / "archives" / (domain + ".zip")
            download(url, archive, checksum)
            archives[domain] = archive
    prepared = dataset_root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    for domain in ("real", "ai"):
        selected = [row for row in cohort if row["domain"] == domain]
        if domain not in raw_dirs and domain not in archives:
            raise ValueError("Supply both raw directories or opt in with --download")
        names = {}
        if domain in raw_dirs:
            for path in raw_dirs[domain].rglob("*"):
                if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    if path.name in names:
                        raise ValueError("Ambiguous raw image basename: " + path.name)
                    names[path.name] = path
        archive = zipfile.ZipFile(archives[domain]) if domain in archives else None
        try:
            members = {}
            if archive is not None:
                for name in archive.namelist():
                    if not name.endswith("/"):
                        basename = Path(name).name
                        if basename in members:
                            raise ValueError("Ambiguous archive basename: " + basename)
                        members[basename] = name
            for row in tqdm(selected, desc="Prepare and hash " + domain):
                filename = row["original_source_file"]
                if domain in raw_dirs:
                    if filename not in names:
                        raise ValueError("Missing raw source: " + filename)
                    raw = names[filename]
                else:
                    if filename not in members:
                        raise ValueError("Missing archive source: " + filename)
                    raw = dataset_root / "selected_raw" / domain / filename
                    raw.parent.mkdir(parents=True, exist_ok=True)
                    raw.write_bytes(archive.read(members[filename]))
                if digest(raw) != row["raw_sha256"]:
                    raise ValueError("Raw image hash mismatch: " + filename)
                image = prepare_pixels(raw)
                if pixel_hash(image) != row["pixel_sha256"]:
                    raise ValueError("Prepared pixel hash mismatch: " + filename)
                target = prepared / (row["source_id"] + ".png")
                if not cv2.imwrite(str(target), image):
                    raise OSError("Cannot save prepared image: " + str(target))
                if digest(target) != row["sha256"]:
                    raise ValueError("Prepared PNG hash differs; check pinned OpenCV version")
        finally:
            if archive is not None:
                archive.close()
    return cohort


def run(dataset_root, output, workers, limit):
    if output.exists():
        raise FileExistsError("Use a new output directory; previous results are never overwritten")
    cohort = load(ROOT / "data" / "hq" / "cohort.json")
    messages = load(ROOT / "data" / "hq" / "messages.json")["assignments"]
    if limit is not None:
        cohort = cohort[:limit]
    tasks = []
    for item in cohort:
        row = {**item, "path": str((dataset_root / "prepared" / (item["source_id"] + ".png")).resolve())}
        if not Path(row["path"]).is_file() or digest(row["path"]) != row["sha256"]:
            raise ValueError("Missing or altered prepared source: " + item["source_id"])
        tasks.append((str(ROOT), str(output.resolve()), row, messages[row["source_id"]]))
    (output / "marked").mkdir(parents=True)
    rows, qualities = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rr, qq, _ in tqdm(pool.map(task_worker, tasks), total=len(tasks), desc="CPU blind H-Q"):
            rows.extend(rr)
            qualities.extend(qq)
    expected = {(r["source_id"], r["rule"], r["area"]): r
                for r in load(ROOT / "data" / "hq" / "rows.json")}
    expected_q = {(r["source_id"], r["rule"]): r
                  for r in load(ROOT / "data" / "hq" / "quality.json")}
    for row in rows:
        reference = expected[row["source_id"], row["rule"], row["area"]]
        for key in ("predicted_bits", "phase", "ER", "BER", "coverage_valid",
                    "message", "top", "left", "height", "width"):
            if row[key] != reference[key]:
                raise ValueError("Reproduced result mismatch: " + str((row["source_id"], key)))
    for row in qualities:
        reference = expected_q[row["source_id"], row["rule"]]
        if row["marked_sha256"] != reference["marked_sha256"]:
            raise ValueError("Marked PNG differs from released run")
        for key in ("psnr", "ssim"):
            if not np.isclose(row[key], reference[key], rtol=0, atol=1e-10):
                raise ValueError("Quality mismatch: " + key)
    for name, value in (("rows.json", rows), ("quality.json", qualities)):
        (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    result = {"status": "PASS", "sources": len(cohort), "rows": len(rows),
              "scope": "full_fixed_cohort" if limit is None else "diagnostic_subset",
              "blind_bits_phase_quality_match": True}
    (output / "reproduction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets" / "hq")
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--ai-dir", type=Path)
    parser.add_argument("--download", action="store_true", help="Explicitly download official archives")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "hq_reproduction")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--limit", type=int, help="Diagnostic subset only, not full-cohort confirmation")
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 848:
        parser.error("--limit must be 1..848")
    print("[START] CPU-only fixed-cohort reproduction: " + args.command, flush=True)
    verify_inventory()
    if args.command == "prepare":
        raw_dirs = {key: path for key, path in (("real", args.real_dir), ("ai", args.ai_dir)) if path is not None}
        prepare(args.dataset_root, raw_dirs, args.download)
    else:
        run(args.dataset_root, args.output, args.workers, args.limit)
    print("[END] H-Q " + args.command + " passed", flush=True)


if __name__ == "__main__":
    main()
