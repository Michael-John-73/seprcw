"""Verify the released file inventory and recount the H-Q result without images."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

from hq_cpu_statistics import paired_values, stratified_interval, quality_summary, hq_verdict

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_inventory(root=ROOT):
    inventory = load(root / "data" / "release_manifest.json")
    for relative, expected in tqdm(inventory["files"].items(), desc="Verify release SHA-256"):
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError("Missing or unsafe release file: " + relative)
        if digest(path) != expected:
            raise ValueError("Release checksum mismatch: " + relative)
    return len(inventory["files"])


def recount(root=ROOT):
    base = root / "data" / "hq"
    cohort, rows, qualities, expected = (
        load(base / name) for name in ("cohort.json", "rows.json", "quality.json", "analysis.json")
    )
    if len(cohort) != 848 or len(rows) != 3392 or len(qualities) != 1696:
        raise ValueError("Unexpected H-Q panel size")
    assignments = load(base / "messages.json")["assignments"]
    for row in tqdm(rows, desc="Recount 48-bit ER/BER"):
        reference = np.unpackbits(np.frombuffer(bytes.fromhex(row["message"]), dtype=np.uint8))
        predicted = np.asarray(row["predicted_bits"])
        if (predicted.shape != (48,) or not np.isin(predicted, [0, 1]).all()
                or assignments[row["source_id"]] != row["message"]):
            raise ValueError("Invalid message or predicted bits")
        exact = int(row["coverage_valid"] and np.array_equal(predicted, reference))
        if exact != row["ER"] or not np.isclose(np.mean(predicted != reference), row["BER"], atol=1e-14, rtol=0):
            raise ValueError("Stored ER/BER disagrees with bits")
    values = paired_values(rows, cohort)
    effect = stratified_interval(values, reps=10000, seed=20261026, chunk=256)
    quality = quality_summary(qualities, cohort)
    for key in ("difference", "low", "high"):
        if not np.isclose(effect[key], expected["effect"][key], atol=1e-14, rtol=0):
            raise ValueError("H-Q effect/interval mismatch: " + key)
    for key in ("E1_psnr_mean", "E2_psnr_mean", "E1_ssim_mean", "E2_ssim_mean"):
        if not np.isclose(quality[key], expected["quality"][key], atol=1e-12, rtol=0):
            raise ValueError("H-Q mean quality mismatch: " + key)
    verdict = hq_verdict(effect, quality["matched"])
    if verdict != expected["verdict"]:
        raise ValueError("H-Q verdict mismatch")
    original_s = load(root / "data" / "protocol_s" / "final.json")
    if [original_s["hypotheses"][key]["status"] for key in ("H1", "H2", "H3")] != [
        "PASS", "PASS", "INCONCLUSIVE"
    ]:
        raise ValueError("Original S decisions changed")
    original_e = load(root / "data" / "protocol_e" / "test_analysis.json")
    if any(value != "WITHHELD" for value in original_e["hypotheses"].values()):
        raise ValueError("Original E decisions changed")
    return {"sources": len(values), "effect": effect, "quality": quality, "verdict": verdict}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "verification.json")
    args = parser.parse_args()
    print("[START] Verify released data; no images, training, or new outcome selection", flush=True)
    count = verify_inventory()
    result = recount()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"verified_files": count, **result}, indent=2), encoding="utf-8")
    print(json.dumps({"verified_files": count, "verdict": result["verdict"],
                      "effect": result["effect"]}, indent=2), flush=True)
    print("[END] Release integrity and H-Q recount passed", flush=True)


if __name__ == "__main__":
    main()
