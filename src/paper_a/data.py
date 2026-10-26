"""Source/message splits and per-case disk streaming beneath /workspace."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity
from .io import (
    atomic_json,
    read_json,
    read_csv,
    write_csv,
    sha,
    stable_seed,
    root,
    log,
    evidence_record,
    code_digest,
    require_stage,
)
from .codec import CodecConfig, embed, png_roundtrip, rank_change, collect

ROLES = (
    "train",
    "validation",
    "detector_fit",
    "calibration",
    "test",
    "external_test",
    "reserve",
)
REQUIRED = {
    "source_id",
    "source_group",
    "path",
    "dataset",
    "domain",
    "license",
    "previously_seen",
    "split",
}


def codec_config(cfg):
    return CodecConfig(
        **{k: v for k, v in cfg["codec"].items() if k != "eta_candidates"}
    )


def processed_image(path, cfg):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image {path}")
    h, w = image.shape[:2]
    scale = min(1.0, cfg["data"]["max_side"] / max(h, w))
    if scale < 1:
        image = cv2.resize(
            image,
            (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    if min(image.shape[:2]) < cfg["data"]["min_side"]:
        raise ValueError(f"image below registered minimum after preprocessing: {path}")
    return image


def dhash(image):
    tiny = cv2.resize(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA
    )
    return int("".join(map(str, (tiny[:, 1:] > tiny[:, :-1]).ravel().astype(int))), 2)


def prepare(cfg):
    rr = root(cfg)
    if (rr / "freeze.json").exists() or (rr / "design_lock.json").exists():
        raise RuntimeError("cannot replace dataset after design lock/freeze; use a new run_root")
    rows = read_csv(cfg["paths"]["input_csv"])
    if not rows or not REQUIRED.issubset(rows[0]):
        raise ValueError(f"input CSV requires {sorted(REQUIRED)}")
    if cfg["study"]["mode"] != "diagnostic":
        internal_datasets = {r["dataset"].strip() for r in rows if r["split"] not in ("external_test", "reserve")}
        external_datasets = {r["dataset"].strip() for r in rows if r["split"] == "external_test"}
        if internal_datasets & external_datasets:
            raise ValueError("external_test dataset must be distinct from development/internal data")
    ids, groups, hashes, pixelhashes = set(), {}, {}, {}
    near_pairs, candidates = [], []
    distance = cfg["data"]["near_duplicate_distance"]
    # Multi-index Hamming lookup: with d+1 bands, <=d differing bits share one band.
    bands = np.array_split(np.arange(64), distance + 1)
    buckets = {}
    for row in tqdm(rows, desc="audit sources", file=__import__("sys").stdout):
        if row["source_id"] in ids or row["split"] not in ROLES:
            raise ValueError("duplicate source_id or unsupported split")
        ids.add(row["source_id"])
        if not all(row[k].strip() for k in REQUIRED):
            raise ValueError("blank required metadata")
        group = row["source_group"]
        if group in groups:
            raise ValueError("one independent representative per source_group required")
        groups[group] = row["split"]
        if row["previously_seen"].lower() not in ("true", "false"):
            raise ValueError("previously_seen must be true or false")
        if row["previously_seen"].lower() == "true" and row["split"] not in (
            "train",
            "validation",
        ):
            raise ValueError("previously explored source used in independent role")
        p = Path(row["path"])
        if not p.is_absolute():
            p = Path(cfg["paths"]["dataset_root"]) / p
        p = p.resolve()
        if (cfg["study"]["mode"] != "diagnostic"
                and not p.is_relative_to(Path(cfg["paths"]["dataset_root"]).resolve())):
            raise ValueError("production image must remain beneath dataset_root in /workspace")
        if not p.is_file():
            raise FileNotFoundError(p)
        digest = sha(p)
        image = processed_image(p, cfg)
        pixel_digest = hashlib.sha256(
            image.tobytes() + str(image.shape).encode()
        ).hexdigest()
        if digest in hashes or pixel_digest in pixelhashes:
            raise ValueError(
                "duplicate file/pixels: independently resplit copies are forbidden"
            )
        hashes[digest], pixelhashes[pixel_digest] = row["source_id"], row["source_id"]
        dh = dhash(image)
        seen = set()
        for b, bitidx in enumerate(bands):
            start, length = int(bitidx[0]), len(bitidx)
            key = (b, (dh >> start) & ((1 << length) - 1))
            for previous in buckets.get(key, []):
                if previous not in seen:
                    seen.add(previous)
                    old = candidates[previous]
                    hd = (old["dhash"] ^ dh).bit_count()
                    if hd <= distance:
                        near_pairs.append(
                            {
                                "source_a": old["source_id"],
                                "source_b": row["source_id"],
                                "split_a": old["split"],
                                "split_b": row["split"],
                                "distance": hd,
                            }
                        )
            buckets.setdefault(key, []).append(len(candidates))
        row.update(
            path=str(p),
            sha256=digest,
            pixel_sha256=pixel_digest,
            width=image.shape[1],
            height=image.shape[0],
            dhash=dh,
        )
        candidates.append(row)
    # Input-only image access is allowed for duplicate auditing, never outcome/model selection.
    write_csv(
        rr / "near_duplicate_candidates.csv",
        near_pairs,
        ["source_a", "source_b", "split_a", "split_b", "distance"],
    )
    if near_pairs:
        raise RuntimeError(
            "near duplicates need source-group resolution and a fresh intake CSV; no auto PASS"
        )
    write_csv(rr / "sources.csv", candidates)
    messages = {}
    all_messages = set()
    for role in ROLES:
        pool = []
        rng = np.random.default_rng(stable_seed(cfg["study"]["seed"], "messages", role))
        while len(pool) < cfg["data"]["message_pool_size"]:
            value = int(rng.integers(0, 1 << 48))
            if value != int("B3EC907BB19E", 16) and value not in all_messages:
                all_messages.add(value)
                pool.append(f"{value:012X}")
        messages[role] = pool
    atomic_json(rr / "messages.json", messages)
    counts = {r: sum(x["split"] == r for x in candidates) for r in ROLES}
    atomic_json(
        rr / "data_audit.json",
        {
            "counts": counts,
            "unique_groups": len(groups),
            "no_cross_role_duplicates": True,
            "source_csv_hash": sha(rr / "sources.csv"),
            "message_hash": sha(rr / "messages.json"),
            "near_duplicate_distance": distance,
            "note": "dHash is a screening heuristic; content provenance remains required",
        },
    )
    from .io import record_data_access

    for role in ROLES:
        if counts[role]:
            record_data_access(
                cfg,
                role,
                "INPUT_ONLY_DUPLICATE_AND_ROLE_AUDIT",
                rr / "sources.csv",
                outcomes_seen=False,
            )
    return evidence_record(
        cfg,
        "prepare",
        [rr / "sources.csv", rr / "messages.json", rr / "data_audit.json"],
    )


def sources(cfg, role):
    if role not in ROLES:
        raise ValueError("role")
    rr = root(cfg)
    audit = read_json(rr / "data_audit.json")
    if (
        sha(rr / "sources.csv") != audit["source_csv_hash"]
        or sha(rr / "messages.json") != audit["message_hash"]
    ):
        raise RuntimeError("source/message manifest changed")
    return [x for x in read_csv(rr / "sources.csv") if x["split"] == role]


def message_for(cfg, row):
    pool = read_json(root(cfg) / "messages.json")[row["split"]]
    word = pool[stable_seed(row["source_id"], "assignment") % len(pool)]
    return np.array([int(b) for b in f"{int(word,16):048b}"], np.uint8), word


def crop_panel(image, cfg, source_id):
    h, w = image.shape[:2]
    for area in cfg["evaluation"]["areas"]:
        if area == 1:
            top, left, ch, cw = 0, 0, h, w
        else:
            ch, cw = max(8, round(h * np.sqrt(area))), max(8, round(w * np.sqrt(area)))
            rng = np.random.default_rng(
                stable_seed(cfg["study"]["seed"], source_id, area, "crop")
            )
            top, left = int(rng.integers(h - ch + 1)), int(rng.integers(w - cw + 1))
        yield {
            "condition": f"area_{area:g}",
            "area": area,
            "top": top,
            "left": left,
            "height": ch,
            "width": cw,
        }, image[top : top + ch, left : left + cw]


def quality(original, marked):
    from .compute import cuda_enabled
    if cuda_enabled():
        from .cuda_codec import quality as cuda_quality
        return cuda_quality(original, marked)
    mse = float(np.mean((original.astype(np.float64) - marked.astype(np.float64)) ** 2))
    return {
        "psnr": 120.0 if mse == 0 else float(10 * np.log10(255**2 / mse)),
        "psnr_infinite": mse == 0,
        "ssim": float(
            structural_similarity(original, marked, channel_axis=2, data_range=255)
        ),
    }


def bank_dir(cfg, role, eta):
    return root(cfg) / "banks" / role / f"eta_{eta:g}"


def save_bank(path, bank, bits, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in bank.items() if isinstance(v, np.ndarray)}
    payload.update(bits=bits, meta=np.array(json.dumps(meta, allow_nan=False)))
    tmp = path.with_suffix(".partial.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)


def load_bank(path):
    with np.load(path, allow_pickle=False) as z:
        b = {
            k: z[k].copy()
            for k in ("features", "soft", "hard", "residue", "index", "counts", "bits")
        }
        b["meta"] = json.loads(str(z["meta"]))
    b.update(
        {
            k: b["meta"][k]
            for k in (
                "pilot_score",
                "phase_margin",
                "offset",
                "search_s",
                "collect_s",
                "pilot_observations",
            )
        }
    )
    return b


def make_bank(cfg, role, eta, include_null=False, allow_final=False):
    if role in ("test", "external_test") and not allow_final:
        raise RuntimeError(
            "final input path requires evaluation lock, not cache command"
        )
    rr = root(cfg)
    require_stage(cfg, "prepare")
    require_stage(cfg, "math-check")
    if role == "detector_fit" and cfg["study"]["mode"] != "diagnostic":
        from .experiment import verify_design

        verify_design(cfg)
    if role == "calibration" and not (rr / "freeze.json").exists():
        raise RuntimeError("calibration requires frozen protocol")
    out = bank_dir(cfg, role, eta)
    if (out / "manifest.json").exists():
        raise RuntimeError(f"bank exists; validate or use new run_root: {out}")
    all_rows = []
    cc = codec_config(cfg)
    input_rows = sources(cfg, role)
    if not input_rows:
        raise RuntimeError(f"no sources in {role}")
    for row in tqdm(
        input_rows, desc=f"{role} eta={eta:g}", file=__import__("sys").stdout
    ):
        if sha(row["path"]) != row["sha256"]:
            raise RuntimeError("source bytes changed")
        image = processed_image(row["path"], cfg)
        bits, word = message_for(cfg, row)
        for rule in ("E0", "E1", "E2"):
            start = __import__("time").perf_counter()
            marked, pngbytes = png_roundtrip(embed(image, bits, cc, rule, eta))
            embedding_s = __import__("time").perf_counter() - start
            q = quality(image, marked)
            rank = rank_change(image, marked, cc)
            for info, crop in crop_panel(marked, cfg, row["source_id"]):
                modes = (
                    ("search", "tile_oracle", "fixed") if rule == "E2" else ("search",)
                )
                for mode in modes:
                    bank = collect(
                        crop,
                        cc,
                        mode,
                        origin=(
                            (info["top"], info["left"])
                            if mode == "tile_oracle"
                            else None
                        ),
                    )
                    case = {
                        **row,
                        **info,
                        **q,
                        "message": word,
                        "rule": rule,
                        "mode": mode,
                        "label": 1,
                        "eta": eta,
                        "rank_change": rank,
                        "embedding_s": embedding_s,
                        "marked_sha256": hashlib.sha256(pngbytes).hexdigest(),
                        **{
                            k: bank[k]
                            for k in (
                                "pilot_score",
                                "phase_margin",
                                "offset",
                                "search_s",
                                "collect_s",
                                "pilot_observations",
                            )
                        },
                    }
                    cid = hashlib.sha256(
                        f"{row['source_id']}|{rule}|{info['condition']}|{mode}|1".encode()
                    ).hexdigest()[:24]
                    p = out / (cid + ".npz")
                    save_bank(p, bank, bits, case)
                    all_rows.append(
                        {
                            "path": str(p),
                            "sha256": sha(p),
                            "source_id": row["source_id"],
                            "rule": rule,
                            "mode": mode,
                            "label": 1,
                            "condition": info["condition"],
                        }
                    )
        if include_null:
            for info, crop in crop_panel(image, cfg, row["source_id"]):
                bank = collect(crop, cc)
                case = {
                    **row,
                    **info,
                    "message": word,
                    "rule": "null",
                    "mode": "search",
                    "label": 0,
                    "eta": eta,
                    "rank_change": 0.0,
                    "psnr": 120.0,
                    "ssim": 1.0,
                    "embedding_s": 0.0,
                    **{
                        k: bank[k]
                        for k in (
                            "pilot_score",
                            "phase_margin",
                            "offset",
                            "search_s",
                            "collect_s",
                            "pilot_observations",
                        )
                    },
                }
                cid = hashlib.sha256(
                    f"{row['source_id']}|null|{info['condition']}".encode()
                ).hexdigest()[:24]
                p = out / (cid + ".npz")
                save_bank(p, bank, bits, case)
                all_rows.append(
                    {
                        "path": str(p),
                        "sha256": sha(p),
                        "source_id": row["source_id"],
                        "rule": "null",
                        "mode": "search",
                        "label": 0,
                        "condition": info["condition"],
                    }
                )
    atomic_json(
        out / "manifest.json",
        {
            "role": role,
            "eta": eta,
            "include_null": include_null,
            "source_manifest_hash": sha(rr / "sources.csv"),
            "message_manifest_hash": sha(rr / "messages.json"),
            "code_hash": code_digest(),
            "config": cfg,
            "rows": all_rows,
        },
    )
    log("bank saved", role=role, cases=len(all_rows), source_count=len(input_rows))
    return out / "manifest.json"


def bank_paths(cfg, role, eta, rule=None, mode="search", label=1):
    p = bank_dir(cfg, role, eta) / "manifest.json"
    manifest = read_json(p)
    if manifest["source_manifest_hash"] != sha(root(cfg) / "sources.csv"):
        raise RuntimeError("bank/source identity mismatch")
    if manifest.get("message_manifest_hash") != sha(root(cfg) / "messages.json"):
        raise RuntimeError("bank/message identity mismatch: rebuild in a new run")
    if manifest.get("code_hash") != code_digest():
        raise RuntimeError("bank/code identity mismatch: rebuild in a new run")
    if manifest["config"] != cfg:
        raise RuntimeError("bank/config mismatch: rebuild affected cache in a new run")
    paths = []
    for row in manifest["rows"]:
        if (
            (rule is None or row["rule"] == rule)
            and row["mode"] == mode
            and row["label"] == label
        ):
            if sha(row["path"]) != row["sha256"]:
                raise RuntimeError("corrupted observation bank")
            paths.append(Path(row["path"]))
    return paths
