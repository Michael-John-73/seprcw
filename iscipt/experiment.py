"""Staged, source-disjoint watermark experiments with one frozen final evaluation.

Development is an engineering search on reused data, never preregistration or an
independent replication. No image bytes from calibration/test are opened during
development. The manifest commitment covers CSV bytes, identities and file stats;
actual image hashes are recorded only when that split is permitted to be read.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import reuse
from .codec import CodecConfig, decode, embed
from .metrics import (binomial_ci, conformal_threshold, feature_vector,
                      fit_logistic, paired_bootstrap, predict_scores)

METHODS = ("raster", "tile_fixed", "tile_search", "tile_oracle")
DEPLOYABLE = METHODS[:3]
SPLITS = ("development_fit", "development_validation", "calibration", "test")
FAIL_CLOSED_SCORE = -1e12


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical(value):
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2,
                                   ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _file_hash(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(*parts):
    return int(_hash(list(parts))[:16], 16) % (2**32)


def _implementation_fingerprint():
    base = Path(__file__).resolve().parent
    return {"source_hashes": {name: _file_hash(base / name)
                              for name in ("experiment.py", "codec.py", "metrics.py", "reuse.py")},
            "runtime_versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                                 "opencv": cv2.__version__,
                                 "scipy": getattr(sys.modules.get("scipy"), "__version__", "unknown"),
                                 "pywavelets": getattr(sys.modules.get("pywt"), "__version__", "unknown")}}


def _config(path):
    cfg = _read(path)
    required = {"seed", "data", "splits", "codec", "development", "hypotheses", "evaluation"}
    if not required.issubset(cfg):
        raise ValueError("Missing config sections: " + ", ".join(sorted(required - cfg.keys())))
    if set(cfg["splits"]) != set(SPLITS) or any(int(cfg["splits"][s]) < 1 for s in SPLITS):
        raise ValueError("All four split sizes must be positive integers.")
    areas = cfg["evaluation"]["retained_area"]
    primary = cfg["evaluation"]["primary_retained_area"]
    if not areas or any(not 0 < float(a) <= 1 for a in areas):
        raise ValueError("Retained areas must lie in (0,1].")
    if not primary or not set(primary).issubset(areas) or 1.0 in primary:
        raise ValueError("Primary cropped conditions must be a nonempty subset excluding 1.0.")
    if len(set(float(a) for a in areas)) != len(areas):
        raise ValueError("Duplicate crop conditions are not permitted.")
    dev = cfg["development"]
    if int(dev["max_trials"]) < 1 or not dev["delta_candidates"]:
        raise ValueError("Development must have a finite nonempty candidate schedule.")
    if len(set(float(x) for x in dev["delta_candidates"])) != len(dev["delta_candidates"]):
        raise ValueError("Candidate delta values must be unique.")
    if not 0 < cfg["hypotheses"]["alpha"] < 1:
        raise ValueError("alpha must lie in (0,1).")
    reuse.validate_config(cfg)
    return cfg


def _context(config_path, out_dir):
    out = Path(out_dir).resolve()
    cfg = _config(config_path)
    saved = _read(out / "config.json")
    state = _read(out / "state.json")
    manifest = _read(out / "manifest.json")
    if _hash(cfg) != state["config_hash"] or _hash(saved) != state["config_hash"]:
        raise ValueError("Configuration changed after prepare; use a new experiment directory.")
    if _hash(manifest) != state["manifest_hash"]:
        raise ValueError("Frozen split manifest was modified.")
    if _implementation_fingerprint() != state["implementation"]:
        raise ValueError("Experiment implementation or runtime changed after prepare; use a new experiment directory.")
    for domain, info in state["index_files"].items():
        if _file_hash(info["path"]) != info["sha256"]:
            raise ValueError(f"Dataset index changed for {domain}.")
    reuse.verify_pins(cfg, state)
    # Stat checks are permitted for unopened splits and never decode their pixels.
    for row in manifest:
        st = Path(row["path"]).stat()
        if st.st_size != row["file_size"] or st.st_mtime_ns != row["file_mtime_ns"]:
            raise ValueError(f"Committed data file changed: {row['sample_id']}")
    reuse.verify_marked_stat(manifest)
    return cfg, out, state, manifest


def _set_state(out, state, phase, **extra):
    state.update(extra)
    state["status"] = phase
    _write(out / "state.json", state)


def prepare(config_path, out_dir):
    """Commit source identities and disjoint splits without opening image bytes."""
    cfg = _config(config_path)
    out = Path(out_dir).resolve()
    if (out / "state.json").exists():
        _context(config_path, out)
        return status(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("A new experiment directory must be empty.")
    out.mkdir(parents=True, exist_ok=True)
    root = Path(cfg["data"]["root"]).resolve()
    indexes, groups = {}, {}
    for domain in cfg["data"].get("domains", ["real", "ai"]):
        supplied = cfg["data"].get("index_csvs", {}).get(domain, f"index_{domain}.csv")
        index = Path(supplied)
        if not index.is_absolute():
            index = root / index
        indexes[domain] = {"path": str(index.resolve()), "sha256": _file_hash(index)}
        by_source = {}
        with index.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"source_file", "work_file"}.issubset(reader.fieldnames or []):
                raise ValueError(f"{index}: source_file and work_file columns required")
            for item in reader:
                identity = domain + "|" + item["source_file"].replace("\\", "/").casefold()
                path = Path(item["work_file"])
                if not path.is_absolute():
                    direct = root / path
                    path = direct if direct.is_file() else root / domain / path
                path = path.resolve()
                st = path.stat()
                row = {"sample_id": _hash(identity)[:24], "group_id": _hash(identity),
                       "domain": domain, "source_file": item["source_file"],
                       "path": str(path), "file_size": st.st_size,
                       "file_mtime_ns": st.st_mtime_ns,
                       "source_index_metadata": item}
                # One representative work file per source prevents repeated-source leakage.
                if identity not in by_source or str(path) < by_source[identity]["path"]:
                    by_source[identity] = row
        groups[domain] = sorted(by_source.values(), key=lambda x: _hash([cfg["seed"], x["group_id"]]))
    domains = sorted(groups)
    if not domains:
        raise ValueError("No data domains configured.")
    manifest, cursors = [], {d: 0 for d in domains}
    for split in SPLITS:
        n = int(cfg["splits"][split])
        for j in range(n):
            domain = domains[j % len(domains)]
            idx = cursors[domain]
            if idx >= len(groups[domain]):
                raise ValueError(f"Insufficient distinct {domain} sources for balanced splits.")
            row = dict(groups[domain][idx], split=split)
            manifest.append(row)
            cursors[domain] += 1
    if len({r["group_id"] for r in manifest}) != len(manifest):
        raise ValueError("Source-group identities collide across split manifests.")
    paths = [r["path"].casefold() for r in manifest]
    if len(set(paths)) != len(paths):
        raise ValueError("A work image occurs under multiple source identities.")
    length = int(cfg["codec"]["payload_bits"])
    legacy_payload = reuse.load_payload_source(cfg)
    if legacy_payload is None:
        payload = np.random.default_rng(_seed(cfg["seed"], "target-payload")).integers(0, 2, length).tolist()
    else:
        payload = legacy_payload["bits_list"]
    external = {}
    if legacy_payload is not None:
        external["payload_source"] = {"path": legacy_payload["path"], "sha256": legacy_payload["file_sha256"]}
    reuse_summary = {"enabled": False}
    if reuse.reuse_enabled(cfg):
        # Metadata and file stats only: no image bytes are opened before the owning stage.
        links, link_pins, reuse_summary = reuse.link_sources(cfg, manifest, legacy_payload)
        for row in manifest:
            row["reuse_link"] = links[row["sample_id"]]
        external.update(link_pins)
    state = {"status": "PREPARED", "config_hash": _hash(cfg), "manifest_hash": _hash(manifest),
             "implementation": _implementation_fingerprint(),
             "index_files": indexes, "payload": payload, "data_status": "REUSED_PREVIOUSLY_EXPLORED_DATA",
             "payload_source": reuse.payload_record(legacy_payload, payload), "crop_rule": reuse.crop_rule(cfg),
             "external_inputs": external, "reuse": reuse_summary,
             "data_commitment_scope": "index bytes + source identity + path + file size/mtime; image bytes read by stage only",
             "independent_external_validation": False, "preregistered": False,
             "counts": {s: sum(r["split"] == s for r in manifest) for s in SPLITS},
             "available_sources": {d: len(groups[d]) for d in domains},
             "unused_sources": {d: len(groups[d]) - cursors[d] for d in domains},
             "notes": ["Source grouping is domain plus source_file, not perceptual duplicate detection.",
                       "Pilot downsampling uses INTER_AREA; historical full-resolution baselines are not directly reusable.",
                       "Null calibration uses the per-image maximum over all configured crop conditions."]}
    _write(out / "config.json", cfg)
    _write(out / "manifest.json", manifest)
    _write(out / "state.json", state)
    return status(out)


def _load_image(row, cfg):
    image_bytes = Path(row["path"]).read_bytes()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image could not be decoded by OpenCV")
    original_shape = list(image.shape)
    maximum = int(cfg["evaluation"].get("image_maxside", 0))
    if maximum > 0 and max(image.shape[:2]) > maximum:
        scale = maximum / max(image.shape[:2])
        image = cv2.resize(image, (max(1, round(image.shape[1] * scale)),
                                   max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return image, {"image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                   "original_shape": original_shape, "processed_shape": list(image.shape),
                   "resize_interpolation": "INTER_AREA" if original_shape != list(image.shape) else "none"}


def _psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * math.log10(255**2 / mse)


def _crop_geometry(image, area, sample_id, seed):
    h, w = image.shape[:2]
    ch = max(1, min(h, int(math.floor(h * math.sqrt(float(area))))))
    cw = max(1, min(w, int(math.floor(w * math.sqrt(float(area))))))
    rng = np.random.default_rng(_seed(seed, sample_id, float(area), "crop"))
    top, left = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
    return top, left, ch, cw


def _geometry(cfg, source, image, area):
    if reuse.crop_rule(cfg) == reuse.LEGACY_CROP_RULE:
        return _crop_geometry(image, area, source["sample_id"], cfg["seed"])
    return reuse.crop_geometry(source, image.shape[:2], area)


def _embed_marked(original, meta, source, mode, payload, codec, codec_values, cfg):
    """Embed one layout; the legacy marked image replaces a fresh raster embedding only when every cache condition holds."""
    status, reason = reuse.embedding_plan(cfg, source, mode, codec_values, payload, meta)
    if status == reuse.CANDIDATE:
        try:
            marked, info = reuse.read_marked(source, original.shape)
            return marked, dict(embedding_origin=reuse.EMBED_LEGACY, reuse_status=reuse.CACHE_HIT, reuse_reason="", **info)
        except Exception as exc:  # unreadable or mismatching legacy image: recompute and record, never substitute silently
            status, reason = reuse.RECOMPUTED, f"legacy marked image unusable: {type(exc).__name__}: {exc}"
    return (embed(original, payload, codec, mode=mode),
            dict(embedding_origin=reuse.EMBED_NEW, reuse_status=status, reuse_reason=reason))


def _measure(rows, cfg, codec_values, payload, include_oracle=True, labels=(0, 1)):
    """Read exactly the supplied split; retain failures as denominator observations."""
    codec = CodecConfig(**codec_values)
    measurements, quality = [], []
    methods = METHODS if include_oracle else DEPLOYABLE
    for source in rows:
        identity = {k: source[k] for k in ("sample_id", "group_id", "domain", "split")}
        try:
            original, meta = _load_image(source, cfg)
            image_error = None
        except Exception as exc:
            original, meta, image_error = None, {}, f"{type(exc).__name__}: {exc}"
        provenance = reuse.carrier_provenance(cfg, source, meta if original is not None else None)
        if provenance["size_check"] == "MISMATCH":
            # Frozen crop boxes belong to the prepared size; a different decoded size stays as a failed observation.
            image_error, original = reuse.size_mismatch_error(source, meta), None
        meta = dict(meta, **provenance)
        embedded, embed_errors, records = {}, {}, {}
        for mode in ("raster", "tile"):
            positive = reuse.not_embedded_record(image_error or "no positive embedding requested")
            try:
                if original is None:
                    raise ValueError(image_error)
                if 1 in labels:
                    embedded[mode], positive = _embed_marked(original, meta, source, mode, payload, codec,
                                                             codec_values, cfg)
                    quality.append(dict(identity, mode=mode, psnr=_psnr(original, embedded[mode]),
                                        status="ok", **meta, **positive))
            except Exception as exc:
                embed_errors[mode] = f"{type(exc).__name__}: {exc}"
                positive = reuse.not_embedded_record(embed_errors[mode])
                quality.append(dict(identity, mode=mode, psnr=None, status="embed_failed",
                                    error=embed_errors[mode], **meta, **positive))
            records[mode] = {0: reuse.unmarked_record(), 1: positive}
        for area in cfg["evaluation"]["retained_area"]:
            geometry = _geometry(cfg, source, original, area) if original is not None else (0, 0, 0, 0)
            top, left, ch, cw = geometry
            for method in methods:
                mode = "raster" if method == "raster" else "tile"
                for label in labels:
                    if label == 0 and method == "tile_oracle":
                        continue  # oracle cannot make deployable detection claims
                    row = dict(identity, method=method, label=int(label), retained_area=float(area),
                               top=top, left=left, crop_h=ch, crop_w=cw, resize_after_crop=False,
                               legacy_library_eligible=bool(ch * cw >= 256 * 256),
                               baseline_scope="shared minimum-8-pixel codec core; not the legacy library size gate",
                               legacy_artifact_id=reuse.legacy_artifact_id(source, area, method, label),
                               bit_accuracy=0.0, exact_recovery=0, features=None, **meta, **records[mode][int(label)])
                    try:
                        if original is None:
                            raise ValueError(image_error)
                        if label and mode in embed_errors:
                            raise ValueError(embed_errors[mode])
                        image = embedded[mode] if label else original
                        crop = image[top:top + ch, left:left + cw].copy()
                        kwargs = {"origin": (top, left)} if method == "tile_oracle" else {}
                        result = decode(crop, len(payload), codec, mode=method, **kwargs)
                        bits = np.asarray(result["bits"], dtype=int)
                        if len(bits) != len(payload) or result.get("status", "ok") not in ("ok", "OK", "success"):
                            raise ValueError("Decoder returned invalid length/status: " + str(result.get("status")))
                        features = np.asarray(feature_vector(result, np.asarray(payload)), dtype=float)
                        if features.shape != (4,) or not np.all(np.isfinite(features)):
                            raise ValueError("Nonfinite or malformed decoder feature vector")
                        row.update(status="ok", bits=bits.tolist(), soft=_jsonable(result["soft"]),
                                   bit_accuracy=float(np.mean(bits == payload)),
                                   exact_recovery=int(np.array_equal(bits, payload)), features=features.tolist(),
                                   pilot_score=float(result["pilot_score"]),
                                   phase_margin=float(result["phase_margin"]), coverage=float(result["coverage"]),
                                   offset=_jsonable(result["offset"]), elapsed_s=float(result["elapsed_s"]))
                    except Exception as exc:
                        row.update(status="undecodable", error=f"{type(exc).__name__}: {exc}",
                                   score=None, accepted=False)
                    measurements.append(row)
    return measurements, quality


def _save_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(_canonical(row).decode("utf-8") + "\n")
    if rows:
        fields = sorted(set().union(*(r.keys() for r in rows)))
        with path.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(_jsonable(v), ensure_ascii=False, allow_nan=False)
                                 if isinstance(v, (list, dict)) else v for k, v in _jsonable(row).items()})


def _rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _fit_models(rows, training):
    models = {}
    for method in DEPLOYABLE:
        usable = [r for r in rows if r["method"] == method and r["status"] == "ok"]
        if not usable or {r["label"] for r in usable} != {0, 1}:
            raise ValueError(f"Cannot fit {method}: both labels with valid measurements required")
        models[method] = fit_logistic(np.asarray([r["features"] for r in usable]),
                                      np.asarray([r["label"] for r in usable]), **training)
    return _jsonable(models)


def _score(rows, models):
    for row in rows:
        if row["method"] in models and row["status"] == "ok":
            row["score"] = float(predict_scores(np.asarray([row["features"]]), models[row["method"]])[0])
        else:
            row["score"] = None


def _null_maxima(rows, method):
    grouped = defaultdict(list)
    for row in rows:
        if row["method"] == method and row["label"] == 0:
            grouped[row["sample_id"]].append(row)
    return [{"sample_id": sid, "group_id": vals[0]["group_id"], "domain": vals[0]["domain"],
             "score": max([v["score"] for v in vals if v.get("score") is not None] or [FAIL_CLOSED_SCORE]),
             "n_conditions": len(vals), "n_undecodable": sum(v["status"] != "ok" for v in vals)}
            for sid, vals in sorted(grouped.items())]


def _vectors(rows, cfg, field="exact_recovery"):
    primary = {float(a) for a in cfg["evaluation"]["primary_retained_area"]}
    grouped = defaultdict(dict)
    for row in rows:
        if row["label"] == 1 and row["retained_area"] in primary:
            grouped[(row["method"], row["sample_id"])][row["retained_area"]] = float(row.get(field, 0))
    sample_ids = sorted({key[1] for key in grouped})
    result = {}
    for method in METHODS:
        result[method] = np.asarray([sum(grouped.get((method, sid), {}).get(a, 0) for a in primary) / len(primary)
                                     for sid in sample_ids], dtype=float)
    return sample_ids, result


def _accept(rows, thresholds):
    for row in rows:
        value = thresholds.get(row["method"], {}).get("threshold")
        row["accepted"] = bool(value is not None and row.get("score") is not None and row["score"] > value)
        row["accepted_correct_recovery"] = int(row["accepted"] and row["exact_recovery"] == 1)


def _development_summary(fit, validation, quality, cfg):
    _, vec = _vectors(validation, cfg)
    h1 = float(np.mean(vec["tile_search"] - vec["raster"]))
    h2 = float(np.mean(vec["tile_search"] - vec["tile_fixed"]))
    gap = float(np.mean(vec["tile_oracle"] - vec["tile_search"]))
    qmeans = {m: float(np.mean([q["psnr"] for q in quality if q["mode"] == m and q["psnr"] is not None]))
              if any(q["mode"] == m and q["psnr"] is not None for q in quality) else None
              for m in ("raster", "tile")}
    failures = sum(r["status"] != "ok" for r in validation)
    qfail = sum(q["status"] != "ok" for q in quality)
    qcfg, hyp = cfg["development"], cfg["hypotheses"]
    reasons = []
    if h1 < hyp["h1_min_gain"]:
        reasons.append("H1 development exact-recovery gain below fixed target")
    if h2 < hyp["h2_min_gain"]:
        reasons.append("H2 development search-versus-fixed gain below fixed target")
    if gap > hyp["oracle_max_gap"]:
        reasons.append("Development oracle deficit exceeds fixed target")
    if any(v is None or v < qcfg["min_mean_psnr"] for v in qmeans.values()):
        reasons.append("Mean PSNR below fixed minimum or embedding unavailable")
    if all(v is not None for v in qmeans.values()) and abs(qmeans["raster"] - qmeans["tile"]) > qcfg["max_psnr_gap_db"]:
        reasons.append("Raster/tile mean PSNR difference exceeds fixed quality budget")
    if failures or qfail:
        reasons.append(f"Undecodable validation rows={failures}; failed embedding rows={qfail}")
    provisional = {}
    for method in DEPLOYABLE:
        values = [r["score"] for r in _null_maxima(fit, method)]
        provisional[method] = {"threshold": float(np.quantile(values, 1 - hyp["alpha"], method="higher")),
                               "status": "NOT_CERTIFIED", "source": "development_fit_empirical_quantile"}
    _accept(validation, provisional)
    _, accepted = _vectors(validation, cfg, "accepted_correct_recovery")
    dev_fpr = {}
    for method in DEPLOYABLE:
        values = _null_maxima(validation, method)
        dev_fpr[method] = float(np.mean([r["score"] > provisional[method]["threshold"] for r in values]))
    return {"passed": not reasons, "failure_reasons": reasons, "h1_gain": h1, "h2_gain": h2,
            "oracle_gap": gap, "mean_psnr": qmeans, "undecodable_validation_rows": failures,
            "failed_embedding_rows": qfail, "gate_type": "provisional engineering thresholds; no inferential claim",
            "h3_development": {"status": "NOT_CERTIFIED", "informational_only": True,
                               "accepted_correct_gain": float(np.mean(accepted["tile_search"] - accepted["raster"])),
                               "imagewise_fpr": dev_fpr, "thresholds": provisional},
            "selection_key": [h1, h2, -gap]}


def _next_delta(current, remaining, summary):
    """Respond to development failures without enlarging the declared search pool."""
    reasons = summary.get("failure_reasons", [])
    quality_failed = any("PSNR" in reason or "embedding unavailable" in reason for reason in reasons)
    recovery_failed = any("recovery" in reason or "oracle" in reason or "Undecodable" in reason for reason in reasons)
    lower = sorted(d for d in remaining if d < current)
    higher = sorted(d for d in remaining if d > current)
    if quality_failed and lower:
        return lower[-1], "Quality gate failed: try the nearest lower predeclared delta; keep all quality and hypothesis targets fixed"
    if recovery_failed and not quality_failed and higher:
        return higher[0], "Recovery/synchronization gate failed with acceptable quality: try the nearest higher predeclared delta"
    if remaining:
        return remaining[0], "Use the next untried predeclared candidate; no permitted directional refinement remains"
    return None, "Declared candidate pool exhausted"


def develop(config_path, out_dir):
    """Run the prebounded candidate list on development splits only."""
    cfg, out, state, manifest = _context(config_path, out_dir)
    if (out / "final.lock.json").exists() or (out / "calibration.json").exists():
        raise ValueError("Development is locked once calibration/final evaluation begins.")
    if (out / "development.json").exists():
        report = _read(out / "development.json")
        if _hash(report) != state.get("development_hash"):
            raise ValueError("Development report changed after completion.")
        if report.get("selected") is not None:
            _verify_selection(out, state)
        return report
    fit_sources = [r for r in manifest if r["split"] == "development_fit"]
    val_sources = [r for r in manifest if r["split"] == "development_validation"]
    remaining = [float(x) for x in cfg["development"]["delta_candidates"]]
    n_trials = min(len(remaining), int(cfg["development"]["max_trials"]))
    training = dict(lr=0.1, steps=300, l2=0.01)
    training.update(cfg.get("training", {}))
    trials = []
    _set_state(out, state, "DEVELOPING")
    delta = remaining[0]
    for i in range(n_trials):
        remaining.remove(delta)
        codec_values = dict(cfg["codec"], delta=float(delta))
        folder = out / "trials" / f"trial_{i:03d}"
        trial_config = {"index": i, "codec": codec_values, "training": training, "config_hash": state["config_hash"],
                        "manifest_hash": state["manifest_hash"]}
        if (folder / "summary.json").exists():
            if _read(folder / "candidate.json") != trial_config:
                raise ValueError("Cannot resume a modified candidate schedule")
            summary = _read(folder / "summary.json")
            for name, digest in summary["artifact_hashes"].items():
                if _file_hash(folder / name) != digest:
                    raise ValueError("Completed development candidate artifact changed: " + name)
        else:
            _write(folder / "candidate.json", trial_config)
            fit, fit_quality, validation, val_quality = [], [], [], []
            try:
                fit, fit_quality = _measure(fit_sources, cfg, codec_values, state["payload"])
                validation, val_quality = _measure(val_sources, cfg, codec_values, state["payload"])
                models = _fit_models(fit, training)
                _score(fit, models)
                _score(validation, models)
                summary = _development_summary(fit, validation, val_quality, cfg)
            except (ValueError, RuntimeError) as exc:
                models = {}
                summary = {"passed": False, "failure_reasons": [str(exc)], "h1_gain": None,
                           "h2_gain": None, "oracle_gap": None, "selection_key": [-1, -1, -1]}
            _save_rows(folder / "development_fit.jsonl", fit)
            _save_rows(folder / "development_validation.jsonl", validation)
            _save_rows(folder / "quality.jsonl", fit_quality + val_quality)
            _write(folder / "models.json", models)
            summary.update(index=i, codec=codec_values, training=training,
                           artifact_hashes={name: _file_hash(folder / name) for name in
                                           ("candidate.json", "development_fit.jsonl", "development_validation.jsonl", "quality.jsonl", "models.json")})
            next_delta, reason = _next_delta(delta, remaining, summary) if i + 1 < n_trials else (None, "Frozen trial budget exhausted")
            summary["next_adjustment"] = {"proposed_next_delta": next_delta,
                                           "action": "REFINE_WITHIN_DECLARED_POOL" if next_delta is not None else "STOP_MAX_TRIALS",
                                           "reason": reason,
                                           "criteria_unchanged": True,
                                           "previous_failure_reasons": summary["failure_reasons"]}
            _write(folder / "summary.json", summary)
        trials.append(summary)
        if i + 1 < n_trials:
            delta = summary["next_adjustment"]["proposed_next_delta"]
            if delta not in remaining:
                raise ValueError("Saved next-candidate decision is inconsistent with frozen pool.")
    passing = [t for t in trials if t["passed"]]
    selected = max(passing, key=lambda t: tuple(t["selection_key"])) if passing else None
    report = {"status": "DEVELOPMENT_PASSED" if selected else "BLOCKED_DEV", "trials": trials,
              "max_trials": int(cfg["development"]["max_trials"]), "n_trials": len(trials),
              "selected": selected["index"] if selected else None,
              "selection_rule": "lexicographic H1 gain, H2 gain, negative oracle gap among candidates passing all fixed development gates",
              "hypothesis_thresholds": cfg["hypotheses"], "data_status": state["data_status"],
              "next_step": "calibrate" if selected else "Stop this experiment; any further design search needs a new documented development experiment, not final-data tuning"}
    if selected:
        folder = out / "trials" / f"trial_{selected['index']:03d}"
        frozen = {"codec": selected["codec"], "models": _read(folder / "models.json"),
                  "payload": state["payload"], "training": training, "trial_index": selected["index"],
                  "trial_summary_hash": _hash(selected), "config_hash": state["config_hash"],
                  "manifest_hash": state["manifest_hash"], "development_validation_hash": selected["artifact_hashes"]["development_validation.jsonl"],
                  "development_fit_hash": selected["artifact_hashes"]["development_fit.jsonl"]}
        _write(out / "frozen_selection.json", frozen)
        state["selection_hash"] = _hash(frozen)
    _write(out / "development.json", report)
    _set_state(out, state, report["status"], development_hash=_hash(report))
    return report


def _verify_selection(out, state):
    if not (out / "frozen_selection.json").exists():
        raise ValueError("No passing development selection; calibration/final is blocked.")
    frozen = _read(out / "frozen_selection.json")
    if _hash(frozen) != state.get("selection_hash"):
        raise ValueError("Frozen codec parameters or detection models changed.")
    report = _read(out / "development.json")
    if _hash(report) != state.get("development_hash") or report["status"] != "DEVELOPMENT_PASSED":
        raise ValueError("Development report changed or did not pass.")
    return frozen


def calibrate(config_path, out_dir):
    """Freeze per-method image-maximum null thresholds on calibration only."""
    cfg, out, state, manifest = _context(config_path, out_dir)
    frozen = _verify_selection(out, state)
    if (out / "final.lock.json").exists():
        raise ValueError("Final evaluation has already started; calibration cannot run again.")
    if (out / "calibration.json").exists():
        report = _read(out / "calibration.json")
        if _hash(report) != state.get("calibration_hash") or _file_hash(out / "calibration_raw.jsonl") != report["raw_scores_hash"]:
            raise ValueError("Frozen calibration artifacts changed.")
        return report
    sources = [r for r in manifest if r["split"] == "calibration"]
    rows, _ = _measure(sources, cfg, frozen["codec"], frozen["payload"], include_oracle=False, labels=(0,))
    _score(rows, frozen["models"])
    _save_rows(out / "calibration_raw.jsonl", rows)
    maxima = {m: _null_maxima(rows, m) for m in DEPLOYABLE}
    thresholds = {m: conformal_threshold([x["score"] for x in maxima[m]], cfg["hypotheses"]["alpha"])
                  for m in DEPLOYABLE}
    report = {"status": "CALIBRATED" if all(t["feasible"] for t in thresholds.values()) else "CALIBRATION_INCONCLUSIVE",
              "thresholds": thresholds, "selection_hash": state["selection_hash"],
              "raw_scores_hash": _file_hash(out / "calibration_raw.jsonl"),
              "image_maxima": maxima, "image_maxima_hash": _hash(maxima),
              "unit": "one source image; maximum null score over every configured crop",
              "undecodable_rows": sum(r["status"] != "ok" for r in rows),
              "guarantee_scope": "Marginal split-conformal control under exchangeability; fixed-threshold population FPR requires independent test confidence bounds",
              "precision_note": "A threshold of null means an infinite, always-negative threshold; insufficient calibration is not a PASS"}
    _write(out / "calibration.json", report)
    _set_state(out, state, report["status"], calibration_hash=_hash(report))
    return report


def _decision_gain(interval, target):
    if interval["low"] > target:
        return "PASS"
    if interval["high"] <= target:
        return "FAIL"
    return "INCONCLUSIVE"


def _combine(states):
    return "FAIL" if "FAIL" in states else ("INCONCLUSIVE" if "INCONCLUSIVE" in states else "PASS")


def finalize(config_path, out_dir):
    """Consume the untouched test split exactly once; no tuning or automatic retry."""
    cfg, out, state, manifest = _context(config_path, out_dir)
    frozen = _verify_selection(out, state)
    if (out / "final.lock.json").exists():
        raise ValueError("Final evaluation already started. It cannot be repeated in this experiment directory; use status to inspect results.")
    calibration = _read(out / "calibration.json")
    if _hash(calibration) != state.get("calibration_hash") or _file_hash(out / "calibration_raw.jsonl") != calibration["raw_scores_hash"]:
        raise ValueError("Calibration thresholds or raw scores changed after freezing.")
    lock = {k: state[k] for k in ("config_hash", "manifest_hash", "selection_hash", "development_hash", "calibration_hash")}
    lock.update(thresholds_hash=_hash(calibration["thresholds"]), implementation_hash=_hash(state["implementation"]),
                status="FINAL_STARTED_SINGLE_USE")
    # Exclusive creation precedes the very first access to test image bytes.
    with (out / "final.lock.json").open("x", encoding="utf-8") as stream:
        json.dump(lock, stream, indent=2, allow_nan=False)
    _set_state(out, state, "FINAL_RUNNING")
    try:
        sources = [r for r in manifest if r["split"] == "test"]
        rows, quality = _measure(sources, cfg, frozen["codec"], frozen["payload"])
        _score(rows, frozen["models"])
        _accept(rows, calibration["thresholds"])
        _save_rows(out / "test_raw.jsonl", rows)
        _save_rows(out / "test_quality.jsonl", quality)
        _, vectors = _vectors(rows, cfg)
        _, accepted = _vectors(rows, cfg, "accepted_correct_recovery")
        options = {"seed": int(cfg["seed"]), "reps": int(cfg["evaluation"]["bootstrap_reps"]),
                   "confidence": float(cfg["evaluation"]["confidence"])}
        h1 = paired_bootstrap(vectors["tile_search"], vectors["raster"], **options)
        h2 = paired_bootstrap(vectors["tile_search"], vectors["tile_fixed"], **options)
        oracle = paired_bootstrap(vectors["tile_oracle"], vectors["tile_search"], **options)
        h3_gain = paired_bootstrap(accepted["tile_search"], accepted["raster"], **options)
        fprs, domain_fprs = {}, {}
        for method in DEPLOYABLE:
            maxima = _null_maxima(rows, method)
            threshold = calibration["thresholds"][method]["threshold"]
            decisions = [bool(threshold is not None and m["score"] > threshold) for m in maxima]
            fprs[method] = binomial_ci(sum(decisions), len(decisions), confidence=options["confidence"], one_sided=True)
            # Upper bounds establish control; complementary exact lower bounds
            # separately establish a violation (upper-only intervals have low=0).
            fprs[method]["failure_lower"] = 1.0 - binomial_ci(len(decisions) - sum(decisions), len(decisions),
                                                            confidence=options["confidence"], one_sided=True)["high"]
            domain_fprs[method] = {}
            for domain in sorted({m["domain"] for m in maxima}):
                selected = [d for m, d in zip(maxima, decisions) if m["domain"] == domain]
                domain_fprs[method][domain] = binomial_ci(sum(selected), len(selected), confidence=options["confidence"], one_sided=True)
        hyp = cfg["hypotheses"]
        h1_state = _decision_gain(h1, hyp["h1_min_gain"])
        oracle_state = "PASS" if oracle["high"] <= hyp["oracle_max_gap"] else ("FAIL" if oracle["low"] > hyp["oracle_max_gap"] else "INCONCLUSIVE")
        h2_state = _combine([_decision_gain(h2, hyp["h2_min_gain"]), oracle_state])
        feasible = all(calibration["thresholds"][m]["feasible"] for m in ("raster", "tile_search"))
        fpr_states = ["PASS" if fprs[m]["high"] <= hyp["alpha"] else
                      ("FAIL" if fprs[m]["failure_lower"] > hyp["alpha"] else "INCONCLUSIVE") for m in ("raster", "tile_search")]
        h3_state = _combine(fpr_states + [_decision_gain(h3_gain, 0.0)]) if feasible else "INCONCLUSIVE"
        hypotheses = {"H1": {"status": h1_state, "gain_ci": h1, "minimum_gain": hyp["h1_min_gain"]},
                      "H2": {"status": h2_state, "gain_ci": h2, "minimum_gain": hyp["h2_min_gain"],
                             "oracle_deficit_ci": oracle, "maximum_oracle_deficit": hyp["oracle_max_gap"]},
                      "H3": {"status": h3_state, "accepted_correct_gain_ci": h3_gain,
                             "nominal_fpr": hyp["alpha"], "imagewise_fpr_ci": fprs,
                             "calibration_feasible": feasible,
                             "scope": "Pooled source-domain mixture; both raster and tile_search require FPR upper bounds <= alpha"}}
        minimum_sources = int(cfg["evaluation"].get("min_test_sources_for_inference", 100))
        if len(sources) < minimum_sources:
            for result in hypotheses.values():
                result["diagnostic_interval_status"] = result["status"]
                result["status"] = "INCONCLUSIVE"
                result["exploratory_reason"] = "Test cohort below the prespecified minimum; interval calculations are engineering diagnostics only"
                result["minimum_test_sources_for_inference"] = minimum_sources
            h1_state = h2_state = h3_state = "INCONCLUSIVE"
        report = {"status": _combine([h1_state, h2_state, h3_state]), "hypotheses": hypotheses,
                  "data_status": state["data_status"], "independent_external_validation": False,
                  "frozen_lock": lock, "test_raw_hash": _file_hash(out / "test_raw.jsonl"),
                  "test_quality_hash": _file_hash(out / "test_quality.jsonl"),
                  "n_test_source_images": len(sources), "n_rows": len(rows),
                  "undecodable_rows": sum(r["status"] != "ok" for r in rows),
                  "failures_by_method": {m: sum(r["method"] == m and r["status"] != "ok" for r in rows) for m in METHODS},
                  "domain_fpr_exploratory": domain_fprs,
                  "inference_notes": ["Paired bootstrap resamples source images, after averaging the fixed primary crop conditions per image.",
                                      "Null FPR unit is any accepted condition for a source image, not independent crop rows.",
                                      "Oracle is a synchronization upper-bound diagnostic, excluded from deployable FPR claims.",
                                      "A nonsignificant or imprecise result is INCONCLUSIVE, not proof of no effect.",
                                      "No final result may trigger parameter refinement in this experiment."]}
        _write(out / "final.json", report)
        _set_state(out, state, report["status"], final_hash=_hash(report))
        return report
    except Exception as exc:
        _set_state(out, state, "FINAL_INTERRUPTED", final_error=f"{type(exc).__name__}: {exc}")
        raise


def status(out_dir):
    out = Path(out_dir).resolve()
    if not (out / "state.json").exists():
        return {"status": "NOT_PREPARED", "out_dir": str(out)}
    state = _read(out / "state.json")
    result = dict(state, out_dir=str(out), final_locked=(out / "final.lock.json").exists())
    if (out / "final.json").exists():
        report = _read(out / "final.json")
        if _hash(report) != state.get("final_hash"):
            raise ValueError("Final report changed after completion.")
        result["hypotheses"] = report["hypotheses"]
    elif (out / "development.json").exists():
        report = _read(out / "development.json")
        result["selected_trial"] = report["selected"]
        result["trial_results"] = [{"index": t["index"], "passed": t["passed"],
                                    "failure_reasons": t["failure_reasons"]} for t in report["trials"]]
    return result
