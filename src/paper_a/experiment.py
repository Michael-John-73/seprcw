"""Development selection, separately fitted acceptance, freeze, and independent testing."""

from __future__ import annotations
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
from tqdm import tqdm
from .io import *
from .data import *
from .codec import baseline, detector_features, collect, embed, png_roundtrip
from .learning import load_model, predict, train_one, device_for
from .statistics import *

METHODS = {
    "E0D0": ("E0", "D0"),
    "E1D0": ("E1", "D0"),
    "E2D0": ("E2", "D0"),
    "E2D1": ("E2", "D1"),
    "E2D2": ("E2", "D2"),
    "E2D3": ("E2", "D3"),
    "E2D4": ("E2", "D4"),
}
CONTRASTS = [
    ("H_A_embed", "rank", "E1D0", "E2D0", "margin_embed"),
    ("H_A_decode_D0", "er", "E2D3", "E2D0", "margin_decode"),
    ("H_A_decode_D1", "er", "E2D3", "E2D1", "margin_decode"),
    ("H_A_decode_D2", "er", "E2D3", "E2D2", "margin_decode"),
    ("H_A_decode_D4", "er", "E2D3", "E2D4", "margin_decode"),
    ("H_A_joint", "er", "E2D3", "E0D0", "margin_joint"),
]


def selected_artifacts(cfg):
    rr = root(cfg)
    s = read_json(rr / "selected.json")
    for key, hashkey in (
        ("checkpoint", "checkpoint_hash"),
        ("histogram", "histogram_hash"),
    ):
        if sha(s[key]) != s[hashkey]:
            raise RuntimeError("selected state changed")
    model, _ = load_model(s["checkpoint"], device_for(cfg))
    if getattr(model, "_uniform_affine", None) is None:
        raise RuntimeError("selected checkpoint lacks required D4 train-fitted state")
    return s, model, np.load(s["histogram"], allow_pickle=False)


def rows_for_bank(cfg, role, eta, model, histogram):
    """Decode each E2 bank once per method; synchronization bank stays shared."""
    rows = []
    for method, (rule, dec) in METHODS.items():
        marked = bank_paths(cfg, role, eta, rule, label=1)
        null = bank_paths(cfg, role, eta, "null", label=0)
        for p in tqdm(marked + null, desc=f"{role} {method}", file=sys.stdout):
            b = load_bank(p)
            meta = b["meta"]
            begin = time.perf_counter()
            weight_summary = None
            if dec == "D3":
                bits, ev, weight_summary = predict(
                    b, model, device_for(cfg), return_summary=True
                )
            else:
                bits, ev = baseline(b, dec, histogram, getattr(model, "_uniform_affine", None))
            pooling_s = time.perf_counter() - begin
            total_s = b["collect_s"] + pooling_s
            timeout = (
                cfg["evaluation"]["max_decode_seconds"] is not None
                and total_s > cfg["evaluation"]["max_decode_seconds"]
            )
            er = int(np.array_equal(bits, b["bits"]) and not timeout)
            feature_start = time.perf_counter()
            feature = detector_features(b, ev, b["bits"])
            reference_features_s = time.perf_counter() - feature_start
            total_s += reference_features_s
            timeout = bool(timeout or (cfg["evaluation"]["max_decode_seconds"] is not None
                           and total_s > cfg["evaluation"]["max_decode_seconds"]))
            er = int(er and not timeout)
            valid = bool(
                np.all(b["counts"] > 0) and b["pilot_observations"] >= 4 and not timeout
            )
            rows.append(
                {
                    "source_id": meta["source_id"],
                    "source_group": meta["source_group"],
                    "domain": meta["domain"],
                    "dataset": meta["dataset"],
                    "role": role,
                    "method": method,
                    "label": meta["label"],
                    "condition": meta["condition"],
                    "area": meta["area"],
                    "top": meta["top"],
                    "left": meta["left"],
                    "height": meta["height"],
                    "width": meta["width"],
                    "message": meta["message"],
                    "er": er,
                    "bit_accuracy": float(np.mean(bits == b["bits"])),
                    "coverage": float(np.mean(b["counts"] > 0)),
                    "valid_for_acceptance": valid,
                    "phase_correct": bool(
                        tuple(b["offset"]) == (meta["top"] % 64, meta["left"] % 64)
                    ),
                    "offset_y": int(b["offset"][0]),
                    "offset_x": int(b["offset"][1]),
                    "erasure_bits": int(np.sum(b["counts"] == 0)),
                    "rank": meta["rank_change"],
                    "psnr": meta["psnr"],
                    "ssim": meta["ssim"],
                    "embedding_s": meta["embedding_s"],
                    "search_s": b["search_s"],
                    "collect_s": b["collect_s"],
                    "pooling_s": pooling_s,
                    "reference_features_s": reference_features_s,
                    "decode_s": total_s,
                    "timeout": timeout,
                    "features": feature.tolist(),
                    "predicted_bits": bits.tolist(),
                    "reference_bits": b["bits"].tolist(),
                    "evidence": np.asarray(ev).tolist(),
                    "weight_summary": weight_summary,
                    "observation_bank_hash": sha(p),
                    "checkpoint_hash": getattr(model, "_checkpoint_hash", None),
                }
            )
    return rows


def source_values(rows, metric, method, primary):
    grouped = {}
    for r in rows:
        if r["label"] == 1 and r["method"] == method and r["area"] in primary:
            grouped.setdefault(r["source_group"], []).append(float(r[metric]))
    if any(len(x) != len(primary) for x in grouped.values()):
        raise RuntimeError("missing/duplicate primary conditions")
    return {k: float(np.mean(v)) for k, v in grouped.items()}


def quality_gate(cfg, rows):
    e = cfg["evaluation"]
    missing = [
        k for k in ("min_psnr", "min_ssim", "max_decode_seconds") if e[k] is None
    ]
    if missing:
        return {
            "ready": False,
            "reason": "unlocked quality/time requirements",
            "missing": missing,
        }
    # No cherry-picking of low-quality images: gate every source image and report every failure.
    marked = [r for r in rows if r["label"] == 1]
    badq = sorted(
        {
            r["source_id"]
            for r in marked
            if r["psnr"] < e["min_psnr"] or r["ssim"] < e["min_ssim"]
        }
    )
    timeout = sum(r["timeout"] for r in marked)
    return {
        "ready": not badq and timeout == 0,
        "failed_sources": badq,
        "timeouts": timeout,
    }


def supplemental_candidate_selection(cfg, results):
    """Select a development candidate, never certify operational quality or testing."""
    expected = {(eta, seed) for eta in cfg["codec"]["eta_candidates"]
                for seed in cfg["training"]["seeds"]}
    if len(results) != len(expected) or {(r["eta"], r["seed"]) for r in results} != expected:
        raise RuntimeError("Supplemental selection requires the entire registered candidate panel")
    if any(not r.get("training_healthy") or not r.get("reload_verified_against_live_best_model")
           or not math.isfinite(r["primary_er"]) or not math.isfinite(r["best_validation_loss"])
           for r in results):
        raise RuntimeError("Supplemental selection requires healthy verified finite candidates")
    canonical = cfg["training"]["seeds"][0]
    candidates = [r for r in results if r["seed"] == canonical]
    candidates.sort(key=lambda r: (-r["primary_er"], r["best_validation_loss"], r["eta"]))
    return {
        "scope": "SUPPLEMENTAL_DEVELOPMENT_CANDIDATE_ONLY",
        "candidate": candidates[0],
        "canonical_seed": canonical,
        "selection_rule": "max primary validation ER; min BCE; min eta; canonical seed only",
        "quality_time_certified": False,
        "independent_confirmatory_admission": False,
        "not_a_replacement_for_selected_json": True,
    }


def develop(cfg):
    rr = root(cfg)
    require_stage(cfg, "prepare")
    require_stage(cfg, "math-check")
    if cfg["study"]["mode"] != "diagnostic":
        require_stage(cfg, "preflight")
        require_stage(cfg, "memory_profile")
    if (rr / "freeze.json").exists() or (rr / "design_lock.json").exists():
        raise RuntimeError("development forbidden after design lock/freeze")
    if (rr / "development_report.json").exists():
        raise RuntimeError(
            "development already completed; fork a new run to change the budget"
        )
    result = []
    for eta in cfg["codec"]["eta_candidates"]:
        for role in ("train", "validation"):
            if not (bank_dir(cfg, role, eta) / "manifest.json").exists():
                make_bank(cfg, role, eta)
        for seed in cfg["training"]["seeds"]:
            report = train_one(
                cfg, eta, seed, rr / "models" / f"eta_{eta:g}_seed_{seed}"
            )
            model, _ = load_model(report["checkpoint"], device_for(cfg))
            hist = np.load(report["histogram"], allow_pickle=False)
            rows = rows_for_bank(cfg, "validation", eta, model, hist)
            vals = source_values(rows, "er", "E2D3", cfg["evaluation"]["primary_areas"])
            gate = quality_gate(cfg, rows)
            score = float(np.mean(list(vals.values())))
            result.append({**report, "primary_er": score, "quality": gate})
            atomic_json(
                rr / "models" / f"eta_{eta:g}_seed_{seed}" / "validation_rows.json",
                rows,
            )
    canonical = cfg["training"]["seeds"][0]
    atomic_json(rr / "supplemental_candidate.json", supplemental_candidate_selection(cfg, result))
    eligible = [r for r in result if r["seed"] == canonical and r["quality"]["ready"]]
    if not eligible:
        atomic_json(
            rr / "development_report.json",
            {
                "ready": False,
                "results": result,
                "reason": "canonical candidate quality/time or missing requirements",
            },
        )
        raise RuntimeError(
            "DEV_NOT_READY: inspect actual evidence; do not relax gates against final data"
        )
    # Validation ER high -> loss low -> eta small; never choose a lucky seed.
    eligible.sort(key=lambda r: (-r["primary_er"], r["best_validation_loss"], r["eta"]))
    selected = eligible[0]
    selected = {
        **selected,
        "histogram_hash": sha(selected["histogram"]),
        "canonical_seed": canonical,
        "selection_rule": "max primary validation ER; min BCE; min eta; canonical seed only",
    }
    atomic_json(rr / "selected.json", selected)
    atomic_json(
        rr / "development_report.json",
        {
            "ready": True,
            "results": result,
            "selected_eta": selected["eta"],
            "empirical_hypotheses_not_yet_tested": True,
            "seed_sensitivity": [
                {"eta": r["eta"], "seed": r["seed"], "primary_er": r["primary_er"]}
                for r in result
            ],
        },
    )
    return evidence_record(
        cfg,
        "develop",
        [
            rr / "development_report.json",
            rr / "selected.json",
            selected["checkpoint"],
            selected["histogram"],
        ],
    )


def fit_detector(cfg):
    rr = root(cfg)
    require_stage(cfg, "develop")
    if cfg["study"]["mode"] != "diagnostic":
        verify_design(cfg)
    if (rr / "freeze.json").exists():
        raise RuntimeError("detector fitting is pre-freeze only in this protocol")
    s, model, hist = selected_artifacts(cfg)
    if not (bank_dir(cfg, "detector_fit", s["eta"]) / "manifest.json").exists():
        make_bank(cfg, "detector_fit", s["eta"], include_null=True)
    rows = rows_for_bank(cfg, "detector_fit", s["eta"], model, hist)
    detectors = {}
    for name in METHODS:
        sub = [r for r in rows if r["method"] == name and r["valid_for_acceptance"]]
        x = np.asarray([r["features"] for r in sub], np.float64)
        y = np.asarray([r["label"] for r in sub], np.float64)
        if x.ndim != 2 or len(np.unique(y)) != 2:
            raise RuntimeError("detector fit needs both classes and valid observations")
        from .compute import cuda_enabled, tensor
        on_cuda = cuda_enabled()
        if on_cuda:
            x, y = tensor(x), tensor(y)
            mean, std = x.mean(0), x.std(0, correction=0).clamp_min(1e-6)
        else:
            mean, std = x.mean(0), np.maximum(x.std(0), 1e-6)
        x = (x - mean) / std
        x = (torch.cat((x, torch.ones((len(x), 1), device="cuda", dtype=torch.float64)), 1)
             if on_cuda else np.c_[x, np.ones(len(x))])
        l2 = cfg["detector"]["l2"]
        # Explicit deterministic gradient descent: all lr/l2/steps are used and logged.
        w = torch.zeros(5, device="cuda", dtype=torch.float64) if on_cuda else np.zeros(5, np.float64)
        for step in tqdm(
            range(cfg["detector"]["steps"]), desc=f"acceptance fit {name}",
            leave=False, file=sys.stdout,
        ):
            p = (x @ w).sigmoid() if on_cuda else expit(x @ w)
            penalty = (torch.cat((l2 * w[:-1], torch.zeros(1, device="cuda", dtype=w.dtype)))
                       if on_cuda else np.r_[l2 * w[:-1], 0])
            grad = x.T @ (p - y) / len(y) + penalty
            w -= cfg["detector"]["lr"] * grad
        if not (bool(torch.isfinite(w).all()) if on_cuda else np.isfinite(w).all()):
            raise RuntimeError("nonfinite fitted detector")
        detectors[name] = {
            "weights": w.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "training_role": "detector_fit",
            "n_rows": len(sub),
            "optimizer": "full-batch gradient descent",
            "parameters": cfg["detector"],
        }
    atomic_json(rr / "detectors.json", detectors)
    atomic_json(rr / "detector_fit_rows.json", rows)
    return evidence_record(
        cfg, "detector_fit", [rr / "detectors.json", rr / "detector_fit_rows.json"]
    )


def score_rows(rows, detectors, deadline_seconds=None):
    """Apply the deadline to transform/search/pooling/reference features/scoring."""
    for r in rows:
        start = time.perf_counter()
        d = detectors[r["method"]]
        x = (np.asarray(r["features"]) - d["mean"]) / d["std"]
        value = float(x @ np.asarray(d["weights"][:-1]) + d["weights"][-1])
        r["score_s"] = time.perf_counter() - start
        r["total_acceptance_s"] = r["decode_s"] + r["score_s"]
        if not np.isfinite(value):
            raise RuntimeError("nonfinite detector score")
        if deadline_seconds is not None and r["total_acceptance_s"] > deadline_seconds:
            r["timeout"] = True
            r["er"] = 0
            r["valid_for_acceptance"] = False
        # A timed-out request is a deterministic rejection and remains in the denominator.
        r["score"] = value if r["valid_for_acceptance"] else None
    return rows


def null_maxima(rows, method):
    grouped = {}
    for r in rows:
        if r["method"] == method and r["label"] == 0:
            grouped.setdefault(r["source_group"], []).append(
                -np.inf if r["score"] is None else r["score"]
            )
    return {k: float(max(v)) for k, v in grouped.items()}


def design_power(cfg):
    require_stage(cfg, "develop")
    require_stage(cfg, "controls_audit")
    rr = root(cfg)
    e = cfg["evaluation"]
    s = read_json(rr / "selected.json")
    p = (
        rr
        / "models"
        / f"eta_{s['eta']:g}_seed_{s['canonical_seed']}"
        / "validation_rows.json"
    )
    rows = read_json(p)
    planned = []
    for h, metric, a, b, key in CONTRASTS:
        aa = source_values(rows, metric, a, e["primary_areas"])
        bb = source_values(rows, metric, b, e["primary_areas"])
        if set(aa) != set(bb):
            raise RuntimeError("unpaired pilot sources")
        diff = np.array([aa[k] - bb[k] for k in sorted(aa)])
        if len(diff) < 2 or e[key] is None:
            raise RuntimeError(f"pilot >=2 and independently justified {key} required")
        sd = float(diff.std(ddof=1))
        # Zero pilot variance must not certify zero population variance.
        conservative_sd = max(sd, 0.25 if metric == "er" else 0.1)
        n = planning_n(
            conservative_sd, e[key], e["power_target"], e["effect_family_alpha"],
            comparisons=len(CONTRASTS)
        )
        planned.append(
            {
                "contrast": h,
                "pilot_sources": len(diff),
                "pilot_sd": sd,
                "planning_sd": conservative_sd,
                "assumed_true_effect": planning_effect(e[key]),
                "margin": e[key],
                "recommended_n": n,
                "note": "normal planning scenario with regularization floor, not a conservative variance bound or power guarantee",
            }
        )
    fpr = []
    for n in (1000, 3000, 5000, 10000, 20000):
        fpr.append(
            exact_fpr_power(
                n,
                e["assumed_fpr"],
                e["alpha_test"],
                e["fpr_family_alpha"] / len(METHODS),
            )
        )
    locked_n = e["test_n"]
    fpr_at_n = (
        exact_fpr_power(
            locked_n,
            e["assumed_fpr"],
            e["alpha_test"],
            e["fpr_family_alpha"] / len(METHODS),
        )
        if locked_n is not None
        else None
    )
    ready = (
        locked_n is not None
        and locked_n >= max(r["recommended_n"] for r in planned)
        and fpr_at_n["power"] >= e["power_target"]
    )
    ncal = read_json(rr / "data_audit.json")["counts"]["calibration"]
    kcal = math.ceil((ncal + 1) * (1 - e["alpha_cal"]))
    calibration_resolution = {
        "n": ncal,
        "k": kcal,
        "finite_threshold_possible": ncal > 0 and kcal <= ncal,
    }
    ready = ready and calibration_resolution["finite_threshold_possible"]
    output = {
        "ready": ready,
        "calibration_resolution": calibration_resolution,
        "effect_planning": planned,
        "fpr_scenarios": fpr,
        "fpr_at_locked_n": fpr_at_n,
        "assumed_fpr_is_design_assumption": True,
        "sample_size_not_performance_guarantee": True,
    }
    atomic_json(rr / "power_design.json", output)
    return evidence_record(
        cfg,
        "power",
        [rr / "power_design.json"],
        status="TECH_PASS" if ready else "NOT_READY",
    )


def tracked_files(cfg):
    rr = root(cfg)
    # Both configuration and method code are part of the immutable protocol.
    src = Path(__file__).resolve().parent
    files = list(src.glob("*.py"))
    files += [
        rr / x
        for x in (
            "sources.csv",
            "messages.json",
            "data_audit.json",
            "selected.json",
            "development_report.json",
            "detectors.json",
            "power_design.json",
            "preflight.json",
            "decisions.json",
            "controls_audit.json",
        )
    ]
    s = read_json(rr / "selected.json")
    files += [
        Path(s["checkpoint"]),
        Path(s["histogram"]),
        rr / "memory_profile.json",
        rr / "precision_equivalence.json",
        rr / "runtime_summary.json",
        rr / "design_lock.json",
    ]
    return {str(p.resolve()): sha(p) for p in files}


def freeze(cfg):
    rr = root(cfg)
    e = cfg["evaluation"]
    if cfg["study"]["mode"] == "diagnostic":
        raise RuntimeError(
            "diagnostic/synthetic runs cannot produce a confirmatory freeze"
        )
    if (rr / "freeze.json").exists():
        raise RuntimeError("protocol already frozen")
    verify_design(cfg)
    unresolved = [
        k
        for k in (
            "margin_embed",
            "margin_decode",
            "margin_joint",
            "test_n",
            "external_test_n",
            "min_psnr",
            "min_ssim",
            "max_decode_seconds",
            "quality_gap_db",
        )
        if e[k] is None
    ]
    if unresolved:
        raise RuntimeError(f"unlocked scientific requirements: {unresolved}")
    for p in ("development_report.json", "power_design.json", "preflight.json"):
        if not read_json(rr / p).get("ready", False):
            raise RuntimeError(f"{p} gate not ready")
    audit = read_json(rr / "data_audit.json")
    for role, n in (("test", e["test_n"]), ("external_test", e["external_test_n"])):
        if audit["counts"][role] != n:
            raise RuntimeError(
                f"{role} requires exactly locked n independent representatives"
            )
    for stage in (
        "prepare",
        "math-check",
        "preflight",
        "memory_profile",
        "oracle_diagnostics",
        "develop",
        "controls_audit",
        "detector_fit",
        "power",
        "comparator",
        "quality_match",
        "runtime",
    ):
        require_stage(cfg, stage)
    # External comparison and similar-quality evidence must be actual versioned inputs.
    gate = read_json(rr / "scope_and_quality_gate.json")
    if (
        gate.get("matched_quality_performance_claim") is not False
        or gate.get("complete_external_method_superiority_claim") is not False
    ):
        raise RuntimeError(
            "unsupported expanded claims require a new implementation/protocol"
        )
    if (
        not gate.get("ready")
        or not gate.get("quality_matched_report")
        or not gate.get("comparator_report")
    ):
        raise RuntimeError("quality-matched/comparator scope evidence missing")
    extras = {}
    for key in ("quality_matched_report", "comparator_report"):
        p = Path(gate[key])
        extras[str(p.resolve())] = sha(p)
    config_path = rr / "resolved_config.json"
    atomic_json(config_path, cfg)
    hashes = tracked_files(cfg)
    hashes.update(extras)
    hashes[str((rr / "scope_and_quality_gate.json").resolve())] = sha(
        rr / "scope_and_quality_gate.json"
    )
    hashes[str(config_path.resolve())] = sha(config_path)
    atomic_json(
        rr / "freeze.json",
        {
            "protocol": cfg["study"],
            "files": hashes,
            "config": cfg,
            "test_access_before_freeze": assert_no_final_outcome_access(cfg),
            "frozen_at": time.time(),
        },
    )
    return evidence_record(cfg, "freeze", [rr / "freeze.json"])


def verify_freeze(cfg, seal=False):
    rr = root(cfg)
    f = read_json(rr / "freeze.json")
    if cfg["study"]["mode"] != "diagnostic":
        verify_design(cfg)
        require_stage(cfg, "freeze")
    if f["config"] != cfg:
        raise RuntimeError("configuration changed since freeze")
    for p, digest in f["files"].items():
        if not Path(p).is_file() or sha(p) != digest:
            raise RuntimeError(f"freeze invalid: {p}")
    if seal:
        require_stage(cfg, "calibrate")
        lock = read_json(rr / "threshold_seal.json")
        if lock["freeze_hash"] != sha(rr / "freeze.json") or lock[
            "threshold_hash"
        ] != sha(rr / "thresholds.json"):
            raise RuntimeError("threshold seal invalid")
        for p, digest in lock["calibration_files"].items():
            if not Path(p).is_file() or sha(p) != digest:
                raise RuntimeError("calibration artifact changed")
    return f


def calibrate(cfg):
    rr = root(cfg)
    verify_freeze(cfg)
    if (rr / "thresholds.json").exists():
        raise RuntimeError("calibration is already fixed")
    s, model, hist = selected_artifacts(cfg)
    manifest = make_bank(cfg, "calibration", s["eta"], include_null=True)
    rows = score_rows(
        rows_for_bank(cfg, "calibration", s["eta"], model, hist),
        read_json(rr / "detectors.json"),
        deadline_seconds=cfg["evaluation"]["max_decode_seconds"],
    )
    thresholds = {}
    for method in METHODS:
        null = null_maxima(rows, method)
        t, k = order_threshold(list(null.values()), cfg["evaluation"]["alpha_cal"])
        if np.isposinf(t):
            raise RuntimeError(
                "calibration n too small: finite threshold unavailable; reject-only is not readiness"
            )
        thresholds[method] = {
            "threshold": None if np.isneginf(t) else t,
            "threshold_kind": "negative_infinity" if np.isneginf(t) else "finite",
            "k": k,
            "n": len(null),
            "alpha_cal": cfg["evaluation"]["alpha_cal"],
            "event": "single assigned reference; any registered crop condition",
        }
    atomic_json(rr / "calibration_rows.json", rows)
    atomic_json(rr / "thresholds.json", thresholds)
    atomic_json(
        rr / "threshold_seal.json",
        {
            "freeze_hash": sha(rr / "freeze.json"),
            "threshold_hash": sha(rr / "thresholds.json"),
            "calibration_files": {
                str((rr / "calibration_rows.json").resolve()): sha(
                    rr / "calibration_rows.json"
                ),
                str(manifest.resolve()): sha(manifest),
            },
            "sealed_at": time.time(),
        },
    )
    return evidence_record(
        cfg, "calibrate", [rr / "thresholds.json", rr / "threshold_seal.json"]
    )


def apply_acceptance(rows, thresholds):
    for r in rows:
        t = thresholds[r["method"]]
        threshold = (
            -np.inf if t["threshold_kind"] == "negative_infinity" else t["threshold"]
        )
        r["accepted"] = bool(r["score"] is not None and r["score"] > threshold)
        r["accepted_er"] = bool(r["label"] == 1 and r["accepted"] and r["er"])
    return rows


def evaluate(cfg, role):
    if role not in ("test", "external_test"):
        raise ValueError("final role")
    rr = root(cfg)
    verify_freeze(cfg, seal=True)
    claim = rr / f"{role}.opened.json"
    if claim.exists():
        raise RuntimeError("final set already opened; no automatic rerun/overwrite")
    # Claim is written BEFORE feature extraction to preserve exposure on crashes.
    atomic_json(
        claim,
        {
            "opened_at": time.time(),
            "freeze_hash": sha(rr / "freeze.json"),
            "role": role,
            "status": "IN_PROGRESS",
        },
    )
    record_data_access(cfg, role, "FINAL_EVALUATION_STARTED", claim, outcomes_seen=True)
    s, model, hist = selected_artifacts(cfg)
    manifest = make_bank(cfg, role, s["eta"], include_null=True, allow_final=True)
    rows = apply_acceptance(
        score_rows(
            rows_for_bank(cfg, role, s["eta"], model, hist),
            read_json(rr / "detectors.json"),
            deadline_seconds=cfg["evaluation"]["max_decode_seconds"],
        ),
        read_json(rr / "thresholds.json"),
    )
    atomic_json(rr / f"{role}_rows.json", rows)
    atomic_json(
        claim,
        {
            "opened_at": read_json(claim)["opened_at"],
            "role": role,
            "status": "PREDICTIONS_LOCKED",
            "rows_hash": sha(rr / f"{role}_rows.json"),
            "freeze_hash": sha(rr / "freeze.json"),
        },
    )
    return evidence_record(
        cfg, f"evaluate_{role}", [rr / f"{role}_rows.json", claim, manifest]
    )


def analyze(cfg, role="test"):
    rr = root(cfg)
    verify_freeze(cfg, seal=True)
    require_stage(cfg, f"evaluate_{role}")
    require_stage(cfg, f"recount_{role}")
    p = rr / f"{role}_rows.json"
    rows = read_json(p)
    e = cfg["evaluation"]
    lock = read_json(rr / f"{role}.opened.json")
    if lock.get("rows_hash") != sha(p):
        raise RuntimeError("final predictions changed or never locked")
    contrast = []
    primary = e["primary_areas"]
    for h, metric, a, b, key in CONTRASTS:
        aa = source_values(rows, metric, a, primary)
        bb = source_values(rows, metric, b, primary)
        if set(aa) != set(bb):
            raise RuntimeError("unpaired results")
        d = [aa[k] - bb[k] for k in sorted(aa)]
        mean, low, high = bootstrap_difference(
            d, e["bootstrap_reps"], e["effect_family_alpha"],
            comparisons=len(CONTRASTS), seed=cfg["study"]["seed"]
        )
        contrast.append(
            {
                "id": h,
                "n_sources": len(d),
                "difference": mean,
                "low": low,
                "high": high,
                "margin": e[key],
                "verdict": verdict(low, high, e[key]),
                "confirmatory": role == "test",
            }
        )
    fpr = []
    for name in METHODS:
        grouped = {}
        for r in rows:
            if r["method"] == name and r["label"] == 0:
                grouped.setdefault(r["source_group"], []).append(r["accepted"])
        if any(len(v) != len(e["areas"]) for v in grouped.values()):
            raise RuntimeError("incomplete max-event panel")
        n = len(grouped)
        x = sum(any(v) for v in grouped.values())
        gamma = e["fpr_family_alpha"] / len(METHODS)
        upper, lower = cp_upper(x, n, gamma), cp_lower(x, n, gamma)
        fpr.append(
            {
                "method": name,
                "n_sources": n,
                "false_accept_sources": x,
                "fpr": x / n,
                "lower": lower,
                "upper": upper,
                "alpha_test": e["alpha_test"],
                "verdict": (
                    "VERIFIED"
                    if upper <= e["alpha_test"]
                    else ("EXCEEDED" if lower > e["alpha_test"] else "INCONCLUSIVE")
                ),
            }
        )
    summary = []
    for name in METHODS:
        marked = [r for r in rows if r["method"] == name and r["label"] == 1]
        for area in e["areas"]:
            sub = [r for r in marked if r["area"] == area]
            summary.append(
                {
                    "method": name,
                    "area": area,
                    "n_sources": len(sub),
                    "ER": float(np.mean([r["er"] for r in sub])),
                    "accepted_ER": float(np.mean([r["accepted_er"] for r in sub])),
                    "coverage": float(np.mean([r["coverage"] for r in sub])),
                    "phase_accuracy": float(np.mean([r["phase_correct"] for r in sub])),
                    "median_decode_s": float(np.median([r["decode_s"] for r in sub])),
                }
            )
    verdicts = {r["id"]: r["verdict"] for r in contrast}
    decode = [verdicts[f"H_A_decode_D{i}"] for i in (0, 1, 2, 4)]
    verdicts["H_A_decode"] = (
        "SUPPORTED"
        if all(x == "SUPPORTED" for x in decode)
        else (
            "MARGIN_NOT_MET"
            if any(x == "MARGIN_NOT_MET" for x in decode)
            else "INCONCLUSIVE"
        )
    )
    independent = read_json(rr / f"{role}_independent_recount.json")
    if not independent["ready"] or independent["raw_hash"] != sha(p):
        raise RuntimeError("independent recount must pass before hypothesis analysis")
    negative = (
        any(x != "SUPPORTED" for x in verdicts.values())
        or any(x["verdict"] != "VERIFIED" for x in fpr)
        or not quality_gate(cfg, rows)["ready"]
    )
    output = {
        "role": role,
        "contrasts": contrast,
        "fpr": fpr,
        "hypotheses": verdicts,
        "quality": quality_gate(cfg, rows),
        "summary": summary,
        "technical_status": "TECH_PASS",
        "scientific_status": (
            "DIAGNOSTIC_REVIEW_REQUIRED"
            if negative
            else "SUPPORTED_WITHIN_REGISTERED_SCOPE"
        ),
        "advance_allowed": not negative,
        "negative_results_must_be_preserved": True,
    }
    atomic_json(rr / f"{role}_analysis.json", output)
    write_csv(rr / f"{role}_summary.csv", summary)
    return evidence_record(
        cfg,
        f"analyze_{role}",
        [rr / f"{role}_analysis.json", rr / f"{role}_summary.csv"],
        status="FAILURE_ANALYSIS_REQUIRED" if negative else "TECH_PASS",
    )


def preflight(cfg):
    rr = root(cfg)
    env = environment()
    import os, subprocess, platform

    env["container_image_identity"] = os.environ.get("PAPER_A_CONTAINER_IMAGE")
    env["container_image_identity_required"] = False
    env["container_image_policy"] = "Excluded as a progression gate by user on 2026-10-25"
    try:
        env["nvidia_smi"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        env["nvidia_smi"] = None
    try:
        env["workspace_mount"] = subprocess.check_output(
            ["findmnt", "-J", "-T", "/workspace"], text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        env["workspace_mount"] = None
    ready = (
        env["cuda_available"]
        and env["device_count"] == 1
        and "4090" in env.get("gpu", "")
    )
    if ready:
        # Verify real CUDA computation, not merely an installed wheel.
        x = torch.arange(256, device="cuda", dtype=torch.float32)
        if float((x * x).sum()) != float((np.arange(256, dtype=np.float32) ** 2).sum()):
            raise RuntimeError("CUDA smoke mismatch")
        total = env["total_bytes"]
        free = env["free_bytes"]
        ready = total >= 23 * 1024**3 and free > 0.15 * total
    ready = bool(
        ready
        and env["nvidia_smi"]
        and env["workspace_mount"]
        and platform.system() == "Linux"
    )
    disk = shutil.disk_usage(
        cfg["paths"]["dataset_root"]
        if Path(cfg["paths"]["dataset_root"]).exists()
        else rr
    )
    env.update(
        ready=bool(ready),
        disk_free_bytes=disk.free,
        mount_note="Run df -h/findmnt in launch_runpod.sh; /workspace name alone does not establish persistence",
    )
    atomic_json(rr / "preflight.json", env)
    return evidence_record(
        cfg,
        "preflight",
        [rr / "preflight.json"],
        status="TECH_PASS" if ready else "BLOCKED_GPU",
    )


def lock_design(cfg):
    """F08: lock the selected decoder and final fitting/calibration/evaluation rules before detector fit."""
    rr = root(cfg)
    if cfg["study"]["mode"] == "diagnostic":
        raise RuntimeError("diagnostic mode cannot lock a confirmatory design")
    if (rr / "design_lock.json").exists():
        raise RuntimeError("design already locked")
    assert_no_final_outcome_access(cfg)
    for stage in (
        "prepare",
        "math-check",
        "preflight",
        "memory_profile",
        "oracle_diagnostics",
        "develop",
        "controls_audit",
        "power",
        "comparator",
        "quality_match",
    ):
        require_stage(cfg, stage)
    e = cfg["evaluation"]
    for key in (
        "margin_embed",
        "margin_decode",
        "margin_joint",
        "test_n",
        "external_test_n",
        "min_psnr",
        "min_ssim",
        "max_decode_seconds",
        "quality_gap_db",
    ):
        if e[key] is None:
            raise RuntimeError(f"unlocked requirement: {key}")
    from .decisions import validate_decisions
    decision_record, decision_evidence = validate_decisions(cfg)
    if not read_json(rr / "power_design.json")["ready"]:
        raise RuntimeError("power/calibration resolution not ready")
    audit = read_json(rr / "data_audit.json")
    for role, n in (("test", e["test_n"]), ("external_test", e["external_test_n"])):
        if audit["counts"][role] != n:
            raise RuntimeError("locked sample count mismatch")
    selected = read_json(rr / "selected.json")
    artifacts = [
        rr / name
        for name in (
            "sources.csv",
            "messages.json",
            "data_audit.json",
            "selected.json",
            "development_report.json",
            "power_design.json",
            "scope_and_quality_gate.json",
            "quality_matched_report.json",
            "comparator_report.md",
        )
    ]
    artifacts.extend([rr / "decisions.json", rr / "controls_audit.json", *decision_evidence])
    artifacts.extend([Path(selected["checkpoint"]), Path(selected["histogram"])])
    artifacts.extend(Path(__file__).resolve().parent.glob("*.py"))
    atomic_json(
        rr / "design_lock.json",
        {
            "config": cfg,
            "files": {str(p.resolve()): sha(p) for p in artifacts},
            "status": "READY_FOR_FROZEN_FIT",
            "locked_at": time.time(),
            "runtime_signature": {
                k: environment()[k]
                for k in (
                    "python",
                    "torch",
                    "cuda_runtime",
                    "cudnn",
                    "dependency_versions",
                    "torch_threads",
                    "opencv_threads",
                )
            },
            "rules": "decoder/eta already selected; detector_fit/calibration algorithms fixed; no final data outcome access",
        },
    )
    return evidence_record(cfg, "lock_design", [rr / "design_lock.json"])


def verify_design(cfg):
    require_stage(cfg, "lock_design")
    rr = root(cfg)
    lock = read_json(rr / "design_lock.json")
    if lock["config"] != cfg:
        raise RuntimeError("design configuration changed")
    current = environment()
    if lock["runtime_signature"] != {k: current[k] for k in lock["runtime_signature"]}:
        raise RuntimeError("runtime dependency signature changed")
    for path, digest in lock["files"].items():
        if not Path(path).is_file() or sha(path) != digest:
            raise RuntimeError("design lock artifact changed")
    return lock


def controls_audit(cfg):
    """Validate identical observations and train-only affine state before scientific lock."""
    require_stage(cfg, 'develop')
    rr = root(cfg)
    selected, model, hist = selected_artifacts(cfg)
    reports = read_json(rr/'development_report.json')['results']
    by_eta = {}
    audited = []
    artifacts = []
    for report in tqdm(reports, desc='same-bank learned-control audit', file=sys.stdout):
        candidate, saved = load_model(report['checkpoint'], device_for(cfg))
        if saved.get('config') != cfg:
            raise RuntimeError('control checkpoint/config mismatch')
        train_manifest = bank_dir(cfg, 'train', report['eta'])/'manifest.json'
        if saved.get('train_manifest') != sha(train_manifest):
            raise RuntimeError('control train manifest changed')
        affine = candidate._uniform_affine
        if affine is None or affine != report.get('uniform_affine'):
            raise RuntimeError('D4 checkpoint/report mismatch')
        parameters = (affine['temperature'], affine['bias'])
        previous = by_eta.setdefault(report['eta'],parameters)
        if not np.allclose(parameters,previous,rtol=0,atol=1e-12):
            raise RuntimeError('deterministic train-only D4 differs across D3 seeds')
        rows = read_json(Path(report['checkpoint']).parent/'validation_rows.json')
        paired = {}
        for row in rows:
            if row['method'].startswith('E2'):
                key=(row['source_id'],row['condition'],row['label'])
                paired.setdefault(key,[]).append(row)
        for group in paired.values():
            if {r['method'] for r in group} != {'E2D0','E2D1','E2D2','E2D3','E2D4'} or len(group)!=5:
                raise RuntimeError('missing/duplicate same-bank decoder control')
            for field in ['observation_bank_hash','message','offset_x','offset_y','coverage']:
                if len({r[field] for r in group})!=1:
                    raise RuntimeError('unequal decoder information: '+field)
        if not paired:
            raise RuntimeError('no validation panel for control audit')
        artifacts.extend([Path(report['checkpoint']), train_manifest,
                          Path(report['checkpoint']).parent/'validation_rows.json'])
        audited.append({'eta':report['eta'],'seed':report['seed'],
            'cases':len(paired),'checkpoint_hash':sha(report['checkpoint']),
            'D4_training_healthy':True,'identical_blind_observations':True})
    output={'ready':True,'methods':list(METHODS),'contrasts':[c[0] for c in CONTRASTS],
            'audited':audited,'test_outcomes_used':False}
    atomic_json(rr/'controls_audit.json',output)
    return evidence_record(cfg,'controls_audit',[rr/'controls_audit.json', *artifacts])
