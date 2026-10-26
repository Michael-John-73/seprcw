"""Independent raw-bit recount and controlled full-path timing."""

import sys
import time
import numpy as np
import torch
from scipy.optimize import brentq
from scipy.stats import binom
from tqdm import tqdm
from .io import *
from .data import *
from .codec import *
from .learning import predict, device_for
from .experiment import METHODS, selected_artifacts, verify_freeze


def validate_prediction_panel(cfg, role, rows, thresholds):
    """Check every scheduled source/method/condition/class and the strict decision."""
    scheduled = {r["source_id"]: r for r in sources(cfg, role)}
    if not scheduled:
        raise RuntimeError("no scheduled independent sources")
    expected = {
        (source_id, method, f"area_{area:g}", label)
        for source_id in scheduled
        for method in METHODS
        for area in cfg["evaluation"]["areas"]
        for label in (0, 1)
    }
    observed = set()
    messages = {sid: message_for(cfg, item)[0].tolist() for sid, item in scheduled.items()}
    for row in rows:
        key = (row["source_id"], row["method"], row["condition"], row["label"])
        if key not in expected or key in observed:
            raise RuntimeError("unexpected/duplicate prediction panel row")
        observed.add(key)
        origin = scheduled[row["source_id"]]
        if row["source_group"] != origin["source_group"] or row["role"] != role:
            raise RuntimeError("prediction source/role mismatch")
        if row["condition"] != f"area_{row['area']:g}":
            raise RuntimeError("prediction condition/area mismatch")
        predicted, reference = row["predicted_bits"], row["reference_bits"]
        if (len(predicted) != 48 or len(reference) != 48
                or any(type(v) is not int or v not in (-1, 0, 1) for v in predicted)
                or reference != messages[row["source_id"]]):
            raise RuntimeError("invalid prediction/reference bit vector")
        rule = thresholds[row["method"]]
        threshold = -np.inf if rule["threshold_kind"] == "negative_infinity" else rule["threshold"]
        accept = row["score"] is not None and row["score"] > threshold
        if row["accepted"] != bool(accept):
            raise RuntimeError("strict acceptance recount mismatch")
    if observed != expected:
        raise RuntimeError("missing scheduled prediction panel rows")
    return len(scheduled)


def independent_recount(cfg, role):
    rr = root(cfg)
    verify_freeze(cfg, seal=True)
    require_stage(cfg, f"evaluate_{role}")
    raw = rr / f"{role}_rows.json"
    lock = read_json(rr / f"{role}.opened.json")
    if lock.get("rows_hash") != sha(raw):
        raise RuntimeError("final predictions changed or were never locked")
    rows = read_json(raw)
    scheduled_n = validate_prediction_panel(cfg, role, rows, read_json(rr / "thresholds.json"))
    errors = []
    summary = []
    fpr = []
    for r in tqdm(rows, desc=f"independent bit recount {role}", file=sys.stdout):
        exact = int(r["predicted_bits"] == r["reference_bits"] and not r["timeout"])
        if r["er"] != exact or r["accepted_er"] != bool(
            r["label"] == 1 and r["accepted"] and exact
        ):
            errors.append(
                {
                    "source": r["source_id"],
                    "method": r["method"],
                    "issue": "raw bit/ER mismatch",
                }
            )
    for method in METHODS:
        for area in cfg["evaluation"]["areas"]:
            sub = [
                r
                for r in rows
                if r["method"] == method and r["area"] == area and r["label"] == 1
            ]
            if not sub:
                raise RuntimeError("missing recount condition")
            summary.append(
                {
                    "method": method,
                    "area": area,
                    "n": len(sub),
                    "ER": sum(
                        int(
                            r["predicted_bits"] == r["reference_bits"]
                            and not r["timeout"]
                        )
                        for r in sub
                    )
                    / len(sub),
                }
            )
        groups = {}
        for r in rows:
            if r["method"] == method and r["label"] == 0:
                groups[r["source_group"]] = (
                    groups.get(r["source_group"], False) or r["accepted"]
                )
        n = len(groups)
        x = sum(groups.values())
        gamma = cfg["evaluation"]["fpr_family_alpha"] / len(METHODS)
        upper = (
            1.0
            if x == n
            else brentq(lambda p: binom.cdf(x, n, p) - gamma, 0.0, 1.0, xtol=1e-13)
        )
        fpr.append({"method": method, "x": x, "n": n, "upper": upper})
    output = {
        "ready": not errors,
        "raw_hash": sha(raw),
        "scheduled_sources": scheduled_n,
        "complete_prediction_panel_verified": True,
        "strict_acceptance_independently_recounted": True,
        "errors": errors,
        "ER": summary,
        "FPR": fpr,
        "uses_primary_metric_functions": False,
    }
    atomic_json(rr / f"{role}_independent_recount.json", output)
    if errors:
        raise RuntimeError("independent recount mismatch")
    return evidence_record(
        cfg, f"recount_{role}", [rr / f"{role}_independent_recount.json"]
    )


def runtime_budget(records, limit):
    """Apply the registered deadline to measured full decode/score calls."""
    values = [float(r["total_decode_score_s"]) for r in records]
    ready = bool(values) and limit is not None and all(
        np.isfinite(x) and x <= limit for x in values
    )
    return {"ready": ready, "deadline_seconds": limit,
            "measured_calls": len(values),
            "maximum_seconds": max(values) if values else None,
            "scope": "transform/search/pooling/reference features/acceptance score; I/O excluded"}


def measure_runtime(cfg):
    """Measure both labels and every registered crop on the largest validation inputs."""
    rr = root(cfg)
    require_stage(cfg, "detector_fit")
    s, model, hist = selected_artifacts(cfg)
    cc = codec_config(cfg)
    detectors = read_json(rr / "detectors.json")
    records = []
    device = device_for(cfg)
    cases = sorted(sources(cfg, "validation"),
                   key=lambda row: (-int(row["width"])*int(row["height"]), row["source_id"]))[:8]
    if not cases:
        raise RuntimeError("validation timing inputs missing")
    total_calls = len(cases) * len(METHODS) * len(cfg["evaluation"]["areas"]) * 2 * 7
    with tqdm(total=total_calls, desc="full runtime calls", file=sys.stdout) as progress:
        for row in cases:
            image = processed_image(row["path"], cfg)
            bits, _ = message_for(cfg, row)
            for method, (rule, dec) in METHODS.items():
                marked, _ = png_roundtrip(embed(image, bits, cc, rule, s["eta"]))
                for label, received in ((0, image), (1, marked)):
                    for info, crop in crop_panel(received, cfg, row["source_id"]):
                        for repeat in range(7):
                            if device.type == "cuda":
                                torch.cuda.synchronize()
                            started = time.perf_counter()
                            bank = collect(crop, cc)
                            pred, ev = (predict(bank, model, device) if dec == "D3" else
                                        baseline(bank, dec, hist, getattr(model, "_uniform_affine", None)))
                            pooled = time.perf_counter()
                            x = detector_features(bank, ev, bits)
                            detector = detectors[method]
                            score = float(((x-detector["mean"])/detector["std"])
                                          @ detector["weights"][:-1] + detector["weights"][-1])
                            if not np.isfinite(score):
                                raise RuntimeError("nonfinite timed acceptance score")
                            if device.type == "cuda":
                                torch.cuda.synchronize()
                            ended = time.perf_counter()
                            if repeat >= 2:
                                records.append({"source_id":row["source_id"], "method":method,
                                    "condition":info["condition"], "area":info["area"], "label":label,
                                    "height":info["height"], "width":info["width"], "repeat":repeat-2,
                                    "search_s":bank["search_s"], "collect_s":bank["collect_s"],
                                    "pooling_s":pooled-started-bank["collect_s"],
                                    "score_s":ended-pooled, "total_decode_score_s":ended-started,
                                    "I_O_included":False})
                            progress.update(1)
    write_csv(rr / "runtime_rows.csv", records)
    atomic_json(rr / "runtime_environment.json", {**environment(), "warmup":2, "repeats":5,
        "cv2_threads":cv2.getNumThreads(), "torch_threads":torch.get_num_threads(),
        "I_O_included":False, "input_role":"validation", "areas":cfg["evaluation"]["areas"],
        "labels":[0,1], "source_selection":"up to 8 largest preprocessed pixel counts; source_id tie",
        "source_ids":[row["source_id"] for row in cases],
        "scope":"measured validation subset; not a worst-case latency guarantee"})
    summary = []
    for method in METHODS:
        values = [r["total_decode_score_s"] for r in records if r["method"] == method]
        summary.append({"method":method, "median_s":float(np.median(values)),
            "q25_s":float(np.quantile(values,.25)), "q75_s":float(np.quantile(values,.75)),
            "n_measurements":len(values)})
    atomic_json(rr / "runtime_summary.json", summary)
    budget_result = runtime_budget(records, cfg["evaluation"]["max_decode_seconds"])
    atomic_json(rr / "runtime_gate.json", budget_result)
    return evidence_record(cfg, "runtime", [rr/"runtime_rows.csv", rr/"runtime_environment.json",
        rr/"runtime_summary.json", rr/"runtime_gate.json"],
        status="TECH_PASS" if budget_result["ready"] else "NOT_READY")


def final_audit(cfg):
    verify_freeze(cfg, seal=True)
    rr = root(cfg)
    checks = []
    for role in ("test", "external_test"):
        p = rr / f"{role}_analysis.json"
        q = rr / f"{role}_independent_recount.json"
        if not p.exists() or not q.exists():
            checks.append({"role": role, "status": "NOT_RUN", "advance_allowed": False})
            continue
        require_stage(cfg, f"evaluate_{role}")
        require_stage(cfg, f"recount_{role}")
        require_stage(cfg, f"analyze_{role}", allowed_statuses=("TECH_PASS", "FAILURE_ANALYSIS_REQUIRED"))
        result, ind = read_json(p), read_json(q)
        if not ind["ready"] or ind["raw_hash"] != sha(rr / f"{role}_rows.json"):
            raise RuntimeError("stale/failed recount")
        for x in result["summary"]:
            y = next(
                y
                for y in ind["ER"]
                if y["method"] == x["method"] and y["area"] == x["area"]
            )
            if abs(x["ER"] - y["ER"]) > 1e-12:
                raise RuntimeError("ER recount disagreement")
        for x in result["fpr"]:
            y = next(y for y in ind["FPR"] if y["method"] == x["method"])
            if (
                x["false_accept_sources"] != y["x"]
                or abs(x["upper"] - y["upper"]) > 1e-9
            ):
                raise RuntimeError("FPR recount disagreement")
        checks.append(
            {
                "role": role,
                "technical_status": "TECH_PASS",
                "scientific_status": result["scientific_status"],
                "advance_allowed": result["advance_allowed"],
            }
        )
    for item in checks:
        role = item["role"]
        resolution = rr / f"{role}_negative_resolution.json"
        if not item.get("advance_allowed", False) and resolution.exists():
            require_stage(cfg, f"resolve_negative_{role}", allowed_statuses=("RESOLVED_NEGATIVE_RESULT",))
            r = read_json(resolution)
            if (
                r["status"] == "RESOLVED_NEGATIVE_RESULT"
                and r["raw_hash"] == sha(rr / f"{role}_rows.json")
                and r["analysis_hash"] == sha(rr / f"{role}_analysis.json")
            ):
                item["valid_negative_report_allowed"] = True
    ready = all(x.get("advance_allowed", False) for x in checks)
    reportable = all(
        x.get("advance_allowed", False) or x.get("valid_negative_report_allowed", False)
        for x in checks
    )
    atomic_json(
        rr / "final_audit.json",
        {
            "checks": checks,
            "ready_for_positive_manuscript": ready,
            "ready_for_evidence_based_manuscript": reportable,
            "negative_results_reportable_after_documented_diagnostic_resolution": True,
            "runtime_completed": (rr / "runtime_summary.json").exists(),
        },
    )
    return evidence_record(
        cfg,
        "audit",
        [rr / "final_audit.json"],
        status=(
            "TECH_PASS"
            if reportable and (rr / "runtime_summary.json").exists()
            else "REVIEW_REQUIRED"
        ),
    )


def resolve_negative(cfg, role):
    """Close valid negative/inconclusive results without changing data, thresholds, or criteria."""
    rr = root(cfg)
    verify_freeze(cfg, seal=True)
    require_stage(cfg, f"evaluate_{role}")
    require_stage(cfg, f"recount_{role}")
    require_stage(cfg, f"analyze_{role}", allowed_statuses=("FAILURE_ANALYSIS_REQUIRED",))
    analysis = read_json(rr / f"{role}_analysis.json")
    if analysis["scientific_status"] != "DIAGNOSTIC_REVIEW_REQUIRED":
        raise RuntimeError("no negative/inconclusive result requires this closure")
    recount = read_json(rr / f"{role}_independent_recount.json")
    rows = read_json(rr / f"{role}_rows.json")
    if recount["raw_hash"] != sha(rr / f"{role}_rows.json") or not recount["ready"]:
        raise RuntimeError("negative result validity unresolved")
    selected = read_json(rr / "selected.json")
    if not selected.get("training_healthy") or not analysis["quality"]["ready"]:
        raise RuntimeError(
            "unresolved learning/quality gate is not a valid negative-result closure"
        )
    breakdown = []
    for method in METHODS:
        marked = [r for r in rows if r["method"] == method and r["label"] == 1]
        breakdown.append(
            {
                "method": method,
                "cases": len(marked),
                "coverage_failures": sum(r["coverage"] < 1 for r in marked),
                "wrong_phase": sum(not r["phase_correct"] for r in marked),
                "right_phase_nonexact": sum(
                    r["phase_correct"] and not r["er"] for r in marked
                ),
                "timeouts": sum(r["timeout"] for r in marked),
                "note": "overlapping diagnostic events; no causal decomposition asserted",
            }
        )
    result = {
        "status": "RESOLVED_NEGATIVE_RESULT",
        "advance_allowed": True,
        "raw_hash": sha(rr / f"{role}_rows.json"),
        "analysis_hash": sha(rr / f"{role}_analysis.json"),
        "unchanged_hypotheses": analysis["hypotheses"],
        "diagnostics": breakdown,
        "causal_explanations": "WITHHELD where no intervention evidence exists",
        "permitted_claim": "report the observed supported/negative/inconclusive contrasts within fixed scope",
        "positive_claims_for_unsupported_effects": False,
    }
    atomic_json(rr / f"{role}_negative_resolution.json", result)
    return evidence_record(
        cfg,
        f"resolve_negative_{role}",
        [rr / f"{role}_negative_resolution.json"],
        status="RESOLVED_NEGATIVE_RESULT",
    )
