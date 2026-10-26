"""Strict configuration, atomic artifacts, hashes, and terminal-visible events."""

from __future__ import annotations
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import PurePosixPath
from pathlib import Path
from datetime import datetime, timezone
import numpy as np


def log(message, **fields):
    print(
        json.dumps(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "message": message,
                **fields,
            },
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def stable_seed(*values):
    return int.from_bytes(
        hashlib.sha256("|".join(map(str, values)).encode()).digest()[:8], "big"
    )


# Exact schemas prevent unnoticed misspellings and unused settings.
SCHEMA = {
    "study": {"id", "protocol_version", "mode", "seed"},
    "paths": {"dataset_root", "run_root", "input_csv"},
    "codec": {
        "delta",
        "eta_candidates",
        "tile_side",
        "payload_bits",
        "pilot_count",
        "pattern_seed",
        "swap_details",
    },
    "data": {"max_side", "min_side", "message_pool_size", "near_duplicate_distance"},
    "evaluation": {
        "areas",
        "primary_areas",
        "bootstrap_reps",
        "effect_family_alpha",
        "fpr_family_alpha",
        "alpha_test",
        "alpha_cal",
        "test_n",
        "external_test_n",
        "power_target",
        "assumed_fpr",
        "margin_embed",
        "margin_decode",
        "margin_joint",
        "min_psnr",
        "min_ssim",
        "max_decode_seconds",
        "quality_gap_db",
    },
    "training": {
        "seeds",
        "epochs",
        "lr",
        "weight_decay",
        "betas",
        "eps",
        "micro_batch",
        "accumulation_steps",
        "max_observations",
        "clip_grad_norm",
        "device",
        "amp",
        "hist_bins",
        "hist_smoothing",
        "affine_l2",
        "affine_max_iter",
        "affine_gtol",
        "affine_ftol",
        "max_trials",
        "early_stop_patience",
    },
    "detector": {"lr", "l2", "steps"},
}


def config(path):
    cfg = read_json(path)
    if set(cfg) != set(SCHEMA):
        raise ValueError(f"Unknown/missing config sections: {set(cfg) ^ set(SCHEMA)}")
    for group, keys in SCHEMA.items():
        if set(cfg[group]) != keys:
            raise ValueError(f"{group}: unknown/missing keys {set(cfg[group]) ^ keys}")
    if cfg["study"]["mode"] not in ("development", "diagnostic"):
        raise ValueError("mode must be development or diagnostic")
    c, t, e, d = cfg["codec"], cfg["training"], cfg["evaluation"], cfg["data"]
    for key in ("id", "protocol_version"):
        if not isinstance(cfg["study"][key], str) or not cfg["study"][key].strip():
            raise ValueError(f"nonblank study {key} required")
    for key, value in cfg["paths"].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"nonblank path required: {key}")
    for value in (cfg["study"]["seed"], c["pattern_seed"]):
        if type(value) is not int or not 0 <= value < 2**32:
            raise ValueError("study/pattern seed must be uint32 integer")
    if type(c["swap_details"]) is not bool or type(t["amp"]) is not bool:
        raise ValueError("swap_details and amp must be booleans")
    if not isinstance(c["eta_candidates"], list) or not c["eta_candidates"]:
        raise ValueError("at least one eta candidate required")
    if not isinstance(e["primary_areas"], list) or not e["primary_areas"]:
        raise ValueError("nonempty primary area panel required")
    if len(set(e["primary_areas"])) != len(e["primary_areas"]):
        raise ValueError("duplicate primary areas")
    if cfg["study"]["mode"] != "diagnostic" and (
        set(e["primary_areas"]) != {0.8, 0.5}
        or set(e["areas"]) != {1.0, 0.8, 0.5, 0.25}
    ):
        raise ValueError("production must preserve registered primary/range crop panels")

    if (c["tile_side"], c["payload_bits"], c["pilot_count"]) != (8, 48, 16):
        raise ValueError("This implementation requires the registered 48+16 layout")
    positive = [
        c["delta"],
        *c["eta_candidates"],
        t["epochs"],
        t["lr"],
        t["eps"],
        t["micro_batch"],
        t["accumulation_steps"],
        t["max_observations"],
        t["clip_grad_norm"],
        t["hist_bins"],
        t["hist_smoothing"],
        t["affine_l2"],
        t["affine_max_iter"],
        t["affine_gtol"],
        t["affine_ftol"],
        t["max_trials"],
        t["early_stop_patience"],
        e["bootstrap_reps"],
        d["max_side"],
        d["min_side"],
        d["message_pool_size"],
        cfg["detector"]["lr"],
        cfg["detector"]["steps"],
    ]
    if not all(
        type(x) in (int, float) and math.isfinite(x) and x > 0 for x in positive
    ):
        raise ValueError("positive, finite configuration value required")
    for key in (
        "epochs",
        "affine_max_iter",
        "micro_batch",
        "accumulation_steps",
        "max_observations",
        "hist_bins",
        "max_trials",
        "early_stop_patience",
    ):
        if type(t[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    if (
        not t["seeds"]
        or len(set(t["seeds"])) != len(t["seeds"])
        or any(type(x) is not int or not 0 <= x < 2**32 for x in t["seeds"])
    ):
        raise ValueError("distinct integer training seeds required")
    if (
        len(c["eta_candidates"]) != len(set(c["eta_candidates"]))
        or len(c["eta_candidates"]) > t["max_trials"]
    ):
        raise ValueError("eta candidates exceed registered development budget")
    if t["device"] not in ("cuda", "cpu") or t["amp"]:
        raise ValueError("Supported precision contract is FP32 only; AMP must be false")
    if (
        len(t["betas"]) != 2
        or not all(0 <= x < 1 for x in t["betas"])
        or t["weight_decay"] < 0
    ):
        raise ValueError("invalid AdamW settings")
    if not 0 <= d["near_duplicate_distance"] <= 8:
        raise ValueError("near_duplicate_distance must be between 0 and 8")
    if not set(e["primary_areas"]).issubset(e["areas"]) or len(set(e["areas"])) != len(
        e["areas"]
    ):
        raise ValueError("invalid area panel")
    if 1.0 not in e["areas"] or not all(0 < x <= 1 for x in e["areas"]):
        raise ValueError("areas must include none=1 and stay in (0,1]")
    for k in (
        "effect_family_alpha",
        "fpr_family_alpha",
        "alpha_test",
        "alpha_cal",
        "power_target",
    ):
        if type(e[k]) not in (int, float) or not 0 < e[k] < 1:
            raise ValueError(k)
    if (
        e["alpha_cal"] >= e["alpha_test"]
        or e["effect_family_alpha"] + e["fpr_family_alpha"] > 0.0500000001
    ):
        raise ValueError("invalid calibration/error-budget relation")
    for k in (
        "margin_embed",
        "margin_decode",
        "margin_joint",
        "min_psnr",
        "min_ssim",
        "max_decode_seconds",
        "quality_gap_db",
        "test_n",
        "external_test_n",
    ):
        v = e[k]
        if v is not None and (
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
        ):
            raise ValueError(f"invalid {k}")
    if (type(e["assumed_fpr"]) not in (int, float)
        or not 0 <= e["assumed_fpr"] <= 1 or d["min_side"] > d["max_side"]):
        raise ValueError("invalid probability/image bounds")
    if (
        not math.isfinite(t["weight_decay"])
        or t["weight_decay"] < 0
        or not math.isfinite(cfg["detector"]["l2"])
        or cfg["detector"]["l2"] < 0
    ):
        raise ValueError("invalid regularization")
    for key in ("max_side", "min_side", "message_pool_size"):
        if type(d[key]) is not int:
            raise ValueError("integer data dimensions required")
    if (
        type(e["bootstrap_reps"]) is not int
        or type(cfg["detector"]["steps"]) is not int
    ):
        raise ValueError("integer bootstrap/optimizer iteration counts required")
    for key in ("test_n", "external_test_n"):
        if e[key] is not None and (type(e[key]) is not int or e[key] < 2):
            raise ValueError("independent sample counts must be integer >=2")
    for key in ("margin_embed", "margin_decode", "margin_joint"):
        if e[key] is not None and e[key] >= 1:
            raise ValueError("minimum effect margin must be below 1")
    if e["min_ssim"] is not None and e["min_ssim"] > 1:
        raise ValueError("SSIM threshold exceeds 1")
    if type(d["near_duplicate_distance"]) is not int:
        raise ValueError("integer Hamming distance required")
    if cfg["study"]["mode"] != "diagnostic":
        for p in cfg["paths"].values():
            if not str(p).startswith("/workspace/") or ".." in PurePosixPath(p).parts:
                raise ValueError("RunPod durable paths must be beneath /workspace/")
    return cfg


def root(cfg):
    p = Path(cfg["paths"]["run_root"])
    p.mkdir(parents=True, exist_ok=True)
    return p



def refresh_progress(cfg, state):
    """Suggest prerequisite-eligible checks; each command still validates its own inputs."""
    dependencies = {
        "math-check":set(), "prepare":set(), "preflight":set(),
        "cache":{"prepare","math-check"},
        "memory_profile":{"prepare","math-check","preflight"},
        "pilot":{"prepare","math-check","memory_profile"},
        "develop":{"prepare","math-check","preflight","memory_profile"},
        "controls_audit":{"develop"}, "oracle_diagnostics":{"prepare","math-check"},
        "comparator":{"develop"}, "quality_match":{"develop","comparator"},
        "power":{"develop","controls_audit"},
        "lock_design":{"prepare","math-check","preflight","memory_profile","power",
                       "controls_audit","comparator","quality_match","oracle_diagnostics"},
        "detector_fit":{"lock_design","develop"}, "runtime":{"detector_fit"},
        "freeze":{"lock_design","detector_fit","runtime","power","controls_audit"},
        "calibrate":{"freeze"}, "evaluate_test":{"calibrate","freeze"},
        "evaluate_external_test":{"calibrate","freeze"},
        "recount_test":{"evaluate_test"}, "recount_external_test":{"evaluate_external_test"},
        "analyze_test":{"recount_test"}, "analyze_external_test":{"recount_external_test"}}
    digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    code=code_digest()
    def current(name, statuses=("TECH_PASS",)):
        entry=state["stages"].get(name,{})
        if entry.get("status") not in statuses or entry.get("config_hash")!=digest or entry.get("code_hash")!=code:
            return False
        return all(Path(p).is_file() and sha(p)==h for p,h in entry.get("artifacts",{}).items())
    production_only={"memory_profile","lock_design","freeze","calibrate",
                     "evaluate_test","evaluate_external_test"}
    candidates=[]
    for name, parents in dependencies.items():
        if cfg["study"]["mode"]=="diagnostic" and name in production_only:
            continue
        if not current(name) and all(current(p) for p in parents):
            candidates.append(name)
    state["allowed_next"]=candidates
    state["allowed_next_scope"]="necessary current-stage prerequisites only; each command enforces data, decisions and sealed artifacts"
    state["development_status"]="DEV_READY" if current("develop") else "NOT_ASSESSED"
    state["test_readiness"]=("READY_FOR_TEST" if cfg["study"]["mode"]!="diagnostic"
                             and current("freeze") and current("calibrate") else "NOT_READY")
    state["hypothesis_status"]={h:"NOT_EVALUATED" for h in ("H_A_embed","H_A_decode","H_A_joint")}
    if current("analyze_test", ("TECH_PASS","FAILURE_ANALYSIS_REQUIRED")) and (root(cfg)/"test_analysis.json").is_file():
        state["hypothesis_status"]=read_json(root(cfg)/"test_analysis.json")["hypotheses"]
    return state


def evidence_record(cfg, stage, artifacts, status="TECH_PASS", **extra):
    rr = root(cfg)
    statepath = rr / "experiment_state.json"
    state = (
        read_json(statepath)
        if statepath.exists()
        else {"stages": {}, "test_access": [], "history": []}
    )
    entry = {
        "status": status,
        "advance_allowed": status in ("TECH_PASS", "RESOLVED_NEGATIVE_RESULT"),
        "at": time.time(),
        "config_hash": hashlib.sha256(
            json.dumps(cfg, sort_keys=True).encode()
        ).hexdigest(),
        "artifacts": {str(Path(p).resolve()): sha(p) for p in artifacts},
        "code_hash": code_digest(),
        "command": list(sys.argv),
        **extra,
    }
    state.update(
        study_id=cfg["study"]["id"],
        protocol_version=cfg["study"]["protocol_version"],
        current_step=stage,
        last_status=status,
    )
    state["stages"][stage] = entry
    state["history"].append({"stage": stage, **entry})
    refresh_progress(cfg, state)
    atomic_json(statepath, state)
    report = rr / "stage_reports" / (stage + ".md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# " + stage + "\n\n" +
        "실행 상태: " + status + "\n시험 준비: " + state["test_readiness"] +
        "\n개발 상태: " + state["development_status"] +
        "\n\n실제 실행과 증거:\n\n~~~json\n" +
        json.dumps(entry, ensure_ascii=False, indent=2) + "\n~~~\n\n" +
        "다음 선행 검사를 확인할 단계: " + ", ".join(state["allowed_next"]) +
        "\n입력·봉인·판정 검사는 각 명령에서 추가로 수행한다. 기술 통과는 가설 지지를 뜻하지 않는다.\n", encoding="utf8")
    log("stage complete", stage=stage, status=status, artifacts=len(artifacts),
        allowed_next=state["allowed_next"], report=str(report))
    return entry


def require_artifacts(entry):
    for path, digest in entry["artifacts"].items():
        if not Path(path).is_file() or sha(path) != digest:
            raise RuntimeError(f"stale/missing artifact: {path}")


def environment():
    import torch, cv2, pywt, scipy, skimage, tqdm

    value = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "torch_threads": torch.get_num_threads(),
        "opencv_threads": cv2.getNumThreads(),
        "dependency_versions": {
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "pywavelets": pywt.__version__,
            "scipy": scipy.__version__,
            "scikit_image": skimage.__version__,
            "tqdm": tqdm.__version__,
        },
    }
    from .compute import cuda_enabled
    value["numerical_compute_backend"] = "cuda" if cuda_enabled() else "cpu_diagnostic_reference"
    value["production_cpu_fallback"] = False
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        value.update(gpu=prop.name, total_bytes=total, free_bytes=free)
    return value


def code_digest():
    files = sorted(Path(__file__).resolve().parent.glob("*.py"))
    return hashlib.sha256("".join(p.name + sha(p) for p in files).encode()).hexdigest()


def require_stage(cfg, name, allowed_statuses=("TECH_PASS",)):
    state = read_json(root(cfg) / "experiment_state.json")
    entry = state["stages"].get(name)
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    if (
        not entry
        or entry["status"] not in allowed_statuses
        or entry["config_hash"] != digest
        or entry.get("code_hash") != code_digest()
    ):
        raise RuntimeError(f"missing/stale prerequisite: {name}")
    require_artifacts(entry)
    return entry


def record_problem(cfg, stage, error):
    """Persist an observed execution problem without inventing a root cause."""
    rr = root(cfg)
    folder = rr / "evidence"
    folder.mkdir(parents=True, exist_ok=True)
    ledger = folder / "A_failure_ledger.csv"
    rows = read_csv(ledger) if ledger.exists() else []
    fields = [
        "failure_id",
        "step_id",
        "run_id",
        "original_observation",
        "trigger",
        "cause_class",
        "cause_status",
        "root_cause_evidence",
        "diagnostic_attempts",
        "modification",
        "old/new_hash",
        "regression_result",
        "development_result",
        "quality_result",
        "final_data_seen",
        "fresh_holdout_required",
        "acceptance_criteria",
        "stop_rule",
        "resolution_status",
        "advance_allowed",
    ]
    seen = any(
        (rr / f"{role}.opened.json").exists() for role in ("test", "external_test")
    )
    event = {
        "failure_id": f"F_{len(rows)+1:04d}",
        "step_id": stage,
        "run_id": cfg["study"]["id"],
        "original_observation": str(error),
        "trigger": type(error).__name__,
        "cause_class": (
            "input missing" if isinstance(error, FileNotFoundError) else "UNCLASSIFIED"
        ),
        "cause_status": "UNCONFIRMED",
        "root_cause_evidence": "terminal log; diagnosis required",
        "diagnostic_attempts": "[]",
        "modification": "",
        "old/new_hash": json.dumps({"current_code": code_digest()}),
        "regression_result": "",
        "development_result": "",
        "quality_result": "",
        "final_data_seen": seen,
        "fresh_holdout_required": "",
        "acceptance_criteria": "original registered gate",
        "stop_rule": "do not advance until validity resolved",
        "resolution_status": (
            "BLOCKED_DATA"
            if isinstance(error, FileNotFoundError)
            else "FAILURE_ANALYSIS_REQUIRED"
        ),
        "advance_allowed": False,
    }
    rows.append(event)
    write_csv(ledger, rows, fields)
    (folder / "A_failure_ledger.md").write_text(
        "# Observed execution problems\n\n"
        + "\n\n".join(
            f"{r['failure_id']} / {r['step_id']} / {r['resolution_status']}: {r['original_observation']}"
            for r in rows
        ),
        encoding="utf-8",
    )
    failed_stage = invalidate_failed_stage(cfg, stage)
    evidence_record(cfg, failed_stage, [ledger], status="FAIL")
    log(
        "problem preserved",
        failure_id=event["failure_id"],
        advance_allowed=False,
        final_data_seen=seen,
    )


def record_data_access(cfg, role, kind, artifact=None, outcomes_seen=False):
    """Record input-only auditing separately from exposure to final model outcomes."""
    rr = root(cfg)
    statepath = rr / "experiment_state.json"
    state = (
        read_json(statepath)
        if statepath.exists()
        else {"stages": {}, "test_access": [], "history": []}
    )
    event = {
        "at": time.time(),
        "role": role,
        "kind": kind,
        "outcomes_seen": bool(outcomes_seen),
        "artifact": str(artifact) if artifact is not None else None,
    }
    state.setdefault("data_access", []).append(event)
    if role in ("test", "external_test"):
        state.setdefault("test_access", []).append(event)
    atomic_json(statepath, state)
    return event


def assert_no_final_outcome_access(cfg):
    rr = root(cfg)
    path = rr / "experiment_state.json"
    state = read_json(path) if path.exists() else {}
    accesses = state.get("test_access", [])
    if any(event.get("outcomes_seen", False) for event in accesses) or any(
        (rr / f"{role}.opened.json").exists() for role in ("test", "external_test")
    ):
        raise RuntimeError(
            "final outcomes already exposed; do not re-freeze the same test"
        )
    return accesses


def invalidate_failed_stage(cfg, failed):
    """Preserve history but revoke this failure and all dependent current passes."""
    aliases={'profile':'memory_profile','controls-audit':'controls_audit',
             'quality-match':'quality_match','detector-fit':'detector_fit',
             'oracle-diagnostics':'oracle_diagnostics','lock-design':'lock_design'}
    failed=aliases.get(failed,failed)
    dependencies={
        'cache':{'prepare','math-check'},
        'memory_profile':{'prepare','math-check','preflight','cache'},
        'pilot':{'prepare','math-check','memory_profile'},
        'develop':{'prepare','math-check','preflight','memory_profile','cache'},
        'controls_audit':{'develop'},'oracle_diagnostics':{'prepare','math-check'},
        'comparator':{'develop'},'quality_match':{'develop','comparator'},
        'power':{'develop','controls_audit'},
        'lock_design':{'power','controls_audit','comparator','quality_match','oracle_diagnostics'},
        'detector_fit':{'lock_design','develop'},'runtime':{'detector_fit'},
        'freeze':{'lock_design','detector_fit','runtime','power','controls_audit'},
        'calibrate':{'freeze'},
        'evaluate_test':{'calibrate','freeze'},
        'evaluate_external_test':{'calibrate','freeze'},
        'recount_test':{'evaluate_test'},'recount_external_test':{'evaluate_external_test'},
        'analyze_test':{'recount_test'},'analyze_external_test':{'recount_external_test'},
        'resolve_negative_test':{'analyze_test'},'resolve_negative_external_test':{'analyze_external_test'},
        'audit':{'analyze_test','analyze_external_test','recount_test','recount_external_test'}}
    affected={failed}
    while True:
        newer=affected | {stage for stage,deps in dependencies.items() if deps & affected}
        if newer==affected: break
        affected=newer
    statepath=root(cfg)/'experiment_state.json'
    state=read_json(statepath) if statepath.exists() else {'stages':{},'history':[],'test_access':[]}
    for key, entry in state['stages'].items():
        canonical='cache' if key.startswith('cache_') else key
        if canonical in affected and entry.get('status')=='TECH_PASS':
            entry.update(status='INVALIDATED',advance_allowed=False,invalidated_by=failed)
    state['history'].append({'stage':failed,'status':'FAIL','invalidated':sorted(affected),'at':time.time()})
    state.setdefault('invalidated_stages',[]).append({'failure':failed,'stages':sorted(affected)})
    atomic_json(statepath,state)
    return failed
