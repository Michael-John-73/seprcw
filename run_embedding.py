#!/usr/bin/env python3
"""Visible stage runner for VSCode integrated terminal: python -u run.py ..."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    subs = p.add_subparsers(dest="stage", required=True)
    for command in (
        "preflight",
        "math-check",
        "prepare",
        "pilot",
        "profile",
        "develop",
        "controls-audit",
        "detector-fit",
        "power",
        "comparator",
        "quality-match",
        "freeze",
        "calibrate",
        "runtime",
        "audit",
        "oracle-diagnostics",
        "lock-design",
    ):
        subs.add_parser(command)
    c = subs.add_parser("cache")
    c.add_argument(
        "--split", choices=["train", "validation", "detector_fit"], required=True
    )
    c.add_argument("--eta", type=float, required=True)
    for command in ("evaluate", "analyze", "recount", "resolve-negative"):
        s = subs.add_parser(command)
        s.add_argument("--split", choices=["test", "external_test"], default="test")
    a = p.parse_args()
    from paper_a.io import (
        config,
        log,
        root,
        atomic_json,
        evidence_record,
        record_problem,
    )

    cfg = config(a.config)
    from paper_a.compute import configure_compute
    configure_compute(cfg)
    if cfg["study"]["mode"] != "diagnostic":
        import cv2, torch
        cv2.setNumThreads(4)
        torch.set_num_threads(4)
    # Freeze actual resolved values at run creation; edits require a fresh run_root.
    resolved = root(cfg) / "resolved_config.json"
    if resolved.exists():
        import json
        if json.loads(resolved.read_text(encoding="utf8")) != cfg:
            raise RuntimeError("resolved configuration changed; create a fresh run_root")
    else:
        atomic_json(resolved, cfg)

    log("START", stage=a.stage, command=sys.argv, mode=cfg["study"]["mode"])
    from paper_a import experiment as e
    from paper_a import diagnostics as d
    from paper_a import audit as audit

    result = None
    try:
        if a.stage == "math-check":
            out = root(cfg) / "mathematical_checks.json"
            atomic_json(out, d.mathematical_checks())
            result = evidence_record(cfg, a.stage, [out])
        elif a.stage == "cache":
            out = e.make_bank(
                cfg, a.split, a.eta, include_null=a.split == "detector_fit"
            )
            result = evidence_record(cfg, f"cache_{a.split}_{a.eta:g}", [out])
        elif a.stage == "profile":
            result = d.memory_profile(cfg)
        elif a.stage in ("evaluate", "analyze"):
            result = getattr(e, a.stage)(cfg, a.split)
        elif a.stage == "resolve-negative":
            result = audit.resolve_negative(cfg, a.split)
        elif a.stage == "recount":
            result = audit.independent_recount(cfg, a.split)
        elif a.stage == "runtime":
            result = audit.measure_runtime(cfg)
        elif a.stage == "audit":
            result = audit.final_audit(cfg)
        else:
            fn = {
                "preflight": e.preflight,
                "prepare": e.prepare,
                "pilot": d.pilot,
                "develop": e.develop,
                "controls-audit": e.controls_audit,
                "detector-fit": e.fit_detector,
                "power": e.design_power,
                "comparator": d.scoped_comparator,
                "oracle-diagnostics": d.oracle_diagnostics,
                "lock-design": e.lock_design,
                "quality-match": d.quality_matched,
                "freeze": e.freeze,
                "calibrate": e.calibrate,
            }[a.stage]
            result = fn(cfg)
    except Exception as ex:
        failure_stage = (f"{a.stage.replace(chr(45), chr(95))}_{a.split}"
                         if a.stage in ("evaluate", "analyze", "recount", "resolve-negative") else a.stage)
        record_problem(cfg, failure_stage, ex)
        log(
            "BLOCKED/FAILED", stage=a.stage, error_type=type(ex).__name__, error=str(ex)
        )
        raise
    if isinstance(result, dict) and result.get("status") not in (
        None,
        "TECH_PASS",
        "RESOLVED_NEGATIVE_RESULT",
    ):
        log("STAGE_NOT_ADVANCEABLE", stage=a.stage, status=result["status"])
        raise SystemExit(3)
    log("END", stage=a.stage)


if __name__ == "__main__":
    main()
