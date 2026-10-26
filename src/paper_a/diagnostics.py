"""Diagnostic-only tests, realistic feature-memory profiling, scoped controls."""

from __future__ import annotations
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from .io import *
from .data import *
from .codec import *
from .learning import *
from .experiment import rows_for_bank, quality_gate, METHODS, selected_artifacts


def mathematical_checks():
    from .statistics import cp_upper, order_threshold, bootstrap_difference

    rng = np.random.default_rng(517)
    # Brute-force lattice definition independent of the projection implementation.
    for _ in tqdm(range(300), desc="independent lattice checks", file=sys.stdout):
        a = float(rng.uniform(0, 180))
        u = float(rng.uniform(0, 180))
        bit = int(rng.integers(2))
        delta = float(rng.uniform(0.5, 60))
        eta = float(rng.uniform(0.001, 5))
        for margin in (None, eta):
            got = float(project_magnitude(a, u, bit, delta, margin))
            qs = delta * (np.arange(1000) + 0.25 + 0.5 * bit)
            if margin is not None:
                qs = qs[qs >= u + margin]
            expected = qs[np.argmin((qs - a) ** 2)]
            np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10)
    # Independent coordinate trace for nonaligned raster cropping.
    for top, left, w in ((3, 13, 256), (8, 8, 512), (21, 64, 360)):
        py, px = (-top) % 8, (-left) % 8
        for i in range(3):
            for j in range(4):
                x0 = left + px + j * 8
                y0 = top + py + i * 8
                absolute = (y0 // 8) * (w // 8) + x0 // 8
                algebra = ((i + (top + py) // 8) * (w // 8) + j + (left + px) // 8) % 48
                assert absolute % 48 == algebra
    # Analytical softmax gradient and a central finite difference.
    a = torch.tensor([0.2, -0.3, 0.7], dtype=torch.float64, requires_grad=True)
    s = torch.tensor([-0.7, 0.3, 0.9], dtype=torch.float64)
    T = 1.7
    b = 0.8
    w = torch.softmax(a, 0)
    z = (w * s).sum()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        T * z, torch.tensor(b, dtype=torch.float64)
    )
    (grad,) = torch.autograd.grad(loss, a)
    expected = T * (torch.sigmoid(T * z) - b) * w * (s - z)
    np.testing.assert_allclose(grad.detach(), expected.detach(), rtol=1e-10, atol=1e-12)
    numeric = []
    for k in range(3):

        def value(delta):
            x = a.detach().clone()
            x[k] += delta
            return torch.nn.functional.binary_cross_entropy_with_logits(
                T * (torch.softmax(x, 0) * s).sum(),
                torch.tensor(b, dtype=torch.float64),
            ).item()

        numeric.append((value(1e-5) - value(-1e-5)) / (2e-5))
    np.testing.assert_allclose(grad.detach(), numeric, rtol=1e-6, atol=1e-9)
    assert abs(cp_upper(0, 600, 0.05 / 3) - (1 - (0.05 / 3) ** (1 / 600))) < 1e-12
    t, k = order_threshold([0.0, 1.0, 2.0, 3.0], 0.2)
    assert t == 3 and k == 4 and not (3 > t)
    assert np.isposinf(order_threshold([1.0], 0.001)[0])
    # Spectral alpha=0 against explicit pilot correlations: axis/sign independent.
    cfg = CodecConfig()
    soft = rng.normal(size=(11, 17))
    sums, _ = legacy._periodic_aggregate(soft, 8)
    brute = legacy._phase_tables(cfg)[1] @ sums
    np.testing.assert_allclose(
        spomf_scores(soft, cfg, 0), brute, atol=1e-10, rtol=1e-12
    )
    # Ideal perturbation sufficient condition (no uint8 channel claim).
    for _ in range(50):
        block = rng.normal(size=15) * 20
        idx = int(np.argmax(abs(block)))
        u = max(abs(np.delete(block, idx)))
        bit = int(rng.integers(2))
        delta, eta = 36.0, 3.0
        q = float(project_magnitude(abs(block[idx]), u, bit, delta, eta))
        block[idx] = q if block[idx] >= 0 else -q
        eps = 0.99 * min(eta / 2, delta / 4)
        noisy = block + rng.uniform(-eps, eps, size=15)
        assert np.argmax(abs(noisy)) == idx
        assert int(abs(noisy[idx]) % delta > 0.5 * delta) == bit
    return {
        "status": "TECH_PASS",
        "lattice_random_cases": 300,
        "gradient": "analytic/autodiff/finite_difference",
        "synthetic_only": True,
        "empirical_hypotheses_tested": False,
    }


def pilot(cfg):
    rr = root(cfg)
    require_stage(cfg, "prepare")
    require_stage(cfg, "math-check")
    if cfg["study"]["mode"] != "diagnostic":
        require_stage(cfg, "preflight")
        require_stage(cfg, "memory_profile")
    if (rr / "freeze.json").exists() or (rr / "design_lock.json").exists():
        raise RuntimeError("pilot cannot follow design lock/freeze")
    eta = cfg["codec"]["eta_candidates"][0]
    seed = cfg["training"]["seeds"][0]
    for role in ("train", "validation"):
        if not (bank_dir(cfg, role, eta) / "manifest.json").exists():
            make_bank(cfg, role, eta)
    report = train_one(cfg, eta, seed, rr / "pilot_model")
    model, _ = load_model(report["checkpoint"], device_for(cfg))
    rows = rows_for_bank(cfg, "validation", eta, model, np.load(report["histogram"]))
    atomic_json(rr / "pilot_validation_rows.json", rows)
    atomic_json(
        rr / "pilot_report.json",
        {
            "training": report,
            "roles": "train/validation only",
            "quality_not_yet_selected": quality_gate(cfg, rows),
            "main_test_ready": False,
            "note": "Set scientifically justified margins/quality/time and use a fresh main run_root; do not maximize pilot effects as margins",
        },
    )
    return evidence_record(
        cfg, "pilot", [rr / "pilot_report.json", rr / "pilot_validation_rows.json"]
    )


def _profile_training_batch(model, opt, sample, count, device, settings):
    """Keep temporary CUDA tensors inside one scope, including failed attempts."""
    effective = settings["micro_batch"] * settings["accumulation_steps"]

    def update():
        opt.zero_grad(set_to_none=True)
        for start in range(0, effective, count):
            backward_micro(
                model, [sample] * min(count, effective - start), device, effective
            )
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError("nonfinite profiling gradient")
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), settings["clip_grad_norm"], error_if_nonfinite=True
        )
        opt.step()

    with torch.inference_mode():
        f, s, i, _ = packed([sample] * count, device)
        reference = model(f, s, i, 48 * count)[0][:48].cpu().numpy().copy()
    del f, s, i
    for _ in range(2):
        update()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(3):
        update()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 3
    return {
        "micro_batch": count,
        "observations": count * len(sample["soft"]),
        "seconds": elapsed,
        "cases_per_second": effective / elapsed,
        "effective_cases_per_step": effective,
        "microbatches_per_step": (effective + count - 1) // count,
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "free_bytes_after": torch.cuda.mem_get_info()[0],
        "timing_scope": "one effective group: CPU packing, device transfer, accumulated "
                        "forward/backward, finite checks, clipping, AdamW step; excludes disk I/O",
    }, reference


def memory_profile(cfg):
    rr = root(cfg)
    require_stage(cfg, "prepare")
    require_stage(cfg, "math-check")
    require_stage(cfg, "preflight")
    if (rr / "design_lock.json").exists() or (rr / "freeze.json").exists():
        raise RuntimeError("profiling cannot change a locked design")
    device = device_for(cfg)
    if device.type != "cuda":
        raise RuntimeError("real CUDA memory profile unavailable; CPU is not a substitute")
    profiled_etas = [eta for eta in cfg["codec"]["eta_candidates"]
                     if (bank_dir(cfg, "train", eta)/"manifest.json").exists()]
    paths = [p for eta in profiled_etas for p in bank_paths(cfg, "train", eta, "E2")]
    if not paths:
        raise RuntimeError("profile needs real train banks")
    sample = max((load_bank(p) for p in tqdm(paths, desc="scan maximum train shape", file=sys.stdout)),
                 key=lambda b: len(b["soft"]))
    mean, std = feature_statistics(paths)
    model = ReliabilityDecoder(mean, std).to(device)
    # Validate arithmetic before choosing a memory-safe/fast batch. CPU comparison
    # uses the same weights but an independently executed CPU training update.
    from .numerics import cpu_cuda_equivalence
    equivalence = cpu_cuda_equivalence(model, sample, cfg["training"])
    atomic_json(rr / "precision_equivalence.json", equivalence)
    if not equivalence["passed"]:
        raise RuntimeError("CPU/CUDA training arithmetic mismatch; see precision_equivalence.json")
    log("CPU/CUDA training equivalence", passed=True,
        hard_bit_mismatches=equivalence["hard_bit_mismatches"])
    settings = cfg["training"]
    initial_state = copy.deepcopy(model.state_dict())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = []
    base = None
    free, total = torch.cuda.mem_get_info()
    budget = min(0.85 * total, free - 0.15 * total)
    effective = settings["micro_batch"] * settings["accumulation_steps"]
    candidates = sorted({1, effective} | {2**power for power in range(effective.bit_length())
                                         if 2**power <= effective})
    for count in tqdm(candidates, desc="4090 batch profile", file=sys.stdout):
        observations = count * len(sample["soft"])
        if observations > cfg["training"]["max_observations"]:
            results.append({"micro_batch": count, "safe": False, "reason": "observation cap"})
            continue
        try:
            model.load_state_dict(initial_state)
            opt = torch.optim.AdamW(
                model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"],
                betas=tuple(settings["betas"]), eps=settings["eps"],
            )
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            row, first = _profile_training_batch(model, opt, sample, count, device, settings)
            if base is None:
                base = first
            error = float(np.max(abs(first - base)))
            row.update(
                max_logit_error=error,
                safe=bool(
                    row["peak_allocated"] <= budget
                    and row["peak_reserved"] <= 0.85 * total
                    and row["free_bytes_after"] >= 0.15 * total
                    and np.isfinite(first).all()
                    and error <= 1e-5
                    and np.array_equal(first >= 0, base >= 0)
                ),
            )
            results.append(row)
        except torch.cuda.OutOfMemoryError:
            results.append({"micro_batch": count, "safe": False, "reason": "OOM"})
        # Leave the exception scope first so its traceback no longer owns tensors.
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        log("profile candidate", **results[-1])
    safe = [x for x in results if x["safe"]]
    output = {
        "ready": bool(safe),
        "profiled_etas": profiled_etas,
        "maximum_case_observations": len(sample["soft"]),
        "training_manifest_hashes": {str((bank_dir(cfg,"train",eta)/"manifest.json").resolve()):
                                    sha(bank_dir(cfg,"train",eta)/"manifest.json") for eta in profiled_etas},
        "results": results,
        "configured_micro_batch": cfg["training"]["micro_batch"],
        "target_effective_cases": effective,
        "candidate_micro_batches": candidates,
        "selection_objective": "measured end-to-end cases/s, not VRAM occupancy",
        "recommended_micro_batch": max(safe, key=lambda x: x["cases_per_second"])["micro_batch"] if safe else None,
        "total_bytes": total,
        "initial_free_bytes": free,
        "budget_bytes": budget,
        "amp": False,
        "tf32": False,
        "scope": "real training features; no test exposure",
        "precision_equivalence": equivalence,
    }
    atomic_json(rr / "memory_profile.json", output)
    return evidence_record(cfg, "memory_profile", [rr / "memory_profile.json", rr / "precision_equivalence.json",
                               *[bank_dir(cfg,"train",eta)/"manifest.json" for eta in profiled_etas]],
                           status="TECH_PASS" if output["ready"] else "NOT_READY")


def scoped_comparator(cfg):
    """Same-carrier SPOMF-inspired synchronization module. Never full-paper reproduction."""
    rr = root(cfg)
    require_stage(cfg, "develop")
    s, model, hist = selected_artifacts(cfg)
    eta = s["eta"]
    records = []
    for row in tqdm(
        sources(cfg, "validation"), desc="SPOMF module comparison", file=sys.stdout
    ):
        image = processed_image(row["path"], cfg)
        bits, word = message_for(cfg, row)
        marked, _ = png_roundtrip(embed(image, bits, codec_config(cfg), "E2", eta))
        for info, crop in crop_panel(marked, cfg, row["source_id"]):
            for mode in ("search", "spomf"):
                bank = collect(crop, codec_config(cfg), mode)
                predicted, _ = predict(bank, model, device_for(cfg))
                records.append(
                    {
                        "source_id": row["source_id"],
                        "condition": info["condition"],
                        "mode": mode,
                        "ER": bool(np.array_equal(predicted, bits)),
                        "search_s": bank["search_s"],
                        "phase_correct": tuple(bank["offset"])
                        == (info["top"] % 64, info["left"] % 64),
                    }
                )
    write_csv(rr / "comparator_module_rows.csv", records)
    text = "# Scoped synchronization comparison\n\n"
    text += "SPOMF-inspired exponent 0.5 filter on the existing 48+16 carrier; same inputs and decoder.\n"
    text += "This is an implemented module adaptation, not a reproduction of Korus et al.'s spatial pilot or fountain coding system.\n"
    text += "No claim of superiority to the published complete system is authorized by these rows.\n"
    text += "Source: https://link.springer.com/article/10.1007/s11042-011-0986-8, §2.2/§3.2.\n"
    (rr / "comparator_report.md").write_text(text, encoding="utf-8")
    return evidence_record(
        cfg,
        "comparator",
        [rr / "comparator_module_rows.csv", rr / "comparator_report.md"],
    )


def quality_matched(cfg):
    """Validation-only Delta grid matched to E0 delta36 PSNR; no test-conditioned matching."""
    rr = root(cfg)
    require_stage(cfg, "develop")
    require_stage(cfg, "comparator")
    s, model, hist = selected_artifacts(cfg)
    gap = cfg["evaluation"]["quality_gap_db"]
    if gap is None:
        raise RuntimeError(
            "quality_gap_db must be scientifically locked before matching"
        )
    # This supplemental grid never replaces the fixed-Delta primary comparison.
    deltas = [18.0, 24.0, 30.0, 36.0, 42.0, 48.0]
    records = []
    for row in tqdm(
        sources(cfg, "validation"), desc="quality-match Delta grid", file=sys.stdout
    ):
        image = processed_image(row["path"], cfg)
        bits, _ = message_for(cfg, row)
        base = codec_config(cfg)
        reference, _ = png_roundtrip(embed(image, bits, base, "E0"))
        target = quality(image, reference)["psnr"]
        for delta in deltas:
            cc = CodecConfig(
                delta=delta,
                tile_side=8,
                payload_bits=48,
                pilot_count=16,
                pattern_seed=base.pattern_seed,
                swap_details=base.swap_details,
            )
            marked, _ = png_roundtrip(embed(image, bits, cc, "E2", s["eta"]))
            q = quality(image, marked)
            records.append(
                {
                    "source_id": row["source_id"],
                    "delta": delta,
                    "eta": s["eta"],
                    "target_psnr": target,
                    **q,
                    "gap_db": abs(q["psnr"] - target),
                }
            )
    average = {
        delta: float(np.mean([r["gap_db"] for r in records if r["delta"] == delta]))
        for delta in deltas
    }
    chosen = min(deltas, key=lambda d: (average[d], d))
    feasible = average[chosen] <= gap
    write_csv(rr / "quality_matched_rows.csv", records)
    report = {
        "ready": True,
        "matched": feasible,
        "selected_delta": chosen if feasible else None,
        "mean_absolute_psnr_gap_db": average[chosen],
        "gap_limit": gap,
        "rule": "minimum source-paired absolute PSNR gap; min Delta tie",
        "performance_claim": "WITHHELD until separate matched-strength decoder training/fitting and frozen evaluation",
        "fixed_delta_main_comparison_unchanged": True,
        "role": "validation",
    }
    atomic_json(rr / "quality_matched_report.json", report)
    # Honest scope restriction closes implementation scope, never invents matched-strength ER.
    atomic_json(
        rr / "scope_and_quality_gate.json",
        {
            "ready": True,
            "quality_matched_report": str(rr / "quality_matched_report.json"),
            "comparator_report": str(rr / "comparator_report.md"),
            "matched_quality_performance_claim": False,
            "complete_external_method_superiority_claim": False,
            "main_claim": "fixed-Delta mechanism and recovery; actual distortion reported",
        },
    )
    return evidence_record(
        cfg,
        "quality_match",
        [
            rr / "quality_matched_report.json",
            rr / "quality_matched_rows.csv",
            rr / "scope_and_quality_gate.json",
        ],
    )


def oracle_diagnostics(cfg):
    rr = root(cfg)
    require_stage(cfg, "prepare")
    require_stage(cfg, "math-check")
    cc = codec_config(cfg)
    records = []
    for row in tqdm(
        sources(cfg, "validation"),
        desc="raster/tile oracle diagnostics",
        file=sys.stdout,
    ):
        image = processed_image(row["path"], cfg)
        bits, word = message_for(cfg, row)
        for layout in ("raster", "tile"):
            marked, _ = png_roundtrip(embed(image, bits, cc, "E0", layout=layout))
            q = quality(image, marked)
            for info, crop in crop_panel(marked, cfg, row["source_id"]):
                modes = (
                    ("raster", "raster_oracle")
                    if layout == "raster"
                    else ("fixed", "search", "tile_oracle")
                )
                for mode in modes:
                    oracle = mode.endswith("oracle")
                    bank = collect(
                        crop,
                        cc,
                        mode,
                        origin=(info["top"], info["left"]) if oracle else None,
                        original_width=(
                            image.shape[1] if mode == "raster_oracle" else None
                        ),
                    )
                    decoded, evidence = baseline(bank, "D0")
                    records.append(
                        {
                            "source_id": row["source_id"],
                            "condition": info["condition"],
                            "layout": layout,
                            "mode": mode,
                            "ER": bool(np.array_equal(decoded, bits)),
                            "bit_accuracy": float(np.mean(decoded == bits)),
                            "coverage": float(np.mean(bank["counts"] > 0)),
                            "psnr": q["psnr"],
                            "ssim": q["ssim"],
                            "elapsed_s": bank["collect_s"],
                            "oracle_metadata_used": oracle,
                            "original_block_columns": (
                                image.shape[1] // 8 if oracle else None
                            ),
                        }
                    )
    write_csv(rr / "oracle_diagnostics.csv", records)
    return evidence_record(cfg, "oracle_diagnostics", [rr / "oracle_diagnostics.csv"])
