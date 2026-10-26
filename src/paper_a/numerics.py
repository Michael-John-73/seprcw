"""FP32 execution policy and small, explicit CPU/CUDA training comparisons.

Resource profiling and numerical equivalence are different checks. No result
from this module establishes recovery performance or statistical significance.
"""
from __future__ import annotations

import copy
import os
import numpy as np
import torch


TOLERANCES = {
    "logits": {"atol": 1e-5, "rtol": 1e-4},
    "loss": {"atol": 1e-6, "rtol": 1e-5},
    "gradients": {"atol": 2e-6, "rtol": 2e-4},
    "parameters_after_step": {"atol": 3e-6, "rtol": 2e-5},
}


def configure_precision():
    """Apply in every process, including checkpoint-only inference processes."""
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (":4096:8", ":16:8"):
        if torch.cuda.is_initialized():
            raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
        if workspace is not None:
            raise RuntimeError("Unsupported CUBLAS_WORKSPACE_CONFIG for deterministic CUDA")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "feature_dtype": "float32",
        "matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "amp": False,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "bitwise_determinism_claimed": False,
        "repeatability_scope": "fixed input/weights/device/software; not cross-device or historical reference identity",
    }


def one_update_snapshot(model, banks, device, settings, micro_batch=1):
    """Clone weights, compute all cases, clip once, and make one AdamW step.

    The input model, its buffers, and caller RNG state are left unchanged.
    Short microbatches use the full optimizer-group case denominator.
    """
    from .learning import packed, case_loss

    if not banks or micro_batch < 1:
        raise ValueError("nonempty cases and positive micro_batch required")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("actual CUDA required for a CUDA equivalence snapshot")
    clone = copy.deepcopy(model).to(device).train()
    optimizer = torch.optim.AdamW(
        clone.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"],
        betas=tuple(settings["betas"]), eps=settings["eps"],
    )
    optimizer.zero_grad(set_to_none=True)
    logits_all, counts_all = [], []
    value = 0.0
    for start in range(0, len(banks), micro_batch):
        part = banks[start:start + micro_batch]
        f, s, index, labels = packed(part, device)
        with torch.autocast(device_type=device.type, enabled=False):
            logits, counts, _ = clone(f, s, index, 48 * len(part))
            loss = case_loss(logits, counts, labels).sum() / len(banks)
        if not torch.isfinite(loss) or not torch.isfinite(logits).all():
            raise RuntimeError("nonfinite equivalence forward")
        loss.backward()
        value += float(loss.detach().cpu())
        logits_all.extend(logits.detach().cpu().tolist())
        counts_all.extend(counts.detach().cpu().tolist())
    gradients = {}
    for name, parameter in clone.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError("missing/nonfinite equivalence gradient: " + name)
        gradients[name] = parameter.grad.detach().cpu().numpy().copy().tolist()
    torch.nn.utils.clip_grad_norm_(clone.parameters(), settings["clip_grad_norm"],
                                  error_if_nonfinite=True)
    optimizer.step()
    parameters = {k: v.detach().cpu().numpy().copy().tolist()
                  for k, v in clone.named_parameters()}
    if not all(np.isfinite(np.asarray(v)).all() for v in parameters.values()):
        raise RuntimeError("nonfinite equivalence optimizer update")
    bit = (np.asarray(logits_all) >= 0).astype(int)
    bit[np.asarray(counts_all) == 0] = -1
    return {"logits": logits_all, "loss": value, "counts": counts_all,
            "hard_bits": bit.tolist(), "gradients": gradients,
            "parameters_after_step": parameters, "cases": len(banks),
            "micro_batch": micro_batch, "actual_device": str(device)}


def compare_snapshots(reference, candidate):
    """Require finite numerical agreement AND exact hard-bit/count agreement."""
    checks = []
    for field, tolerance in TOLERANCES.items():
        aa, bb = reference[field], candidate[field]
        if isinstance(aa, dict):
            if set(aa) != set(bb):
                raise RuntimeError("parameter keys differ for " + field)
            pairs = [(k, aa[k], bb[k]) for k in sorted(aa)]
        else:
            pairs = [(field, aa, bb)]
        for name, a, b in pairs:
            a, b = np.asarray(a), np.asarray(b)
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            same_shape = a.shape == b.shape
            passed = bool(finite and same_shape and np.allclose(a, b, **tolerance))
            checks.append({"field": field, "parameter": name, "passed": passed,
                           "max_absolute_error": float(np.max(np.abs(a-b)))
                           if finite and same_shape and a.size else None,
                           **tolerance})
    bit_a, bit_b = np.asarray(reference["hard_bits"]), np.asarray(candidate["hard_bits"])
    mismatches = int(np.count_nonzero(bit_a != bit_b)) if bit_a.shape == bit_b.shape else None
    counts_equal = bool(np.array_equal(reference["counts"], candidate["counts"]))
    return {"passed": bool(all(x["passed"] for x in checks) and mismatches == 0 and counts_equal
                           and reference["cases"] == candidate["cases"]),
            "reference_device": reference["actual_device"],
            "candidate_device": candidate["actual_device"],
            "hard_bit_mismatches": mismatches, "counts_equal": counts_equal,
            "checks": checks, "tolerance_policy": "fixed before CUDA observations"}


def cpu_cuda_equivalence(model, sample, settings):
    """A real CUDA comparison; never silently replace it with another CPU run."""
    precision = configure_precision()
    if not torch.cuda.is_available():
        raise RuntimeError("CPU/CUDA equivalence NOT_RUN_CUDA; CUDA device unavailable")
    reference = one_update_snapshot(model, [sample], "cpu", settings)
    candidate = one_update_snapshot(model, [sample], "cuda", settings)
    comparison = compare_snapshots(reference, candidate)
    return {**comparison, "precision": precision,
            "device_name": torch.cuda.get_device_name(0),
            "scope": "one maximum-shape development case and one AdamW update; no test data",
            "scientific_hypothesis_evidence": False}
