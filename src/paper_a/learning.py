"""Small packed reliability MLP, streaming training, and honest checkpoint selection."""

from __future__ import annotations
import copy
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from .io import log, atomic_json, read_json, sha, root
from .data import load_bank, bank_paths
from .codec import baseline
from .affine import fit_uniform_affine, validate_affine
from .numerics import configure_precision


class ReliabilityDecoder(nn.Module):
    def __init__(self, mean=None, std=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1, bias=False)
        )
        # A final common bias cancels in segment softmax. Omitting this
        # unidentifiable parameter avoids AdamW amplifying roundoff gradients.
        self.raw_temperature = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.register_buffer(
            "feature_mean",
            (
                torch.zeros(5)
                if mean is None
                else torch.as_tensor(mean, dtype=torch.float32)
            ),
        )
        self.register_buffer(
            "feature_std",
            torch.ones(5) if std is None else torch.as_tensor(std, dtype=torch.float32),
        )

    def forward(self, features, soft, index, segments):
        if len(soft) == 0:
            return (
                self.bias.expand(segments) * 0,
                torch.zeros(segments, device=soft.device),
                soft,
            )
        a = self.net((features - self.feature_mean) / self.feature_std).squeeze(-1)
        maximum = torch.full((segments,), -torch.inf, device=a.device, dtype=a.dtype)
        maximum.scatter_reduce_(0, index, a, reduce="amax", include_self=True)
        exp = torch.exp(a - maximum[index])
        den = torch.zeros(segments, device=a.device, dtype=a.dtype).scatter_add(
            0, index, exp
        )
        weights = exp / den[index]
        z = torch.zeros(segments, device=a.device, dtype=a.dtype).scatter_add(
            0, index, weights * soft
        )
        counts = torch.zeros(segments, device=a.device).scatter_add(
            0, index, torch.ones_like(soft)
        )
        temperature = F.softplus(self.raw_temperature) + 1e-4
        return temperature * z + self.bias, counts, weights


def packed(banks, device, supervised=True):
    if torch.device(device).type == "cuda":
        x = torch.cat([torch.as_tensor(b["features"], device=device) for b in banks])
        s = torch.cat([torch.as_tensor(b["soft"], device=device) for b in banks])
        idx = torch.cat([torch.as_tensor(b["index"], device=device, dtype=torch.long) + 48 * i
                         for i, b in enumerate(banks)])
        y = (torch.cat([torch.as_tensor(b["bits"], device=device, dtype=torch.float32) for b in banks])
             if supervised else torch.zeros(48 * len(banks), device=device))
        return x, s, idx, y
    x = np.concatenate([b["features"] for b in banks])
    s = np.concatenate([b["soft"] for b in banks])
    idx = np.concatenate([b["index"] + 48 * i for i, b in enumerate(banks)])
    y = (
        np.concatenate([b["bits"] for b in banks])
        if supervised
        else np.zeros(48 * len(banks), np.float32)
    )
    return (
        torch.as_tensor(x, device=device),
        torch.as_tensor(s, device=device),
        torch.as_tensor(idx, device=device, dtype=torch.long),
        torch.as_tensor(y, device=device, dtype=torch.float32),
    )


def case_loss(logits, counts, labels):
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").reshape(
        -1, 48
    )
    mask = (counts > 0).reshape(-1, 48)
    denom = mask.sum(1)
    if (denom == 0).any():
        raise RuntimeError(
            "training case with no observed bits: revise registered train coverage"
        )
    return (raw * mask).sum(1) / denom


def feature_statistics(paths):
    from .compute import cuda_enabled
    if cuda_enabled():
        from .cuda_reductions import feature_statistics as cuda_statistics
        return cuda_statistics(paths, load_bank)
    n = 0
    mean = np.zeros(5, np.float64)
    m2 = np.zeros(5, np.float64)
    for p in tqdm(paths, desc="train normalization", file=__import__("sys").stdout):
        x = load_bank(p)["features"].astype(np.float64)
        if len(x) == 0:
            continue
        bn = len(x)
        bm = x.mean(0)
        b2 = ((x - bm) ** 2).sum(0)
        diff = bm - mean
        m2 += b2 + diff**2 * n * bn / (n + bn)
        mean += diff * bn / (n + bn)
        n += bn
    if n < 2:
        raise RuntimeError("insufficient train observations")
    std = np.sqrt(m2 / n)
    return mean, np.maximum(std, 1e-6)


def fit_histogram(paths, bins, smoothing):
    from .compute import cuda_enabled
    if cuda_enabled():
        from .cuda_reductions import fit_histogram as cuda_histogram
        return cuda_histogram(paths, bins, smoothing, load_bank)
    count = np.full((2, bins), smoothing, np.float64)
    for p in tqdm(paths, desc="train LLR histogram", file=__import__("sys").stdout):
        b = load_bank(p)
        label = b["bits"][b["index"]]
        residue = np.minimum((b["residue"] * bins).astype(int), bins - 1)
        np.add.at(count, (label, residue), 1)
    return count / count.sum(1, keepdims=True)


def device_for(cfg, diagnostic=False):
    configure_precision()
    requested = cfg["training"]["device"]
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested; refusing silent CPU fallback")
    if requested == "cpu" and cfg["study"]["mode"] != "diagnostic":
        raise RuntimeError("CPU training is supported only in explicit diagnostic mode")
    return torch.device(requested)


def predict(bank, model, device, return_summary=False):
    configure_precision()
    model.eval()
    with torch.inference_mode():
        features, soft, index, labels = packed([bank], device, supervised=False)
        logits, counts, weights = model(features, soft, index, 48)
        if not torch.isfinite(logits).all() or not torch.isfinite(weights).all():
            raise RuntimeError("nonfinite learned inference")
        bit = (logits >= 0).to(torch.int8)
        bit[counts == 0] = -1
        # Positive normalization preserves the model's hard boundary at zero.
        ev = torch.tanh(logits)
    result = (bit.cpu().numpy(), ev.cpu().numpy())
    if return_summary:
        if torch.device(device).type == "cuda":
            entropy = torch.zeros(48, device=device).scatter_add_(
                0, index, -weights * weights.clamp_min(1e-30).log())
            maximum = torch.zeros(48, device=device)
            maximum.scatter_reduce_(0, index, weights, reduce="amax", include_self=True)
            valid = counts > 0
            return (*result, {
                "mean_entropy": float(entropy[valid].mean()) if bool(valid.any()) else None,
                "mean_max_weight": float(maximum[valid].mean()) if bool(valid.any()) else None,
                "observed_bits": int(valid.sum()),
            })
        ww = weights.cpu().numpy()
        ii = np.asarray(bank["index"])
        entropy = []
        maximum = []
        for n in range(48):
            w = ww[ii == n]
            if len(w):
                entropy.append(float(-np.sum(w * np.log(np.maximum(w, 1e-30)))))
                maximum.append(float(w.max()))
        return (
            *result,
            {
                "mean_entropy": float(np.mean(entropy)) if entropy else None,
                "mean_max_weight": float(np.mean(maximum)) if maximum else None,
                "observed_bits": len(entropy),
            },
        )
    return result


def evaluate_loss(paths, model, device, micro_batch=1, max_observations=65536):
    model.eval()
    values = []
    er = []
    if micro_batch < 1 or max_observations < 1:
        raise ValueError("positive validation batch and observation cap required")

    def evaluate_part(banks):
        f, s, i, y = packed(banks, device)
        logits, counts, _ = model(f, s, i, 48 * len(banks))
        losses = case_loss(logits, counts, y)
        if not torch.isfinite(losses).all():
            raise RuntimeError("nonfinite validation loss")
        bits = (logits >= 0).to(torch.int8)
        bits[counts == 0] = -1
        expected = y.reshape(-1, 48)
        er.extend((bits.reshape(-1, 48) == expected).all(1).cpu().tolist())
        values.extend(losses.cpu().tolist())

    with torch.inference_mode():
        pending = []
        observations = 0
        for p in paths:
            b = load_bank(p)
            size = len(b["soft"])
            if size > max_observations:
                raise RuntimeError("single validation case exceeds max_observations; never truncate")
            if pending and (len(pending) >= micro_batch or observations + size > max_observations):
                evaluate_part(pending)
                pending = []
                observations = 0
            pending.append(b)
            observations += size
        if pending:
            evaluate_part(pending)
    if not values:
        raise RuntimeError("validation banks required")
    return float(np.mean(values)), float(np.mean(er))


def backward_micro(model, banks, device, denominator):
    f, s, i, y = packed(banks, device)
    logits, counts, _ = model(f, s, i, 48 * len(banks))
    loss = case_loss(logits, counts, y).sum() / denominator
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite loss")
    loss.backward()
    return float(loss.detach())


def train_one(cfg, eta, seed, outdir):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = device_for(cfg)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
    paths = bank_paths(cfg, "train", eta, "E2")
    valpaths = bank_paths(cfg, "validation", eta, "E2")
    if not paths or not valpaths:
        raise RuntimeError("train and validation banks required")
    mean, std = feature_statistics(paths)
    model = ReliabilityDecoder(mean, std).to(device)
    trainable = list(model.parameters())
    opt = torch.optim.AdamW(
        trainable,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        betas=tuple(cfg["training"]["betas"]),
        eps=cfg["training"]["eps"],
    )
    if {id(p) for p in trainable} != {
        id(p) for g in opt.param_groups for p in g["params"]
    }:
        raise RuntimeError("optimizer/model identity mismatch")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    np.save(
        outdir / "histogram.npy",
        fit_histogram(
            paths, cfg["training"]["hist_bins"], cfg["training"]["hist_smoothing"]
        ),
    )
    affine_state = fit_uniform_affine(paths, load_bank, cfg["training"])
    atomic_json(outdir / "uniform_affine.json", affine_state)
    best = (float("inf"), -1)
    stale = 0
    updates = 0
    traces = []
    t = cfg["training"]
    effective = t["micro_batch"] * t["accumulation_steps"]
    actual_micro = t["micro_batch"]
    if device.type == "cuda":
        profile = read_json(root(cfg) / "memory_profile.json")
        if not profile["ready"] or eta not in profile.get("profiled_etas", []):
            raise RuntimeError("no safe measured batch profile for this eta")
        eligible = [
            row
            for row in profile["results"]
            if row["safe"] and row["micro_batch"] <= effective
        ]
        if not eligible:
            raise RuntimeError("no measured microbatch within effective batch")
        actual_micro = max(eligible, key=lambda row: row["cases_per_second"])[
            "micro_batch"
        ]
    log(
        "resolved batch plan",
        target_effective_cases=effective,
        actual_micro_batch=actual_micro,
        accumulation_chunks=(effective + actual_micro - 1) // actual_micro,
    )
    original = {k: v.detach().cpu().clone() for k, v in model.named_parameters()}
    reload_bank = load_bank(valpaths[0])
    best_reference = None
    for epoch in tqdm(
        range(t["epochs"]),
        desc=f"eta={eta:g} seed={seed} epochs",
        file=__import__("sys").stdout,
    ):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(paths))
        losses = []
        for start in tqdm(
            range(0, len(paths), effective),
            desc="optimizer groups",
            leave=False,
            file=__import__("sys").stdout,
        ):
            group = [load_bank(paths[int(i)]) for i in order[start : start + effective]]
            opt.zero_grad(set_to_none=True)
            offset = 0
            micro = actual_micro
            group_loss = 0.0
            while offset < len(group):
                count = min(micro, len(group) - offset)
                banks = group[offset : offset + count]
                total_obs = sum(len(b["soft"]) for b in banks)
                if total_obs > t["max_observations"]:
                    if count == 1:
                        raise RuntimeError(
                            "single case exceeds max_observations: revise preprocessing, never truncate"
                        )
                    micro = max(1, count // 2)
                    continue
                try:
                    # A helper scope releases graph/tensor references even when backward raises OOM.
                    group_loss += backward_micro(model, banks, device, len(group))
                    offset += count
                    del banks
                except torch.cuda.OutOfMemoryError:
                    # Forward/backward may have partially accumulated gradients: restart the full group.
                    opt.zero_grad(set_to_none=True)
                    del banks
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if micro == 1:
                        raise
                    micro = max(1, micro // 2)
                    actual_micro = micro
                    log(
                        "OOM group restart", micro_batch=micro, source_group_start=start
                    )
                    offset = 0
                    group_loss = 0.0
            grads = [p.grad for p in trainable if p.grad is not None]
            if len(grads) != len(trainable) or not all(
                torch.isfinite(g).all() for g in grads
            ):
                raise RuntimeError("missing/nonfinite gradient")
            gradnorm = float(
                torch.nn.utils.clip_grad_norm_(
                    trainable, t["clip_grad_norm"], error_if_nonfinite=True
                )
            )
            opt.step()
            updates += 1
            # Weight a short final optimizer group by its actual case count.
            losses.extend([group_loss] * len(group))
        val_loss, val_er = evaluate_loss(
            valpaths, model, device, actual_micro, t["max_observations"]
        )
        changed = any(
            not torch.equal(v.detach().cpu(), original[k])
            for k, v in model.named_parameters()
        )
        if not changed:
            raise RuntimeError("no parameter update")
        trace = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_loss": val_loss,
            "validation_er": val_er,
            "grad_norm_last": gradnorm,
            "updates": updates,
            "peak_allocated": (
                torch.cuda.max_memory_allocated() if device.type == "cuda" else None
            ),
            "peak_reserved": (
                torch.cuda.max_memory_reserved() if device.type == "cuda" else None
            ),
        }
        traces.append(trace)
        log("epoch", eta=eta, seed=seed, **trace)
        # Earliest epoch wins ties. Validation only; no test/calibration access.
        if val_loss < best[0]:
            best = (val_loss, epoch)
            stale = 0
            # Capture the live best model before serialization, not another reload.
            ref_bits, ref_evidence = predict(reload_bank, model, device)
            best_reference = (ref_bits.copy(), ref_evidence.copy())
            checkpoint = {
                "reload_reference": {
                    "bank_hash": sha(valpaths[0]),
                    "predicted_bits": ref_bits.tolist(),
                    "evidence": ref_evidence.tolist(),
                    "precision_contract": configure_precision(),
                },
                "uniform_affine": affine_state,
                "state_dict": model.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "updates": updates,
                "config": cfg,
                "eta": eta,
                "seed": seed,
                "train_manifest": sha(
                    root(cfg) / "banks" / "train" / f"eta_{eta:g}" / "manifest.json"
                ),
                "validation_manifest": sha(
                    root(cfg)
                    / "banks"
                    / "validation"
                    / f"eta_{eta:g}"
                    / "manifest.json"
                ),
                "resume_supported": False,
                "runtime_batch_plan": {
                    "target_effective_cases": effective,
                    "actual_micro_batch": actual_micro,
                },
            }
            torch.save(checkpoint, outdir / "best.pt")
        else:
            stale += 1
        atomic_json(outdir / "trace.json", traces)
        if stale >= t["early_stop_patience"]:
            break
    verify_checkpoint_reload(outdir / "best.pt", reload_bank, best_reference, device)
    result = {
        "eta": eta,
        "seed": seed,
        "best_epoch": best[1],
        "best_validation_loss": best[0],
        "checkpoint": str(outdir / "best.pt"),
        "checkpoint_hash": sha(outdir / "best.pt"),
        "histogram": str(outdir / "histogram.npy"),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "updates": updates,
        "uniform_affine": affine_state,
        "training_healthy": True,
        "resume_supported": False,
        "reload_verified_against_live_best_model": True,
        "feature_dtype": "float32",
        "amp": False,
        "precision_contract": configure_precision(),
        "runtime_batch_plan": {
            "target_effective_cases": effective,
            "actual_micro_batch": actual_micro,
        },
    }
    atomic_json(outdir / "training_report.json", result)
    return result


def load_model(path, device):
    configure_precision()
    state = torch.load(path, map_location=device, weights_only=False)
    model = ReliabilityDecoder().to(device)
    expected = model.state_dict()
    actual = state["state_dict"]
    for name, value in actual.items():
        if name in expected and (
            value.dtype != expected[name].dtype
            or value.shape != expected[name].shape
            or not torch.isfinite(value).all()
        ):
            raise RuntimeError(f"invalid checkpoint tensor: {name}")
    if "feature_std" in actual and (actual["feature_std"] <= 0).any():
        raise RuntimeError("checkpoint feature_std must be positive")
    model.load_state_dict(actual, strict=True)
    model.eval()
    model._checkpoint_hash = sha(path)
    model._uniform_affine = (validate_affine(state["uniform_affine"])
                             if "uniform_affine" in state else None)
    return model, state


def verify_checkpoint_reload(path, bank, expected, device):
    """Compare saved weights to the live best model output captured before saving."""
    if expected is None:
        raise RuntimeError("no live best-model reload reference")
    restored, _ = load_model(path, device)
    actual = predict(bank, restored, device)
    if not np.array_equal(actual[0], expected[0]) or not np.allclose(
        actual[1], expected[1], rtol=0, atol=1e-7
    ):
        raise RuntimeError(
            "checkpoint reload differs from live best model: "
            f"hard_mismatches={int(np.sum(actual[0] != expected[0]))}, "
            f"max_evidence_error={float(np.max(np.abs(actual[1] - expected[1]))):.12g}, "
            "atol=1e-7, rtol=0"
        )
    return True
