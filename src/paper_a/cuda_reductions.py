"""CUDA reductions for corpus-scale fitting and conventional decoders."""
import numpy as np
import torch
from .compute import tensor


def baseline(bank, decoder, histogram=None, affine=None):
    idx = tensor(bank["index"], torch.long)
    n = tensor(bank["counts"])
    if decoder == "D0":
        value = torch.zeros(48, device="cuda", dtype=torch.float64)
        value.scatter_add_(0, idx, tensor(bank["hard"]))
        mean = value / n.clamp_min(1)
        bit = (mean * 255 > 127).to(torch.int8)
        evidence = 2 * mean - 1
    elif decoder in ("D1", "D4"):
        value = torch.zeros(48, device="cuda", dtype=torch.float64)
        value.scatter_add_(0, idx, tensor(bank["soft"]))
        evidence = value / n.clamp_min(1)
        if decoder == "D4":
            from .affine import validate_affine
            state = validate_affine(affine)
            evidence = state["temperature"] * evidence + state["bias"]
        bit = (evidence >= 0).to(torch.int8)
    elif decoder == "D2":
        if histogram is None:
            raise ValueError("D2 requires a train-fitted histogram")
        hist = tensor(histogram)
        # Preserve the stored float32 residue multiplication before binning.
        bins = (tensor(bank["residue"], torch.float32) * hist.shape[1]).long()
        bins.clamp_(max=hist.shape[1] - 1)
        llr = hist[1, bins].log() - hist[0, bins].log()
        evidence = torch.zeros(48, device="cuda", dtype=torch.float64)
        evidence.scatter_add_(0, idx, llr)
        bit = (evidence >= 0).to(torch.int8)
    else:
        raise ValueError(decoder)
    bit[n == 0] = -1
    if decoder in ("D2", "D4"):
        evidence = evidence.tanh()
    return bit.cpu().numpy(), evidence.cpu().numpy()


def feature_statistics(paths, load_bank):
    from tqdm import tqdm
    import sys
    n = 0
    mean = torch.zeros(5, dtype=torch.float64, device="cuda")
    m2 = torch.zeros_like(mean)
    for path in tqdm(paths, desc="CUDA train normalization", file=sys.stdout):
        x = tensor(load_bank(path)["features"])
        bn = len(x)
        if not bn:
            continue
        bm = x.mean(0)
        b2 = ((x - bm) ** 2).sum(0)
        diff = bm - mean
        m2 += b2 + diff.square() * n * bn / (n + bn)
        mean += diff * bn / (n + bn)
        n += bn
    if n < 2:
        raise RuntimeError("insufficient train observations")
    return mean.cpu().numpy(), (m2 / n).sqrt().clamp_min(1e-6).cpu().numpy()


def fit_histogram(paths, bins, smoothing, load_bank):
    from tqdm import tqdm
    import sys
    count = torch.full((2 * bins,), float(smoothing), device="cuda", dtype=torch.float64)
    for path in tqdm(paths, desc="CUDA train histogram", file=sys.stdout):
        bank = load_bank(path)
        labels = tensor(bank["bits"], torch.long)[tensor(bank["index"], torch.long)]
        residue = (tensor(bank["residue"], torch.float32) * bins).long().clamp_max(bins - 1)
        count += torch.bincount(labels * bins + residue, minlength=2 * bins)
    count = count.reshape(2, bins)
    return (count / count.sum(1, keepdim=True)).cpu().numpy()
