"""Bounded-memory CUDA source-cluster bootstrap."""
import math
import torch
from .compute import tensor


def bootstrap_difference(d, reps, family_alpha, comparisons=6, seed=0):
    if type(reps) is not int or reps < 1 or comparisons < 1 or not 0 < family_alpha < 1:
        raise ValueError("invalid bootstrap parameters")
    values = tensor(d)
    tail = family_alpha / (2 * comparisons)
    mean = float(values.mean())
    if bool(values.max() == values.min()):
        radius = math.sqrt(2 * math.log(1 / tail) / len(values))
        return mean, max(-1.0, mean - radius), min(1.0, mean + radius)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    draws = torch.empty(reps, dtype=torch.float64, device="cuda")
    chunk = max(1, min(256, 1_000_000 // len(values)))
    for start in range(0, reps, chunk):
        count = min(chunk, reps - start)
        indices = torch.randint(len(values), (count, len(values)), device="cuda", generator=generator)
        draws[start:start + count] = values[indices].mean(1)
    bounds = torch.quantile(draws, tensor([tail, 1 - tail]))
    return mean, float(bounds[0]), float(bounds[1])
