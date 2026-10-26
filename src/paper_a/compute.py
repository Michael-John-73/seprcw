"""Explicit production compute policy; CPU is limited to diagnostics and I/O."""
from contextvars import ContextVar
import torch

_CUDA = ContextVar("paper_a_cuda_compute", default=False)


def configure_compute(cfg):
    from .numerics import configure_precision
    configure_precision()
    enabled = cfg["study"]["mode"] != "diagnostic"
    if enabled:
        if cfg["training"]["device"] != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Production numerical computation requires CUDA; no CPU fallback")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    _CUDA.set(enabled)


def cuda_enabled():
    return _CUDA.get()


def tensor(value, dtype=torch.float64):
    return torch.as_tensor(value, dtype=dtype, device="cuda")
