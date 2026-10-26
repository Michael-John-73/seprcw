"""CUDA-only numerical codec with NumPy I/O compatible with ``codec``.

Haar operations deliberately remain separate float64 multiplies/adds: algebraic
simplification changes the legacy float-to-uint8 truncation at integer boundaries.
Search packs eight pixel alignments at a time rather than 4096 full images.
Only configuration-sized pilot metadata is constructed on the CPU.
"""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from . import legacy_codec as legacy

CodecConfig = legacy.CodecConfig
__all__ = ["CodecConfig", "embed", "collect", "rank_change", "quality"]
_HAAR = 0.7071067811865476


def _device():
    if not torch.cuda.is_available():
        raise RuntimeError("paper_a.cuda_codec requires CUDA; CPU fallback is forbidden")
    return torch.device("cuda")


def _image(image):
    legacy._validate_image(image)
    return torch.as_tensor(np.ascontiguousarray(image), device=_device())


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def _bgr_to_yuv(image):
    b, g, r = image.to(torch.int64).unbind(-1)
    y = (b * 1868 + g * 9617 + r * 4899 + 8192) >> 14
    u = ((b - y) * 8061 + (128 << 14) + 8192) >> 14
    v = ((r - y) * 14369 + (128 << 14) + 8192) >> 14
    return torch.stack((y, u, v), -1).clamp(0, 255).to(torch.uint8)


def _yuv_to_bgr(image):
    y, u, v = image.to(torch.int64).unbind(-1)
    u, v = u - 128, v - 128
    b = y + ((u * 33292 + 8192) >> 14)
    g = y + ((u * -6472 + v * -9519 + 8192) >> 14)
    r = y + ((v * 18678 + 8192) >> 14)
    return torch.stack((b, g, r), -1).clamp(0, 255).to(torch.uint8)


def _dwt(channel):
    x = channel.to(torch.float64)
    low = x[..., 0::2, :] * _HAAR + x[..., 1::2, :] * _HAAR
    high = x[..., 0::2, :] * _HAAR - x[..., 1::2, :] * _HAAR
    ll = low[..., 0::2] * _HAAR + low[..., 1::2] * _HAAR
    horizontal = high[..., 0::2] * _HAAR + high[..., 1::2] * _HAAR
    vertical = low[..., 0::2] * _HAAR - low[..., 1::2] * _HAAR
    diagonal = high[..., 0::2] * _HAAR - high[..., 1::2] * _HAAR
    return ll, (horizontal, vertical, diagonal)


def _idwt(ll, details):
    horizontal, vertical, diagonal = details
    low = torch.empty((*ll.shape[:-1], ll.shape[-1] * 2),
                      dtype=torch.float64, device=ll.device)
    high = torch.empty_like(low)
    low[..., 0::2] = ll * _HAAR + vertical * _HAAR
    low[..., 1::2] = ll * _HAAR - vertical * _HAAR
    high[..., 0::2] = horizontal * _HAAR + diagonal * _HAAR
    high[..., 1::2] = horizontal * _HAAR - diagonal * _HAAR
    result = torch.empty((*low.shape[:-2], low.shape[-2] * 2, low.shape[-1]),
                         dtype=torch.float64, device=ll.device)
    result[..., 0::2, :] = low * _HAAR + high * _HAAR
    result[..., 1::2, :] = low * _HAAR - high * _HAAR
    return result


def _blocks(ll):
    nr, nc = ll.shape[-2] // 4, ll.shape[-1] // 4
    lead = ll.shape[:-2]
    return (ll[..., :nr * 4, :nc * 4].reshape(*lead, nr, 4, nc, 4)
            .transpose(-3, -2).reshape(*lead, nr, nc, 16).clone())


def _selected(blocks):
    position = blocks[..., 1:].abs().argmax(-1) + 1
    coefficient = blocks.gather(-1, position[..., None]).squeeze(-1)
    competitors = blocks[..., 1:].abs().clone()
    competitors.scatter_(-1, (position - 1)[..., None], -torch.inf)
    return position, coefficient, competitors.amax(-1)


def _carrier(image):
    yuv = _bgr_to_yuv(image)
    h, w = (n // 4 * 4 for n in yuv.shape[:2])
    ll, details = _dwt(yuv[:h, :w, 1])
    return yuv, h, w, ll, details, _blocks(ll)


def _pattern(cfg, device):
    return tuple(torch.tensor(a.copy(), device=device) for a in legacy._pattern(cfg))


def _project(a, competitor, bit, delta, eta):
    base = 0.25 + 0.5 * bit
    lower = torch.zeros_like(a, dtype=torch.int64)
    if eta is not None:
        lower = torch.maximum(lower, torch.ceil((competitor + eta) / delta - base).long())
        lower = torch.where(delta * (lower + base) < competitor + eta, lower + 1, lower)
    k0 = torch.maximum(lower, torch.floor(a / delta - base).long())
    k1 = torch.maximum(lower, k0 + 1)
    q0, q1 = delta * (k0 + base), delta * (k1 + base)
    return torch.where((q0 - a).abs() <= (q1 - a).abs(), q0, q1)


@torch.no_grad()
def embed(image, bits, cfg, rule="E0", eta=2.0, layout="tile"):
    """Embed E0/E1/E2 on CUDA, returning a uint8 BGR NumPy image."""
    x = _image(image)
    bits = torch.as_tensor(np.asarray(bits), device=x.device)
    length = cfg.payload_bits if rule == "E0" else 48
    if (rule not in ("E0", "E1", "E2") or bits.shape != (length,)
            or not bool(((bits == 0) | (bits == 1)).all())):
        raise ValueError("invalid embedding request")
    if layout not in ("raster", "tile"):
        raise ValueError("invalid layout")
    if rule == "E2" and (not math.isfinite(eta) or eta <= 0):
        raise ValueError("eta must be positive")
    if rule != "E0" and (cfg.tile_side != 8 or cfg.payload_bits != 48):
        raise ValueError("E1/E2 require the paper-A 8x8/48-bit configuration")
    yuv, h, w, ll, details, blocks = _carrier(x)
    nr, nc = blocks.shape[:2]
    rr = torch.arange(nr, device=x.device)[:, None]
    cc = torch.arange(nc, device=x.device)[None, :]
    if layout == "tile":
        pslots, pbits, mslots, _, _ = _pattern(cfg, x.device)
        tile = torch.empty(cfg.tile_side**2, dtype=torch.float64, device=x.device)
        tile[pslots], tile[mslots] = pbits.double(), bits.double()
        payload = tile[(rr % cfg.tile_side) * cfg.tile_side + cc % cfg.tile_side]
    else:
        payload = bits[(rr * nc + cc) % length].double()
    position, coefficient, competitor = _selected(blocks)
    if rule == "E0":
        # numpy floor_divide corrects the quotient near an exact multiple.
        quotient = torch.div(coefficient.abs(), cfg.delta, rounding_mode="floor")
        q = (quotient + 0.25 + 0.5 * payload) * cfg.delta
    else:
        q = _project(coefficient.abs(), competitor, payload, cfg.delta,
                     eta if rule == "E2" else None)
    blocks.scatter_(-1, position[..., None],
                    torch.where(coefficient >= 0, q, -q)[..., None])
    ll[:nr * 4, :nc * 4] = (blocks.reshape(nr, nc, 4, 4)
                            .transpose(1, 2).reshape(nr * 4, nc * 4))
    horizontal, vertical, diagonal = details
    if cfg.swap_details:
        details = (vertical, horizontal, diagonal)
    # Do not saturate: the reference truncates toward zero and wraps modulo 256.
    yuv[:h, :w, 1] = _idwt(ll, details).trunc().to(torch.int64).remainder(256).byte()
    return _numpy(_yuv_to_bgr(yuv))


def _soft(coefficient, delta):
    residue = coefficient.abs().remainder(delta)
    return -torch.sin(2.0 * math.pi * residue / delta), (residue > delta * 0.5).byte()


def _packed_search(channel, cfg, spomf=False):
    """Return stable (pixel-y, pixel-x, phase, margin) from packed GPU scores."""
    h, w = channel.shape
    nr, nc = h // 8, w // 8
    source = F.pad(channel.double(), (0, 8, 0, 8))
    device = channel.device
    rr = torch.arange(nr, device=device)[None, :, None]
    cc = torch.arange(nc, device=device)[None, None, :]
    slots = torch.arange(64, device=device)
    rows, cols = slots // 8, slots % 8
    shifted = ((rows[:, None] + rows[None, :]) % 8) * 8
    shifted = shifted + (cols[:, None] + cols[None, :]) % 8
    signs = _pattern(cfg, device)[4][shifted]
    template_fft = torch.fft.fft2(_pattern(cfg, device)[4].reshape(8, 8))
    score_parts = []
    for py in range(8):
        packed = torch.stack([source[py:py + nr * 8, px:px + nc * 8]
                              for px in range(8)])
        low = packed[:, 0::2, :] * _HAAR + packed[:, 1::2, :] * _HAAR
        ll = low[..., 0::2] * _HAAR + low[..., 1::2] * _HAAR
        blocks = _blocks(ll)
        position = blocks[..., 1:].abs().argmax(-1) + 1
        coefficient = blocks.gather(-1, position[..., None]).squeeze(-1)
        soft, _ = _soft(coefficient, cfg.delta)
        valid_nr = (h - py) // 8
        valid_nc = (w - torch.arange(8, device=device)) // 8
        valid = (rr < valid_nr) & (cc < valid_nc[:, None, None])
        soft = torch.where(valid, soft, 0.0)
        pad_r, pad_c = (-nr) % 8, (-nc) % 8
        sums = (F.pad(soft, (0, pad_c, 0, pad_r))
                .reshape(8, (nr + pad_r) // 8, 8, (nc + pad_c) // 8, 8)
                .sum((1, 3)).reshape(8, 64))
        counts = (F.pad(valid.double(), (0, pad_c, 0, pad_r))
                  .reshape(8, (nr + pad_r) // 8, 8, (nc + pad_c) // 8, 8)
                  .sum((1, 3)).reshape(8, 64))
        if spomf:
            product = torch.fft.fft2(sums.reshape(8, 8, 8)) * template_fft.conj()
            denom = product.abs().sqrt()
            filtered = torch.where(denom > 1e-15, product / denom.clamp_min(1e-300), 0)
            surface = torch.fft.ifft2(filtered).real
            scores = surface[:, (-rows) % 8, (-cols) % 8]
            scores = scores.masked_fill(~valid.any((1, 2))[:, None], -torch.inf)
        else:
            pilot_n = (counts[:, None, :] * signs.abs()[None, :, :]).sum(-1)
            correlation = (sums[:, None, :] * signs[None, :, :]).sum(-1)
            scores = torch.where(pilot_n >= 4, correlation / pilot_n.clamp_min(1),
                                 -torch.inf)
        score_parts.append(scores)
    scores = torch.stack(score_parts).reshape(-1)
    # Stable argmax selects the lowest enumerated candidate on an exact tie.
    best = scores.argmax()
    first = scores[best]
    if not bool(torch.isfinite(first)):
        raise ValueError("insufficient pilot observations for blind phase search")
    second = scores.masked_fill(torch.arange(scores.numel(), device=device) == best,
                                -torch.inf).amax()
    margin = torch.where(torch.isfinite(second), first - second, 0.0)
    candidate, margin_value = int(best), float(margin)
    return candidate // 512, (candidate // 64) % 8, candidate % 64, margin_value


def _alignment(channel, cfg, mode, origin, original_width):
    if mode in ("tile_oracle", "raster_oracle"):
        if (origin is None or len(origin) != 2
                or any(not isinstance(v, (int, np.integer)) or v < 0 for v in origin)):
            raise ValueError("oracle requires nonnegative integer (top, left)")
        if mode == "raster_oracle" and (original_width is None
                                       or original_width < channel.shape[1]):
            raise ValueError("raster oracle requires true original width")
        top, left = origin
        py, px = (-top) % 8, (-left) % 8
        return py, px, (top + py) // 8, (left + px) // 8, 0.0
    if origin is not None or original_width is not None:
        raise ValueError("blind synchronization forbids oracle metadata")
    if mode in ("fixed", "raster"):
        return 0, 0, 0, 0, 0.0
    if mode not in ("search", "spomf"):
        raise ValueError("unsupported synchronization")
    py, px, phase, margin = _packed_search(channel, cfg, mode == "spomf")
    return py, px, phase // 8, phase % 8, margin


def _empty_bank(device):
    return {
        "features": _numpy(torch.empty((0, 5), dtype=torch.float32, device=device)),
        "soft": _numpy(torch.empty(0, dtype=torch.float32, device=device)),
        "hard": _numpy(torch.empty(0, dtype=torch.uint8, device=device)),
        "residue": _numpy(torch.empty(0, dtype=torch.float32, device=device)),
        "index": _numpy(torch.empty(0, dtype=torch.int64, device=device)),
        "counts": _numpy(torch.zeros(48, dtype=torch.int64, device=device)),
        "pilot_score": 0.0, "phase_margin": 0.0, "offset": (-1, -1),
        "search_s": 0.0, "collect_s": 0.0, "pilot_observations": 0,
    }


@torch.no_grad()
def collect(image, cfg, mode="search", origin=None, original_width=None):
    """Build the existing five-feature packed observation bank entirely on CUDA."""
    start = perf_counter()
    if cfg.tile_side != 8 or cfg.payload_bits != 48:
        raise ValueError("collection requires the paper-A 8x8/48-bit configuration")
    x = _image(image)
    channel = _bgr_to_yuv(x)[..., 1]
    py, px, r0, c0, margin = _alignment(channel, cfg, mode, origin, original_width)
    torch.cuda.synchronize(x.device)
    search_s = perf_counter() - start
    h, w = ((channel.shape[0] - py) // 4 * 4,
            (channel.shape[1] - px) // 4 * 4)
    if min(h, w) < 8:
        return _empty_bank(x.device)
    ll, _ = _dwt(channel[py:py + h, px:px + w])
    blocks = _blocks(ll)
    _, coefficient, competitor = _selected(blocks)
    nr, nc = blocks.shape[:2]
    soft, hard = _soft(coefficient, cfg.delta)
    rr = torch.arange(nr, device=x.device)[:, None]
    cc = torch.arange(nc, device=x.device)[None, :]
    if mode in ("raster", "raster_oracle"):
        mapped = ((rr * nc + cc) % 48 if mode == "raster" else
                  ((rr + r0) * (original_width // 8) + cc + c0) % 48)
        pilots = torch.zeros_like(coefficient)
    else:
        slot = ((rr + r0) % 8) * 8 + (cc + c0) % 8
        _, _, _, payload_index, signs = _pattern(cfg, x.device)
        mapped, pilots = payload_index[slot], signs[slot]
    mask = mapped >= 0
    target = mapped[mask].long()
    count = torch.bincount(target, minlength=48)
    residue = coefficient.abs().remainder(cfg.delta) / cfg.delta
    boundary = torch.minimum(torch.minimum(residue, (residue - 0.5).abs()), 1 - residue)
    patch_u = (channel[py:py + nr * 8, px:px + nc * 8]
               .reshape(nr, 8, nc, 8).permute(0, 2, 1, 3).double())
    pix = (x[py:py + nr * 8, px:px + nc * 8]
           .reshape(nr, 8, nc, 8, 3).permute(0, 2, 1, 3, 4))
    variance = patch_u.var(dim=(2, 3), correction=0).log1p() / math.log(1 + 255**2)
    saturation = ((pix == 0) | (pix == 255)).double().mean((2, 3, 4))
    features = torch.stack(((coefficient.abs() - competitor) / cfg.delta,
                            boundary, variance, saturation,
                            count[mapped.clamp_min(0)].double().log1p()), -1)[mask].float()
    if not bool(torch.isfinite(features).all()):
        raise ValueError("nonfinite received feature")
    pilot_mask = pilots != 0
    pilot_n = int(pilot_mask.sum())
    pscore = float((soft[pilot_mask] * pilots[pilot_mask]).mean()) if pilot_n else 0.0
    result = {
        "features": _numpy(features), "soft": _numpy(soft[mask].float()),
        "hard": _numpy(hard[mask]), "residue": _numpy(residue[mask].float()),
        "index": _numpy(target), "counts": _numpy(count),
        "pilot_score": pscore, "phase_margin": float(margin),
        "offset": ((8 * r0 - py) % 64, (8 * c0 - px) % 64),
        "search_s": search_s, "pilot_observations": pilot_n,
    }
    torch.cuda.synchronize(x.device)
    result["collect_s"] = perf_counter() - start
    return result


@torch.no_grad()
def rank_change(original, marked, cfg):
    """Fraction of changed raw-LL selected coefficient positions."""
    a, b = _image(original), _image(marked)
    if a.shape != b.shape:
        raise ValueError("rank-change images must have equal shapes")
    pa = _selected(_carrier(a)[-1])[0]
    pb = _selected(_carrier(b)[-1])[0]
    changed = pa != pb
    denominator = torch.tensor(changed.numel(), dtype=torch.float64, device=a.device)
    return float(changed.sum().double() / denominator)


@torch.no_grad()
def quality(original, marked):
    """PSNR and skimage-default SSIM (7x7 uniform, sample covariance) on CUDA."""
    a, b = _image(original), _image(marked)
    if a.shape != b.shape:
        raise ValueError("quality images must have equal shapes")
    x, y = a.double(), b.double()
    mse = (x - y).square().mean()
    infinite = bool(mse == 0)
    psnr = 120.0 if infinite else float(10 * torch.log10(255**2 / mse))
    # Only the valid interior contributes in skimage; reflection padding is
    # unnecessary. Axis-0 then axis-1 filtering mirrors scipy.uniform_filter.
    values = torch.stack((x, y, x * x, y * y, x * y))
    values = values.permute(0, 3, 1, 2)
    means = F.avg_pool2d(F.avg_pool2d(values, (7, 1), stride=1), (1, 7), stride=1)
    ux, uy, uxx, uyy, uxy = means.unbind(0)
    covariance = 49.0 / 48.0
    vx, vy = covariance * (uxx - ux * ux), covariance * (uyy - uy * uy)
    vxy = covariance * (uxy - ux * uy)
    c1, c2 = (0.01 * 255)**2, (0.03 * 255)**2
    ssim = (((2 * ux * uy + c1) * (2 * vxy + c2)) /
            ((ux * ux + uy * uy + c1) * (vx + vy + c2)))
    return {"psnr": psnr, "psnr_infinite": infinite,
            "ssim": float(ssim.mean((1, 2)).mean())}
