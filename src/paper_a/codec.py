"""E0/E1/E2 and blind observation collection; labels never enter synchronization."""

from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import cv2
import numpy as np
import pywt
from . import legacy_codec as legacy

CodecConfig = legacy.CodecConfig


def project_magnitude(magnitude, competitor, bit, delta, eta=None):
    """Nearest feasible lattice point. Exact ties choose the smaller integer."""
    a = np.asarray(magnitude, dtype=np.float64)
    bit = np.asarray(bit)
    u = np.asarray(competitor, dtype=np.float64)
    if (
        not np.isfinite(a).all()
        or not np.isfinite(u).all()
        or np.any(a < 0)
        or np.any(u < 0)
    ):
        raise ValueError("invalid magnitudes")
    if not np.isfinite(delta) or delta <= 0 or not np.isin(bit, [0, 1]).all():
        raise ValueError("invalid QIM inputs")
    if eta is not None and (not np.isfinite(eta) or eta <= 0):
        raise ValueError("eta must be positive")
    base = 0.25 + 0.5 * bit
    lower = np.zeros(np.broadcast_shapes(a.shape, bit.shape, u.shape), dtype=np.int64)
    if eta is not None:
        lower = np.maximum(lower, np.ceil((u + eta) / delta - base).astype(np.int64))
        # Correct rare floating-point ceil boundary errors without loosening the constraint.
        qlow = delta * (lower + base)
        lower = np.where(qlow < u + eta, lower + 1, lower)
    raw = a / delta - base
    k0 = np.maximum(lower, np.floor(raw).astype(np.int64))
    k1 = np.maximum(lower, k0 + 1)
    q0, q1 = delta * (k0 + base), delta * (k1 + base)
    return np.where(abs(q0 - a) <= abs(q1 - a), q0, q1)


def carrier(image, cfg):
    legacy._validate_image(image)
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    h, w = (np.array(yuv.shape[:2]) // 4 * 4).tolist()
    ll, details = pywt.dwt2(yuv[:h, :w, 1], "haar")
    return yuv, h, w, ll, details, legacy._blocks(ll)


def selected(blocks):
    positions = np.argmax(abs(blocks[..., 1:]), axis=-1) + 1
    c = np.take_along_axis(blocks, positions[..., None], -1)[..., 0]
    competitors = abs(blocks[..., 1:]).copy()
    np.put_along_axis(competitors, (positions - 1)[..., None], -np.inf, -1)
    u = competitors.max(-1)
    return positions, c, u


def embed(image, bits, cfg, rule="E0", eta=2.0, layout="tile"):
    from .compute import cuda_enabled
    if cuda_enabled():
        from .cuda_codec import embed as cuda_embed
        return cuda_embed(image, bits, cfg, rule, eta, layout)
    if rule == "E0":
        return legacy.embed(image, bits, cfg, mode=layout)
    bits = np.asarray(bits)
    if (
        rule not in ("E1", "E2")
        or bits.shape != (48,)
        or not np.isin(bits, [0, 1]).all()
    ):
        raise ValueError("invalid embedding request")
    if layout not in ("raster", "tile"):
        raise ValueError("invalid layout")
    yuv, h, w, ll, details, blocks = carrier(image, cfg)
    nr, nc = blocks.shape[:2]
    if layout == "tile":
        pslots, pbits, mslots, _, _ = legacy._pattern(cfg)
        tile = np.empty(64, dtype=np.uint8)
        tile[pslots], tile[mslots] = pbits, bits
        rr, cc = np.indices((nr, nc))
        payload = tile[(rr % 8) * 8 + cc % 8]
    else:
        payload = bits[np.arange(nr * nc).reshape(nr, nc) % 48]
    pos, c, u = selected(blocks)
    q = project_magnitude(abs(c), u, payload, cfg.delta, eta if rule == "E2" else None)
    np.put_along_axis(blocks, pos[..., None], np.where(c >= 0, q, -q)[..., None], -1)
    ll[: nr * 4, : nc * 4] = (
        blocks.reshape(nr, nc, 4, 4).transpose(0, 2, 1, 3).reshape(nr * 4, nc * 4)
    )
    horizontal, vertical, diagonal = details
    inverse_details = (vertical, horizontal, diagonal) if cfg.swap_details else details
    # Deliberate legacy float->uint8 truncation/wrap, identical across E0/E1/E2.
    yuv[:h, :w, 1] = pywt.idwt2((ll, inverse_details), "haar")
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def png_roundtrip(image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if result is None or not np.array_equal(result, image):
        raise RuntimeError("lossless PNG roundtrip mismatch")
    return result, encoded.tobytes()


def rank_change(original, marked, cfg):
    from .compute import cuda_enabled
    if cuda_enabled():
        from .cuda_codec import rank_change as cuda_rank_change
        return cuda_rank_change(original, marked, cfg)
    a = selected(carrier(original, cfg)[-1])[0]
    b = selected(carrier(marked, cfg)[-1])[0]
    return float(np.mean(a != b))


def spomf_scores(soft, cfg, exponent=0.5):
    """Explicit module adaptation, not the full Korus fountain-watermark system.
    alpha=0 must equal unnormalized circular pilot correlation.
    """
    sums, counts = legacy._periodic_aggregate(soft, cfg.tile_side)
    observed = sums.reshape(8, 8)
    template = legacy._pattern(cfg)[4].reshape(8, 8)
    product = np.fft.fft2(observed) * np.conj(np.fft.fft2(template))
    denom = abs(product) ** exponent
    filtered = np.divide(
        product, denom, out=np.zeros_like(product), where=denom > 1e-15
    )
    surface = np.fft.ifft2(filtered).real
    rr, cc = np.indices((8, 8))
    scores = surface[(-rr) % 8, (-cc) % 8].ravel()
    return scores


def choose_alignment(image, cfg, mode="search", origin=None, original_width=None):
    start = perf_counter()
    channel = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)[..., 1]
    if mode in ("tile_oracle", "raster_oracle"):
        if origin is None:
            raise ValueError("oracle requires origin")
        top, left = origin
        py, px = (-top) % 8, (-left) % 8
        r0, c0 = (top + py) // 8, (left + px) // 8
        if mode == "raster_oracle" and (
            original_width is None or original_width < image.shape[1]
        ):
            raise ValueError("raster oracle requires true original width")
        return py, px, r0, c0, 0.0, perf_counter() - start
    if origin is not None or original_width is not None:
        raise ValueError("blind synchronization forbids oracle metadata")
    if mode in ("fixed", "raster"):
        return 0, 0, 0, 0, 0.0, perf_counter() - start
    if mode == "search":
        r = legacy.decode(image, 48, cfg, "tile_search")
        oy, ox = r["offset"]
        py, px = (-oy) % 8, (-ox) % 8
        return (
            py,
            px,
            ((oy + py) // 8) % 8,
            ((ox + px) // 8) % 8,
            r["phase_margin"],
            perf_counter() - start,
        )
    if mode != "spomf":
        raise ValueError("unsupported synchronization")
    candidates = []
    for py in range(8):
        for px in range(8):
            soft, _ = legacy._observations(channel, py, px, cfg.delta)
            if soft.size:
                scores = spomf_scores(soft, cfg)
                for phase in range(64):
                    candidates.append((float(scores[phase]), py, px, phase))
    if not candidates:
        raise ValueError("insufficient synchronization observations")
    # Stable enumeration and stable sort preserve lowest enumerated tie.
    candidates.sort(key=lambda item: -item[0])
    best, py, px, phase = candidates[0]
    r0, c0 = divmod(phase, 8)
    return py, px, r0, c0, best - candidates[1][0], perf_counter() - start


def collect(image, cfg, mode="search", origin=None, original_width=None):
    """Return a packed observation bank. No payload/reference argument is accepted."""
    from .compute import cuda_enabled
    if cuda_enabled():
        from .cuda_codec import collect as cuda_collect
        return cuda_collect(image, cfg, mode, origin, original_width)
    start = perf_counter()
    py, px, r0, c0, margin, search_s = choose_alignment(
        image, cfg, mode, origin, original_width
    )
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    channel = yuv[..., 1]
    h, w = (np.array(channel.shape) - [py, px]) // 4 * 4
    if min(h, w) < 8:
        return empty_bank()
    ll, _ = pywt.dwt2(channel[py : py + h, px : px + w], "haar")
    blocks = legacy._blocks(ll)
    pos, c, u = selected(blocks)
    nr, nc = blocks.shape[:2]
    soft, hard = legacy._observations(channel, py, px, cfg.delta)
    rr, cc = np.indices((nr, nc))
    if mode == "raster":
        mapped = np.arange(nr * nc).reshape(nr, nc) % 48
        pilots = np.zeros((nr, nc))
    elif mode == "raster_oracle":
        mapped = ((rr + r0) * (original_width // 8) + cc + c0) % 48
        pilots = np.zeros((nr, nc))
    else:
        slot = ((rr + r0) % 8) * 8 + (cc + c0) % 8
        mapped = legacy._pattern(cfg)[3][slot]
        pilots = legacy._pattern(cfg)[4][slot]
    mask = mapped >= 0
    target = mapped[mask].astype(np.int64)
    count = np.bincount(target, minlength=48)
    residue = (abs(c) % cfg.delta) / cfg.delta
    boundary_distance = np.minimum.reduce([residue, abs(residue - 0.5), 1 - residue])
    patch_u = (
        channel[py : py + nr * 8, px : px + nc * 8]
        .reshape(nr, 8, nc, 8)
        .transpose(0, 2, 1, 3)
    )
    pix = (
        image[py : py + nr * 8, px : px + nc * 8]
        .reshape(nr, 8, nc, 8, 3)
        .transpose(0, 2, 1, 3, 4)
    )
    variance = np.log1p(patch_u.astype(np.float64).var(axis=(2, 3))) / (
        np.log(1 + 255**2)
    )
    saturation = ((pix == 0) | (pix == 255)).mean(axis=(2, 3, 4))
    features = np.stack(
        [
            (abs(c) - u) / cfg.delta,
            boundary_distance,
            variance,
            saturation,
            np.log1p(count[np.maximum(mapped, 0)]),
        ],
        -1,
    )[mask].astype(np.float32)
    pilot_mask = pilots != 0
    pscore = (
        float(np.mean(soft[pilot_mask] * pilots[pilot_mask]))
        if pilot_mask.any()
        else 0.0
    )
    result = {
        "features": features,
        "soft": soft[mask].astype(np.float32),
        "hard": hard[mask].astype(np.uint8),
        "residue": residue[mask].astype(np.float32),
        "index": target,
        "counts": count,
        "pilot_score": pscore,
        "phase_margin": float(margin),
        "offset": ((8 * r0 - py) % 64, (8 * c0 - px) % 64),
        "search_s": search_s,
        "collect_s": perf_counter() - start,
        "pilot_observations": int(pilot_mask.sum()),
    }
    if not np.isfinite(features).all():
        raise ValueError("nonfinite received feature")
    return result


def empty_bank():
    return {
        "features": np.empty((0, 5), np.float32),
        "soft": np.empty(0, np.float32),
        "hard": np.empty(0, np.uint8),
        "residue": np.empty(0, np.float32),
        "index": np.empty(0, np.int64),
        "counts": np.zeros(48, np.int64),
        "pilot_score": 0.0,
        "phase_margin": 0.0,
        "offset": (-1, -1),
        "search_s": 0.0,
        "collect_s": 0.0,
        "pilot_observations": 0,
    }


def baseline(bank, decoder, histogram=None, affine=None):
    from .compute import cuda_enabled
    if cuda_enabled():
        from .cuda_reductions import baseline as cuda_baseline
        return cuda_baseline(bank, decoder, histogram, affine)
    idx, n = bank["index"], bank["counts"]
    if decoder == "D0":
        value = np.bincount(idx, weights=bank["hard"], minlength=48)
        mean = np.divide(value, n, out=np.zeros(48), where=n > 0)
        bit = (mean * 255 > 127).astype(np.int8)
        evidence = 2 * mean - 1
    elif decoder == "D1":
        value = np.bincount(idx, weights=bank["soft"], minlength=48)
        evidence = np.divide(value, n, out=np.zeros(48), where=n > 0)
        bit = (evidence >= 0).astype(np.int8)
    elif decoder == "D4":
        from .affine import validate_affine
        state = validate_affine(affine)
        value = np.bincount(idx, weights=bank["soft"], minlength=48)
        mean = np.divide(value, n, out=np.zeros(48), where=n > 0)
        evidence = state["temperature"] * mean + state["bias"]
        bit = (evidence >= 0).astype(np.int8)
    elif decoder == "D2":
        if histogram is None:
            raise ValueError("D2 requires a train-fitted histogram")
        nb = histogram.shape[1]
        b = np.minimum((bank["residue"] * nb).astype(int), nb - 1)
        llr = np.log(histogram[1, b]) - np.log(histogram[0, b])
        evidence = np.bincount(idx, weights=llr, minlength=48)
        bit = (evidence >= 0).astype(np.int8)
    else:
        raise ValueError(decoder)
    bit[n == 0] = -1
    return bit, np.tanh(evidence) if decoder in ("D2", "D4") else evidence


def detector_features(bank, evidence, reference):
    if np.asarray(reference).shape != (48,):
        raise ValueError("one 48-bit reference required")
    return np.array(
        [
            bank["pilot_score"],
            bank["phase_margin"],
            np.mean(bank["counts"] > 0),
            np.mean(evidence * (2 * np.asarray(reference, dtype=np.float64) - 1)),
        ],
        np.float64,
    )
