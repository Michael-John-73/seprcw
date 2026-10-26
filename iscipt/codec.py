"""Pilot-assisted crop synchronization using the audited deployed quantizer.

Both layouts share the uint8 BGR/YUV round trip, U-channel Haar transform,
raw-LL maximum-coefficient selection (excluding element zero), quantizer,
and optional swapped Haar details. No DCT is performed: this deliberately
preserves invisible-watermark's deployed ``dwtDct`` behavior and defects.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import cv2
import numpy as np
import pywt


@dataclass(frozen=True)
class CodecConfig:
    delta: float = 36.0
    tile_side: int = 8
    payload_bits: int = 48
    pilot_count: int = 16
    pattern_seed: int = 20261008
    swap_details: bool = True

    def __post_init__(self):
        for name in ("tile_side", "payload_bits", "pilot_count", "pattern_seed"):
            if not isinstance(getattr(self, name), (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if not np.isfinite(self.delta) or self.delta <= 0:
            raise ValueError("delta must be finite and positive")
        if self.tile_side < 2 or self.payload_bits < 1:
            raise ValueError("tile_side >= 2 and payload_bits >= 1 are required")
        if self.pilot_count < 4 or self.pilot_count % 2:
            raise ValueError("pilot_count must be even and at least four")
        if self.payload_bits + self.pilot_count != self.tile_side ** 2:
            raise ValueError("payload_bits + pilot_count must exactly fill a tile")
        if self.pattern_seed < 0:
            raise ValueError("pattern_seed must be nonnegative")
        if not isinstance(self.swap_details, (bool, np.bool_)):
            raise ValueError("swap_details must be boolean")


def _validate_image(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("image must be a uint8 numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have BGR shape (height, width, 3)")
    if min(image.shape[:2]) < 8:
        raise ValueError("at least one complete 8 by 8 pixel carrier block is required")


@lru_cache(maxsize=32)
def _pattern(cfg):
    """Spread pilots by toroidal farthest-point sampling, then balance bits."""
    side = cfg.tile_side
    rng = np.random.default_rng(cfg.pattern_seed)
    rows, cols = np.indices((side, side)).reshape(2, -1)
    tie_order = rng.permutation(side * side)
    chosen = []
    nearest = np.full(side * side, np.inf)
    for _ in range(cfg.pilot_count):
        slot = int(tie_order[np.argmax(nearest[tie_order])])
        chosen.append(slot)
        dr = np.minimum(abs(rows - rows[slot]), side - abs(rows - rows[slot]))
        dc = np.minimum(abs(cols - cols[slot]), side - abs(cols - cols[slot]))
        nearest = np.minimum(nearest, dr * dr + dc * dc)
        nearest[chosen] = -1
    pilot_slots = np.asarray(chosen, dtype=np.int64)
    pilot_bits = np.tile(np.array([0, 1], dtype=np.uint8), cfg.pilot_count // 2)
    rng.shuffle(pilot_bits)
    payload_slots = np.setdiff1d(np.arange(side * side), pilot_slots)
    payload_index = np.full(side * side, -1, dtype=np.int64)
    payload_index[payload_slots] = np.arange(cfg.payload_bits)
    pilot_sign = np.zeros(side * side, dtype=np.float64)
    pilot_sign[pilot_slots] = 2.0 * pilot_bits - 1.0
    for array in (pilot_slots, pilot_bits, payload_slots, payload_index, pilot_sign):
        array.setflags(write=False)
    return pilot_slots, pilot_bits, payload_slots, payload_index, pilot_sign


@lru_cache(maxsize=32)
def _phase_tables(cfg):
    side = cfg.tile_side
    rr, cc = np.indices((side, side))
    r0, c0 = rr.ravel(), cc.ravel()
    shifted = ((rr.ravel()[None, :] + r0[:, None]) % side) * side
    shifted = shifted + (cc.ravel()[None, :] + c0[:, None]) % side
    signs = _pattern(cfg)[4][shifted]
    shifted.setflags(write=False)
    signs.setflags(write=False)
    return shifted, signs


def _blocks(ll):
    nr, nc = ll.shape[0] // 4, ll.shape[1] // 4
    return ll[:nr * 4, :nc * 4].reshape(nr, 4, nc, 4).transpose(0, 2, 1, 3).reshape(nr, nc, 16).copy()


def _observations(channel, py=0, px=0, delta=36.0):
    """Return soft evidence and deployed hard decisions on an aligned grid."""
    h, w = channel.shape[0] - py, channel.shape[1] - px
    if h < 8 or w < 8:
        return np.empty((0, 0)), np.empty((0, 0), dtype=np.uint8)
    h, w = h // 4 * 4, w // 4 * 4
    ll, _ = pywt.dwt2(channel[py:py + h, px:px + w], "haar")
    blocks = _blocks(ll)
    position = np.argmax(np.abs(blocks[..., 1:]), axis=-1) + 1
    coefficient = np.take_along_axis(blocks, position[..., None], axis=-1)[..., 0]
    residue = np.abs(coefficient) % delta
    hard = (residue > 0.5 * delta).astype(np.uint8)
    soft = -np.sin(2.0 * np.pi * residue / delta)
    return soft, hard


def embed(image_bgr, bits, cfg: CodecConfig, mode="raster"):
    """Embed one payload; tile pilots are generated solely from config."""
    _validate_image(image_bgr)
    bits = np.asarray(bits)
    if bits.ndim != 1 or len(bits) != cfg.payload_bits or not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must be a binary vector of cfg.payload_bits elements")
    if mode not in ("raster", "tile"):
        raise ValueError("embed mode must be raster or tile")
    bits = bits.astype(np.uint8)
    yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
    h, w = yuv.shape[0] // 4 * 4, yuv.shape[1] // 4 * 4
    ll, (horizontal, vertical, diagonal) = pywt.dwt2(yuv[:h, :w, 1], "haar")
    blocks = _blocks(ll)
    nr, nc = blocks.shape[:2]
    if mode == "raster":
        carrier_bits = bits[np.arange(nr * nc).reshape(nr, nc) % len(bits)]
    else:
        pilot_slots, pilot_bits, payload_slots, _, _ = _pattern(cfg)
        tile = np.empty(cfg.tile_side ** 2, dtype=np.uint8)
        tile[pilot_slots], tile[payload_slots] = pilot_bits, bits
        rr, cc = np.indices((nr, nc))
        carrier_bits = tile[(rr % cfg.tile_side) * cfg.tile_side + cc % cfg.tile_side]
    position = np.argmax(np.abs(blocks[..., 1:]), axis=-1) + 1
    old = np.take_along_axis(blocks, position[..., None], axis=-1)[..., 0]
    magnitude = (np.abs(old) // cfg.delta + 0.25 + 0.5 * carrier_bits) * cfg.delta
    modified = np.where(old >= 0.0, magnitude, -magnitude)
    np.put_along_axis(blocks, position[..., None], modified[..., None], axis=-1)
    ll[:nr * 4, :nc * 4] = blocks.reshape(nr, nc, 4, 4).transpose(0, 2, 1, 3).reshape(nr * 4, nc * 4)
    details = (vertical, horizontal, diagonal) if cfg.swap_details else (horizontal, vertical, diagonal)
    # Deliberate deployed uint8 assignment, including truncation/wrapping behavior.
    yuv[:h, :w, 1] = pywt.idwt2((ll, details), "haar")
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def _periodic_aggregate(soft, side):
    rr, cc = np.indices(soft.shape)
    slot = ((rr % side) * side + cc % side).ravel()
    size = side * side
    sums = np.bincount(slot, weights=soft.ravel(), minlength=size)
    counts = np.bincount(slot, minlength=size).astype(np.float64)
    return sums, counts


def _score_grid(soft, cfg):
    """Score every tile phase from pilots only, without payload knowledge."""
    sums, counts = _periodic_aggregate(soft, cfg.tile_side)
    signs = _phase_tables(cfg)[1]
    pilot_n = np.abs(signs) @ counts
    correlations = signs @ sums
    scores = np.full(cfg.tile_side ** 2, -np.inf)
    np.divide(correlations, pilot_n, out=scores, where=pilot_n >= 4)
    return scores, pilot_n


def _collect_tile(soft, hard, cfg, r0, c0):
    rr, cc = np.indices(soft.shape)
    slot = ((rr + r0) % cfg.tile_side) * cfg.tile_side + (cc + c0) % cfg.tile_side
    payload_index, signs = _pattern(cfg)[3:]
    mapped = payload_index[slot].ravel()
    valid = mapped >= 0
    counts = np.bincount(mapped[valid], minlength=cfg.payload_bits)
    sums = np.bincount(mapped[valid], weights=soft.ravel()[valid], minlength=cfg.payload_bits)
    evidence = np.divide(sums, counts, out=np.zeros(cfg.payload_bits), where=counts > 0)
    hard_sum = np.bincount(mapped[valid], weights=hard.ravel()[valid], minlength=cfg.payload_bits)
    hard_mean = np.divide(hard_sum, counts, out=np.zeros(cfg.payload_bits), where=counts > 0)
    pilot_mask = signs[slot] != 0
    pilot_n = int(pilot_mask.sum())
    score = float(np.mean(soft[pilot_mask] * signs[slot][pilot_mask])) if pilot_n else 0.0
    return (hard_mean * 255 > 127).astype(np.uint8), evidence, counts, score, pilot_n


def decode(image_bgr, length, cfg: CodecConfig, mode="raster", origin=None):
    """Decode without a reference payload.

    ``origin`` is the original crop's (top, left) and is accepted ONLY by
    tile_oracle. Search enumerates all pixel and tile phases without it.
    ``offset`` reports origin modulo (8 * tile_side); coverage is the
    fraction of payload positions observed at least once. No error
    correction or hypothesis-success decision is implied by status='ok'.
    """
    start = perf_counter()
    _validate_image(image_bgr)
    if length != cfg.payload_bits:
        raise ValueError("length must equal cfg.payload_bits")
    if mode not in ("raster", "tile_fixed", "tile_search", "tile_oracle"):
        raise ValueError("unsupported decode mode")
    if mode != "tile_oracle" and origin is not None:
        raise ValueError("only tile_oracle accepts origin; blind modes cannot receive it")
    if mode == "tile_oracle":
        if origin is None or len(origin) != 2 or any(not isinstance(v, (int, np.integer)) or v < 0 for v in origin):
            raise ValueError("tile_oracle requires nonnegative integer (top, left)")
    channel = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)[..., 1]
    margin = 0.0
    offset = (0, 0)
    if mode == "raster":
        soft, hard = _observations(channel, delta=cfg.delta)
        index = np.arange(soft.size) % length
        counts = np.bincount(index, minlength=length)
        soft_sum = np.bincount(index, weights=soft.ravel(), minlength=length)
        hard_sum = np.bincount(index, weights=hard.ravel(), minlength=length)
        evidence = np.divide(soft_sum, counts, out=np.zeros(length), where=counts > 0)
        hard_mean = np.divide(hard_sum, counts, out=np.zeros(length), where=counts > 0)
        # Exact deployed rule; the half-vote tie is intentionally classified 1.
        bits = (hard_mean * 255 > 127).astype(np.uint8)
        pilot_score, pilot_n = 0.0, 0
    else:
        if mode == "tile_search":
            candidates = []
            for py in range(8):
                for px in range(8):
                    soft_grid, hard_grid = _observations(channel, py, px, cfg.delta)
                    if soft_grid.size == 0:
                        continue
                    scores, _ = _score_grid(soft_grid, cfg)
                    for phase in np.flatnonzero(np.isfinite(scores)):
                        candidates.append((float(scores[phase]), py, px, int(phase)))
            if not candidates:
                raise ValueError("insufficient pilot observations for blind phase search")
            candidates.sort(key=lambda item: item[0], reverse=True)
            best, py, px, phase = candidates[0]
            margin = float(best - candidates[1][0]) if len(candidates) > 1 else 0.0
            r0, c0 = divmod(phase, cfg.tile_side)
            period = 8 * cfg.tile_side
            offset = ((8 * r0 - py) % period, (8 * c0 - px) % period)
        elif mode == "tile_oracle":
            top, left = map(int, origin)
            py, px = (-top) % 8, (-left) % 8
            r0 = ((top + py) // 8) % cfg.tile_side
            c0 = ((left + px) // 8) % cfg.tile_side
            offset = (top % (8 * cfg.tile_side), left % (8 * cfg.tile_side))
        else:
            py, px, r0, c0 = 0, 0, 0, 0
        soft, hard = _observations(channel, py, px, cfg.delta)
        if soft.size == 0:
            raise ValueError("crop has no complete carrier block at the selected alignment")
        bits, evidence, counts, pilot_score, pilot_n = _collect_tile(soft, hard, cfg, r0, c0)
    coverage = float(np.mean(counts > 0))
    status = "ok" if coverage == 1.0 else "partial_payload"
    if mode != "raster" and pilot_n < 4:
        status = "insufficient_pilots"
    return {"bits": bits, "soft": np.clip(evidence.astype(np.float64), -1.0, 1.0),
            "pilot_score": float(pilot_score), "phase_margin": margin,
            "coverage": coverage, "offset": tuple(map(int, offset)),
            "elapsed_s": float(perf_counter() - start), "status": status,
            "pilot_observations": pilot_n}
