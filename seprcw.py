"""Stable block watermark embedding with pilot-guided recovery after cropping.

One bit is embedded per 8x8 block by quantisation index modulation (QIM) of a
mid-frequency DCT coefficient. Blocks are grouped into BxB tiles that repeat
over the image. Some tile slots carry known pilot bits; the remaining slots
carry the payload. After cropping, the unknown pixel/tile offset is found by
trying all offsets and keeping the one that best matches the pilot bits.
Payload bits are then recovered by majority vote across repeated tiles.
"""
import numpy as np

BLOCK = 8
TILE = 4
COEF = (3, 2)
STEP = 24.0
_rng = np.random.RandomState(12345)
_perm = _rng.permutation(TILE * TILE)
PILOT_SLOTS = np.sort(_perm[:6])
DATA_SLOTS = np.sort(_perm[6:])
PILOT_BITS = _rng.randint(0, 2, len(PILOT_SLOTS))
CAPACITY = len(DATA_SLOTS)


def _dct_matrix(n=BLOCK):
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    m[0] /= np.sqrt(2)
    return m


_D = _dct_matrix()


def _slot_bits(payload):
    bits = np.zeros(TILE * TILE, dtype=int)
    bits[PILOT_SLOTS] = PILOT_BITS
    bits[DATA_SLOTS] = payload
    return bits.reshape(TILE, TILE)


def _check_payload(payload):
    payload = np.asarray(payload, dtype=int).ravel()
    if payload.size != CAPACITY or not np.isin(payload, (0, 1)).all():
        raise ValueError("payload must be %d bits (0/1)" % CAPACITY)
    return payload


def embed(image, payload):
    """Embed payload bits into a 2-D grayscale image; returns float array."""
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError("image must be 2-D grayscale")
    tile_bits = _slot_bits(_check_payload(payload))
    out = img.copy()
    u, v = COEF
    for by in range(img.shape[0] // BLOCK):
        for bx in range(img.shape[1] // BLOCK):
            ys, xs = by * BLOCK, bx * BLOCK
            blk = img[ys:ys + BLOCK, xs:xs + BLOCK]
            c = _D @ blk @ _D.T
            bit = tile_bits[by % TILE, bx % TILE]
            q = np.floor(c[u, v] / STEP)
            # centre of the cell whose parity equals the bit
            if int(q) % 2 != bit:
                q += 1 if (c[u, v] / STEP - q) >= 0.5 else -1
            c[u, v] = (q + 0.5) * STEP
            out[ys:ys + BLOCK, xs:xs + BLOCK] = _D.T @ c @ _D
    return out


def _extract_grid(img, oy, ox):
    """Return (parity-bit grid) of blocks starting at pixel offset (oy, ox)."""
    u, v = COEF
    ny = (img.shape[0] - oy) // BLOCK
    nx = (img.shape[1] - ox) // BLOCK
    bits = np.zeros((ny, nx), dtype=int)
    for by in range(ny):
        for bx in range(nx):
            ys, xs = oy + by * BLOCK, ox + bx * BLOCK
            c = _D @ img[ys:ys + BLOCK, xs:xs + BLOCK] @ _D.T
            bits[by, bx] = int(np.floor(c[u, v] / STEP)) % 2
    return bits


def extract(image):
    """Recover payload from a (possibly cropped) image.

    Returns (payload_bits, (pixel_offset_y, pixel_offset_x, tile_dy, tile_dx),
    pilot_match_fraction).
    """
    img = np.asarray(image, dtype=float)
    pilot_y, pilot_x = np.divmod(PILOT_SLOTS, TILE)
    best = None
    for oy in range(BLOCK):
        for ox in range(BLOCK):
            grid = _extract_grid(img, oy, ox)
            if grid.shape[0] < TILE or grid.shape[1] < TILE:
                continue
            for ty in range(TILE):
                for tx in range(TILE):
                    # block (by, bx) has tile slot ((by+ty)%T, (bx+tx)%T)
                    votes = np.zeros((TILE, TILE))
                    cnt = np.zeros((TILE, TILE))
                    for sy in range(TILE):
                        for sx in range(TILE):
                            sub = grid[(sy - ty) % TILE::TILE, (sx - tx) % TILE::TILE]
                            votes[sy, sx] = sub.sum()
                            cnt[sy, sx] = sub.size
                    ones = votes[pilot_y, pilot_x]
                    n = cnt[pilot_y, pilot_x]
                    agree = np.where(PILOT_BITS == 1, ones, n - ones).sum() / n.sum()
                    if best is None or agree > best[0]:
                        best = (agree, oy, ox, ty, tx, votes.copy(), cnt.copy())
    if best is None:
        raise ValueError("image too small to recover watermark")
    agree, oy, ox, ty, tx, votes, cnt = best
    dy, dx = np.divmod(DATA_SLOTS, TILE)
    payload = (votes[dy, dx] * 2 > cnt[dy, dx]).astype(int)
    return payload, (oy, ox, ty, tx), float(agree)
