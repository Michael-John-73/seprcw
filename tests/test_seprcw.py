import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import seprcw


def _setup():
    rng = np.random.RandomState(0)
    img = rng.randint(40, 215, (128, 128)).astype(float)
    payload = rng.randint(0, 2, seprcw.CAPACITY)
    return img, payload, seprcw.embed(img, payload)


def test_roundtrip():
    _, payload, wm = _setup()
    out, _, match = seprcw.extract(wm)
    assert (out == payload).all() and match == 1.0


def test_cropped_recovery():
    _, payload, wm = _setup()
    for cy, cx in [(3, 5), (13, 22), (37, 50)]:
        cropped = wm[cy:, cx:][:, :100]
        out, _, match = seprcw.extract(cropped)
        assert (out == payload).all() and match > 0.95
