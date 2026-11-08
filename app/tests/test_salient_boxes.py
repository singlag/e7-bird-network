"""Saliency frames for NET4/6/7/9: grid math, fallbacks, list contract.

Script-style test: `python app/tests/test_salient_boxes.py` (e7env).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import draw  # noqa: E402
import pipeline  # noqa: E402
import saliency  # noqa: E402


def heads_with(channels, plane, blob):
    """head0 (1, C, 16, 24) with `blob[(row, col)] = value` stamped on `plane`."""
    arr = np.full((1, channels, saliency.GRID_ROWS, saliency.GRID_COLS), 0.0, np.float32)
    for (row, col), value in blob.items():
        arr[0, plane, row, col] = value
    return {"head0": arr}


def main():
    # 1. one blob -> one model-space frame around the blob cells
    blob = {(4, 10): 1.0, (5, 11): 0.9, (6, 13): 0.8, (5, 12): 0.7}
    recs = saliency.salient_records(heads_with(24, 8, blob), "z8-net9")
    assert len(recs) == 1, recs
    cx, cy, w, h, score_value, cls = recs[0]
    # cells cols 10..13 rows 4..6 -> left 160 top 64 w 64 h 48 (16 px cells)
    assert (cx, cy, w, h) == (192.0, 88.0, 64.0, 48.0), (cx, cy, w, h)
    assert cls == -1, cls
    assert 0 < score_value <= 1000, score_value
    # frame stays inside the model input
    assert 0 <= cx - w / 2 and cx + w / 2 <= saliency.MODEL_W
    assert 0 <= cy - h / 2 and cy + h / 2 <= saliency.MODEL_H

    # 2. every studied NET has its plane and reacts to its own channel
    for key, plane in saliency.SALIENT_OBJ_PLANE.items():
        n_channels = {"z8-net4": 30, "z8-net6": 26, "z8-net7": 24, "z8-net9": 24}[key]
        recs = saliency.salient_records(
            heads_with(n_channels, plane, {(2, 3): 1.0, (3, 4): 0.9}), key)
        assert len(recs) == 1, (key, recs)
        # a blob on a DIFFERENT plane must not light up the frame
        recs_off = saliency.salient_records(
            heads_with(n_channels, (plane + 1) % n_channels, {(2, 3): 1.0, (3, 4): 0.9}), key)
        # off-plane map is all-zero (degenerate) -> no frame
        assert recs_off == [], (key, recs_off)

    # 3. models without a studied frame produce nothing
    assert saliency.salient_records(heads_with(66, 4, {(0, 0): 1.0}), "z8-net5") == []
    assert saliency.salient_records(None, "z8-net9") == []
    assert saliency.salient_records({}, "z8-net9") == []
    assert saliency.salient_records({"head1": np.zeros((1, 24, 16, 24))}, "z8-net9") == []

    # 4. degenerate maps: flat / NaN / wrong shape -> no frame
    flat = {"head0": np.full((1, 24, 16, 24), 3.0, np.float32)}
    assert saliency.salient_records(flat, "z8-net9") == []
    nans = {"head0": np.full((1, 24, 16, 24), np.nan, np.float32)}
    assert saliency.salient_records(nans, "z8-net9") == []
    short = {"head0": np.zeros((1, 8, 16, 24), np.float32)}  # plane 8 out of range
    assert saliency.salient_records(short, "z8-net9") == []
    wrong_grid = {"head0": np.zeros((1, 24, 8, 12), np.float32)}
    assert saliency.salient_records(wrong_grid, "z8-net9") == []

    # 5. diffuse "everything is hot" maps fall back to the single strongest cell
    rows, cols = np.mgrid[0:16, 0:24]
    gaussian = np.exp(-(((rows - 8) ** 2 + (cols - 12) ** 2) / (2 * 10.0 ** 2)))
    diffuse = {"head0": np.stack([gaussian.astype(np.float32)] * 24)}
    recs = saliency.salient_records(diffuse, "z8-net9")
    assert len(recs) == 1, recs
    cx, cy, w, h, _, _ = recs[0]
    assert (w, h) == (16.0, 16.0), (w, h)   # one cell, not a full-frame rectangle
    assert (cx, cy) == (12.5 * 16, 8.5 * 16), (cx, cy)  # peak is at row 8 col 12

    # near-flat with one peak also degrades to one cell
    peak = np.zeros((16, 24), np.float32)
    peak[1, 2] = 1.0
    recs = saliency.salient_records(
        {"head0": np.stack([peak] * 24)}, "z8-net9")
    assert len(recs) == 1
    assert recs[0][2:] == [16.0, 16.0, 1000.0, -1], recs[0]

    # 6. list contract: frames go through draw.map_records (raw NNIN identity crop)
    crop = pipeline.Crop(0, 0, saliency.MODEL_W, saliency.MODEL_H, "raw_nnin")
    mapped = draw.map_records(recs, crop)
    assert len(mapped) == 1
    x, y, bw, bh, m_score, m_cls = mapped[0]
    assert (bw, bh) == (16, 16) and m_cls == -1, mapped[0]

    print("salient boxes ok")


if __name__ == "__main__":
    main()
