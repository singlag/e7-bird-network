"""Research-only saliency frames for Z8 NETs without a published decode contract.

NET4/6/7/9 run inference but have no consumer table / rank anchors, so the
decoder cannot emit candidate boxes. The external-sample study (Nov 2026,
43 3D-tracking + 92 AF ground-truth boxes from NEF AFInfo2 tags vs a
permutation null) found one objectness-like head0 plane per NET whose
high-activation cells are significantly enriched inside the camera's own
recorded subject boxes — significant in BOTH subject groups (p <= 0.005):

    NET4 plane 4 (also 19)   NET6 plane 3 (also 9,16,17)
    NET7 plane 17 (also 9,22)   NET9 plane 8 (also 1,13,19)

We turn the activation map of that plane into ONE coarse bounding frame so the
preview screen shows where the network is "looking". This is explicitly NOT
the decode contract: there are no class thresholds, no NMS and no rank curve,
the box is a rectangle around a group of 16x16-px grid cells. Rows generated
here carry class = -1 ("SAL" in the detection list) so they are never mixed
up with decoder output.
"""

import math

SALIENT_OBJ_PLANE = {"z8-net4": 4, "z8-net6": 3, "z8-net7": 17, "z8-net9": 8}

# head0 is a 24 x 16 cell grid over the 384 x 256 model input
GRID_COLS, GRID_ROWS = 24, 16
CELL_PX = 16
MODEL_W, MODEL_H = GRID_COLS * CELL_PX, GRID_ROWS * CELL_PX

# a cell counts when it is at least half of the map's max AND clearly above
# the map mean; if nothing passes, the single argmax cell is used.
HIGH_ABS = 0.5
HIGH_SIGMA = 2.0
# a rectangle covering more than this share of the frame is the model shouting
# "everything" — fall back to the single strongest cell instead.
MAX_AREA_SHARE = 0.6


def salient_records(heads, model_key):
    """Model-space frames ``[[cx, cy, w, h, confidence*1000, -1]]`` for a NET head set.

    Returns ``[]`` for models without a studied plane, missing heads or
    degenerate (all-equal / non-finite) activation maps.
    """
    plane = SALIENT_OBJ_PLANE.get(model_key)
    if plane is None or not heads:
        return []
    head0 = heads.get("head0")
    if head0 is None:
        return []
    try:
        import numpy as np
    except ImportError:  # numpy is a runtime dependency; keep the caller alive
        return []
    arr = np.asarray(head0)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[0] <= plane:
        return []
    channel = arr[plane].astype(np.float32)
    if channel.shape != (GRID_ROWS, GRID_COLS):
        return []
    finite = channel[~np.isnan(channel)]
    if finite.size == 0:
        return []
    lo, hi = float(finite.min()), float(finite.max())
    if not math.isfinite(hi - lo) or hi - lo <= 0:
        return []
    norm = (channel - lo) / (hi - lo)
    mean = float(norm.mean())
    sigma = float(norm.std())
    gate = max(HIGH_ABS, mean + HIGH_SIGMA * sigma)
    ys, xs = np.nonzero(norm >= gate)
    area_share = (len(ys) * CELL_PX * CELL_PX) / float(MODEL_W * MODEL_H)
    if len(ys) == 0 or area_share > MAX_AREA_SHARE:
        flat = int(np.argmax(norm))
        ys = np.array([flat // GRID_COLS])
        xs = np.array([flat % GRID_COLS])
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    left, top = x0 * CELL_PX, y0 * CELL_PX
    width, height = (x1 - x0 + 1) * CELL_PX, (y1 - y0 + 1) * CELL_PX
    confidence = float(norm[ys, xs].mean())
    cx, cy = left + width / 2.0, top + height / 2.0
    # score field rides the decoder's 0..1000 display scale so the list column
    # reads consistently; the raw confidence is its /1000.
    return [[cx, cy, width, height, confidence * 1000.0, -1]]
