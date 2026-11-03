"""Per-box scoring detail that the decoder's 6-field result row does not expose.

`wide_reference.c` returns only ``[cx, cy, w, h, score, cls]`` per row, and
``score`` is a **0–1000 rank** read off the calibration curve of whichever token
slot won that cell.  Two rows therefore compare as ranks, not as
probabilities — a rank 999 from one slot is not "more certain" than a rank 999
from another slot.  This module replays the same arithmetic the decoder uses
(:mod:`decoder_port`, which is the bit-exact reference) so that every record row
also carries:

``token``    which head plane produced the box: ``(stage << 4) + (2 if stage else 0) + anchor``
``slot``     index k (0–23) of the token slot whose curve produced the rank
``rank``     the 0–1000 calibration rank (the number drawn on the box today)
``combined`` combined probability ``sigmoid(class_logit) * objectness`` *before* calibration
``obj``      objectness ``sigmoid(objectness_logit)``
``classes``  every class that cleared the score gate, with its own rank/combined

It works for both backends: the native library and the Python port both report
``ordinals`` aligned with their accepted candidates, and an ordinal maps straight
back to the ``(stage, anchor, row, col)`` cell the decoder visited.
"""

from decoder_port import (
    PythonDecoder,
    clip,
    half,
    mul,
    rank,
    read_head,
    sigmoid,
)

# Visit order in `reference_consume`: stage 0 = 2 anchors x 16 rows x 24 cols,
# stage 1 = 3 anchors x 8 rows x 12 cols, anchors outermost.
FINE_CELLS = 2 * 16 * 24      # 768
COARSE_CELLS = 3 * 8 * 12     # 288
TOTAL_CELLS = FINE_CELLS + COARSE_CELLS  # 1056, matches `report.visited`


def cell_from_ordinal(ordinal: int):
    """Visit ordinal -> ``(stage, anchor, row, col)`` (inverse of the scan order)."""
    if ordinal < FINE_CELLS:
        stage, base = 0, ordinal
        anchors, rows, columns = 2, 16, 24
    else:
        stage, base = 1, ordinal - FINE_CELLS
        anchors, rows, columns = 3, 8, 12
    if base < 0 or base >= anchors * rows * columns:
        raise ValueError(f"ordinal {ordinal} outside the 0..{TOTAL_CELLS - 1} visit range")
    anchor_cells = rows * columns
    return stage, base // anchor_cells, (base % anchor_cells) // columns, base % columns


def cell_detail(table: PythonDecoder, heads: bytes, ordinal: int, seeds=None) -> dict | None:
    """Score one visited cell exactly as `reference_consume` does.

    Returns ``None`` when the cell was skipped because objectness fell below the
    gate; ``{"error": ...}`` when a stored half pattern is out of range.
    """
    stage, anchor, row, col = cell_from_ordinal(ordinal)
    seeds = table.initial() if seeds is None else seeds

    raw = read_head(heads, stage, anchor, 4, row, col)
    if (raw & 32767) > 0x5400:
        return {"stage": stage, "anchor": anchor, "row": row, "col": col, "error": 4}
    objectness = float(sigmoid(half(raw)))
    token = (stage << 4) + (2 if stage else 0) + anchor
    detail = {
        "stage": stage,
        "anchor": anchor,
        "row": row,
        "col": col,
        "token": token,
        "objectness": objectness,
        "classes": [],
        "best_slot": None,
        "best_cls": None,
        "best_rank": 0,
        "best_combined": 0.0,
    }
    if table.objectness_gate > objectness:
        return None

    for category in range(6):
        wanted = (category << 8) + token
        slot = next((k for k in range(24) if table.tokens[k] == wanted), None)
        if slot is None:
            continue
        raw = read_head(heads, stage, anchor, 5 + category, row, col)
        if (raw & 32767) > 0x5400:
            detail["error"] = 5 + category
            return detail
        combined = float(mul(sigmoid(half(raw)), objectness))
        if table.score_gate > combined:
            continue
        found = rank(table.curves[slot], combined, seeds[slot] - 1)
        detail["classes"].append(
            {"cls": category, "slot": slot, "rank": found, "combined": combined}
        )
        if found > detail["best_rank"]:
            detail["best_rank"] = found
            detail["best_cls"] = category
            detail["best_slot"] = slot
            detail["best_combined"] = combined
    return detail


def annotate(report: dict, heads: bytes, table: PythonDecoder) -> list:
    """Attach scoring detail to `report["records"]`, in the decoder's row order.

    Record rows are matched back to their accepted candidate through the
    candidate's clipped geometry + class, because suppression can zero a row's
    score while it still occupies an output slot.  A row with no match keeps
    ``None`` fields rather than a guess.
    """
    candidates = report.get("candidates") or []
    ordinals = report.get("ordinals") or []
    seeds = table.initial()

    by_geometry: dict = {}
    for index, candidate in enumerate(candidates):
        ordinal = ordinals[index] if index < len(ordinals) else None
        if ordinal is None:
            continue
        clipped = clip(candidate)
        if clipped is None:
            continue
        detail = cell_detail(table, heads, ordinal, seeds)
        if detail is None:
            continue
        key = (clipped[0], clipped[1], clipped[2], clipped[3], clipped[5])
        detail["ordinal"] = ordinal
        by_geometry.setdefault(key, detail)

    scored: list = []
    for row_index, record in enumerate(report.get("records") or []):
        cx, cy, w, h, score, cls = record
        detail = by_geometry.get((int(cx), int(cy), int(w), int(h), int(cls)))
        row = {
            "rank_index": row_index,
            "cx": int(cx), "cy": int(cy), "w": int(w), "h": int(h),
            "rank": int(score), "cls": int(cls),
            "token": None, "slot": None, "combined": None, "objectness": None,
            "stage": None, "anchor": None, "grid": None, "classes": [],
        }
        if detail is not None:
            row.update(
                {
                    "token": detail["token"],
                    "slot": detail["best_slot"],
                    "combined": detail["best_combined"],
                    "objectness": detail["objectness"],
                    "stage": detail["stage"],
                    "anchor": detail["anchor"],
                    "grid": [detail["row"], detail["col"]],
                    "classes": detail["classes"],
                    "ordinal": detail.get("ordinal"),
                }
            )
        scored.append(row)
    return scored
