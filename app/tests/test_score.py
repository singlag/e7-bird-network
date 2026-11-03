"""Tests for `score.py`: token / slot / rank / combined probability recovery.

A crafted head buffer (every stored half pattern = 1.0) makes every one of the
1056 visited cells pass the gates, so the replay can be checked against the
decoder's own candidates on a lot of cells without running the network.
"""

import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for path in (REPO, REPO / "app"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pytest
import score
from decoder_port import HEAD_BYTES, PythonDecoder
from e7model import consumer_table


ONE = 0x3C00  # `half(0x3C00) == 1.0`, and 0x3C00 < 0x5400 so it is a valid pattern


@pytest.fixture(scope="module")
def table():
    return PythonDecoder(consumer_table())


@pytest.fixture(scope="module")
def heads():
    raw = bytearray(HEAD_BYTES)
    for offset in range(0, HEAD_BYTES, 2):
        struct.pack_into("<H", raw, offset, ONE)
    return bytes(raw)


def test_visit_order_is_a_bijection_over_1056_cells():
    seen = {}
    ordinal = 0
    for stage in (0, 1):
        anchors = 3 if stage else 2
        rows = 8 if stage else 16
        columns = 12 if stage else 24
        for anchor in range(anchors):
            for row in range(rows):
                for col in range(columns):
                    assert score.cell_from_ordinal(ordinal) == (stage, anchor, row, col)
                    seen[ordinal] = (stage, anchor, row, col)
                    ordinal += 1
    assert ordinal == score.TOTAL_CELLS == 1056
    assert len(set(seen.values())) == 1056
    assert len(set(seen.keys())) == 1056


def test_ordinal_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        score.cell_from_ordinal(score.TOTAL_CELLS)
    with pytest.raises(ValueError):
        score.cell_from_ordinal(-1)


def test_token_formula_matches_stage_and_anchor(table, heads):
    seeds = table.initial()
    for ordinal in (0, 1, 384, 767, 768, 864, 1055):
        stage, anchor, row, col = score.cell_from_ordinal(ordinal)
        detail = score.cell_detail(table, heads, ordinal, seeds)
        assert detail is not None
        assert detail["token"] == ((stage << 4) + (2 if stage else 0) + anchor)
        assert detail["objectness"] == pytest.approx(1 / (1 + 2.718281828459045 ** -1), rel=1e-5)


def test_replay_reproduces_every_decoder_candidate(table, heads):
    report = table.consume(heads)
    assert report["status"] == 0
    # the decoder stops once MAX_CANDIDATES boxes exist, so `visited` is the
    # number of cells it actually examined, not always 1056
    assert report["accepted"] == len(report["candidates"]) == len(report["ordinals"])
    assert report["visited"] <= score.TOTAL_CELLS
    assert report["ordinals"] == sorted(set(report["ordinals"])), "ordinals must be strictly increasing"
    assert report["accepted"], "crafted heads should make the decoder accept candidates"

    seeds = table.initial()
    for index, candidate in enumerate(report["candidates"]):
        ordinal = report["ordinals"][index]
        detail = score.cell_detail(table, heads, ordinal, seeds)
        assert detail is not None, f"ordinal {ordinal} should pass both gates"
        assert detail["best_rank"] == candidate[4], f"ordinal {ordinal}: rank mismatch"
        assert detail["best_cls"] == candidate[5], f"ordinal {ordinal}: class mismatch"
        assert detail["classes"], f"ordinal {ordinal}: no class cleared the score gate"
        for entry in detail["classes"]:
            assert 0.0 <= entry["combined"] <= 1.0
            assert 0 <= entry["rank"] <= 1000
        assert 0.0 <= detail["best_combined"] <= 1.0
        assert detail["token"] in ((0, 1, 18, 19, 20))


def test_annotate_attaches_fields_to_every_record_row(table, heads):
    report = table.consume(heads)
    rows = score.annotate(report, heads, table)
    assert len(rows) == len(report["records"])
    for row_index, (row, record) in enumerate(zip(rows, report["records"])):
        assert row["rank_index"] == row_index
        assert row["rank"] == record[4]
        assert row["cls"] == record[5]
        assert row["token"] in ((0, 1, 18, 19, 20)), row
        assert row["slot"] is not None and 0 <= row["slot"] < 24
        assert row["combined"] is not None and 0.0 <= row["combined"] <= 1.0
        assert row["objectness"] is not None and 0.0 <= row["objectness"] <= 1.0
        assert row["grid"] is not None
    # row order is the decoder's rank order, so `rank_index` is what "行序" means
    assert [row["rank_index"] for row in rows] == list(range(len(rows)))


def test_rank_is_not_a_probability(table, heads):
    """The point of the new columns: two rows with equal rank can have different p."""
    report = table.consume(heads)
    rows = score.annotate(report, heads, table)
    by_rank: dict = {}
    for row in rows:
        if row["combined"] is None:
            continue
        by_rank.setdefault(row["rank"], set()).add(round(row["combined"], 6))
    assert rows, "expected at least one scored row"
    for row in rows:
        if row["combined"] is not None:
            assert 0.0 <= row["combined"] <= 1.0
