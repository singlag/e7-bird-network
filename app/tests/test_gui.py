"""GUI wiring tests: title, language switch, detection list, selected-box fade.

No inference runs here — boxes are injected — so the test is fast and works on
any platform with a display.  It skips when Tkinter cannot open a display.
"""

import sys
import tkinter as tk
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for path in (REPO, REPO / "app"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from PIL import Image

import draw
import gui
import pipeline


@pytest.fixture()
def app():
    instance = None
    try:
        instance = gui.App()
    except tk.TclError as error:  # headless box with no X server
        pytest.skip(f"no display available: {error}")
    assert instance is not None
    instance.update_idletasks()
    instance.update()
    yield instance
    instance.destroy()


def synthetic_frame(width=1200, height=900):
    """A gradient image, so faded pixels are guaranteed to differ from originals."""
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for x in range(width):
        for y in range(height):
            pixels[x, y] = ((x * 7 + y * 3) % 256, (x * 5 + y * 11) % 256, (x + y * 13) % 256)
    return image


BOXES = [
    (360.0, 360.0, 300.0, 220.0, 974, 0),
    (660.0, 405.0, 240.0, 270.0, 991, 4),
    (820.0, 500.0, 180.0, 150.0, 898, 2),
]
SCORED = [
    {"token": 1, "slot": 3, "combined": 0.8742, "objectness": 0.912},
    {"token": 18, "slot": 9, "combined": 0.6124, "objectness": 0.78},
    {"token": 19, "slot": 14, "combined": 0.4310, "objectness": 0.66},
]

# A region that no box, label or dashed crop line touches, so pixels there are
# either untouched (no selection) or faded (selection made).
CLEAN_BAND = range(60, 160, 8), range(120, 230, 8)
# Interior of BOXES[1] away from the other boxes' label bands.
SELECTED_INTERIOR = range(570, 700, 10), range(300, 500, 12)


def _pixels(ranges):
    return [(x, y) for x in ranges[0] for y in ranges[1]]


def test_window_title_is_the_new_name(app):
    assert app.title() == "E7 object detection simulator - GUI"


def test_language_switch_relabels_without_losing_state(app):
    app._set_frame(synthetic_frame(), BOXES, SCORED, pipeline.crop_to_target(1200, 900))
    app._selected = 1
    app.update()

    app.set_language("en")
    app.update()
    assert app._t("run") == "Run"
    assert app.run_button.cget("text") == "Run"
    assert app.tree.heading("score")["text"] == "Rank"
    assert app.tree.heading("p")["text"] == "Prob"
    assert len(app.tree.get_children()) == 3
    assert app.title() == "E7 object detection simulator - GUI"

    app.set_language("zh")
    app.update()
    assert app.run_button.cget("text") == "執行"
    assert app.tree.heading("score")["text"] == "等級"
    assert len(app.tree.get_children()) == 3
    assert app._selected == 1, "selection survives a language rebuild"


def test_list_rows_show_rank_probability_and_token(app):
    app._set_frame(synthetic_frame(), BOXES, SCORED, pipeline.crop_to_target(1200, 900))
    values = [app.tree.item(iid)["values"] for iid in app.tree.get_children()]
    assert [row[0] for row in values] == [0, 1, 2]
    assert values[1][2] == 991
    assert float(values[1][3].rstrip("%")) == pytest.approx(61.2, abs=0.1)
    assert values[1][3].endswith("%"), "probabilities display as 0-100 %"
    assert values[1][4] == 18 and values[1][5] == 9


def test_heading_click_sorts_the_list_and_selection_follows_the_box(app):
    app._set_frame(synthetic_frame(), BOXES, SCORED, pipeline.crop_to_target(1200, 900))

    app._sort_by_heading("score")           # first click: biggest rank first
    assert [iid for iid in app.tree.get_children()] == ["1", "0", "2"]
    assert app.tree.heading("score")["text"] == "等級 ▼"
    assert app.tree.heading("p")["text"] == "機率"

    app._sort_by_heading("score")           # second click: reverse
    assert [iid for iid in app.tree.get_children()] == ["2", "0", "1"]
    assert app.tree.heading("score")["text"] == "等級 ▲"

    app._sort_by_heading("p")               # probabilities, descending first
    assert [iid for iid in app.tree.get_children()] == ["0", "1", "2"]
    assert app.tree.heading("p")["text"] == "機率 ▼"

    app._sort_by_heading("box")             # by box area, descending
    assert [iid for iid in app.tree.get_children()] == ["0", "1", "2"]

    app._sort_by_heading("token")           # ascending first for token/slot
    assert [iid for iid in app.tree.get_children()] == ["0", "1", "2"]
    assert app.tree.heading("token")["text"] == "token ▲"
    app._sort_by_heading("token")           # again: biggest token first
    assert [iid for iid in app.tree.get_children()] == ["2", "1", "0"]
    assert app.tree.heading("token")["text"] == "token ▼"

    # item ids stay tied to the detection, so sorting cannot move the selection
    app.tree.selection_set("1")
    app._tree_selected()
    assert app._selected == 1
    app._sort_by_heading("score")
    assert app._selected == 1
    assert "1" in app.tree.selection(), "the selected row stays selected after a re-sort"


def test_sort_order_survives_a_language_rebuild_and_new_results(app):
    app._set_frame(synthetic_frame(), BOXES, SCORED, pipeline.crop_to_target(1200, 900))
    app._sort_by_heading("score")
    app.set_language("en")
    app.update()
    assert [iid for iid in app.tree.get_children()] == ["1", "0", "2"]
    assert app.tree.heading("score")["text"] == "Rank ▼"
    app._set_frame(synthetic_frame(), BOXES, SCORED, pipeline.crop_to_target(1200, 900))
    assert [iid for iid in app.tree.get_children()] == ["1", "0", "2"]


def test_selecting_a_box_hides_the_other_boxes(app):
    import numpy as np

    frame = synthetic_frame()
    crop = pipeline.crop_to_target(1200, 900)
    app._set_frame(frame, BOXES, SCORED, crop)
    app._selected = None
    app.update()
    unselected = np.asarray(app._compose().convert("RGB"))
    assert int((unselected == np.array(draw.BOX_COLOR)).all(axis=2).sum()) > 0, \
        "without a selection every box is drawn in the normal colour"

    app._selected = 1
    app.update()
    composed = np.asarray(app._compose().convert("RGB"))
    leftovers = int((composed == np.array(draw.BOX_COLOR)).all(axis=2).sum())
    assert leftovers == 0, f"non-selected boxes must not be drawn, {leftovers} pixels remain"
    assert int((composed == np.array(draw.SELECT_COLOR)).all(axis=2).sum()) > 0, \
        "the selected box is still drawn bright"


def test_box_label_never_runs_off_the_frame():
    """A box touching the right edge must not clip the probability off the label."""
    import numpy as np
    from PIL import ImageDraw

    def longest_run(row):
        best = run = 0
        for value in row:
            run = run + 1 if value else 0
            best = max(best, run)
        return best

    frame = Image.new("RGB", (200, 120), (0, 0, 0))
    out = draw.draw_overlay(frame, [(170.0, 60.0, 40.0, 40.0, 700, 1)],
                            pipeline.crop_to_target(200, 120), scale_for_display=1.0,
                            scored=[{"token": 1, "slot": 9, "combined": 0.25}])
    arr = np.asarray(out.convert("RGB"))
    font = draw._load_font(14)
    text_width = ImageDraw.Draw(out).textbbox((0, 0), "#0 · p 25.0%", font=font)[2] + 6

    fill = (arr == np.array(draw.TEXT_BACK)).all(axis=2)
    longest = max(longest_run(row) for row in fill)
    assert longest >= text_width, f"label background clipped to {longest}px, needs {text_width}px"
    white_x = np.nonzero((arr == np.array(draw.TEXT_FORE)).all(axis=2))[1]
    assert len(white_x) > 0 and int(white_x.max()) <= 199, "label text stays inside the canvas"


def test_box_label_shows_only_index_and_probability():
    """The preview label stays short: `#n · p xx.x %`; everything else is in the list."""
    assert draw._label_text(1, SCORED[1]) == "#1 · p 61.2%"
    assert draw._label_text(0, SCORED[0]) == "#0 · p 87.4%"
    assert draw._label_text(2, None) == "#2 · p -"
    assert "rank" not in draw._label_text(1, SCORED[1])
    assert "tok" not in draw._label_text(1, SCORED[1])
    assert "cls" not in draw._label_text(1, SCORED[1])


def test_selected_box_is_100_percent_and_everything_else_is_faded(app):
    frame = synthetic_frame()
    crop = pipeline.crop_to_target(1200, 900)
    app._set_frame(frame, BOXES, SCORED, crop)
    app._selected = 1
    app.update()
    composed = app._compose().convert("RGB")
    base = frame.convert("RGB")

    interior = _pixels(SELECTED_INTERIOR)
    assert interior
    mismatched = [xy for xy in interior if composed.getpixel(xy) != base.getpixel(xy)]
    assert not mismatched, f"box interior must stay at 100%, differs at {mismatched[:5]}"

    band = _pixels(CLEAN_BAND)
    assert band
    faded = [xy for xy in band if composed.getpixel(xy) == base.getpixel(xy)]
    assert not faded, f"everything outside the selected box must be faded, unchanged at {faded[:5]}"

    # the faded pixels really are the gray blend PIL produces, not a black cover
    x, y = band[0]
    original = base.getpixel((x, y))
    assert isinstance(original, tuple)
    strip = Image.new("RGB", (1, 1), original)
    expected = Image.blend(strip, Image.new("RGB", (1, 1), draw.FADE_COLOR), 1.0 - draw.DEFAULT_KEEP).getpixel((0, 0))
    assert composed.getpixel((x, y)) == expected


def test_no_selection_keeps_the_original_pixels(app):
    frame = synthetic_frame()
    app._set_frame(frame, BOXES, SCORED, pipeline.crop_to_target(1200, 900))
    app._selected = None
    composed = app._compose().convert("RGB")
    base = frame.convert("RGB")
    band = _pixels(CLEAN_BAND)
    assert band
    changed = [xy for xy in band if composed.getpixel(xy) != base.getpixel(xy)]
    assert not changed, f"without a selection nothing should be dimmed, changed at {changed[:5]}"


def test_clear_selection_and_stale_index(app):
    app._set_frame(synthetic_frame(), BOXES, SCORED, pipeline.crop_to_target(1200, 900))
    app._selected = 2
    app._clear_selection()
    assert app._selected is None
    # a shorter box list than the stored selection drops the selection
    app._set_frame(synthetic_frame(), BOXES[:1], SCORED[:1], pipeline.crop_to_target(1200, 900))
    app._selected = 2
    app._set_frame(synthetic_frame(), BOXES[:1], SCORED[:1], pipeline.crop_to_target(1200, 900))
    assert app._selected is None


def test_missing_scored_rows_render_as_dashes(app):
    app._set_frame(synthetic_frame(), BOXES, [], pipeline.crop_to_target(1200, 900))
    values = [app.tree.item(iid)["values"] for iid in app.tree.get_children()]
    assert values[0][3] == "-" and values[0][4] == "-" and values[0][5] == "-"


def test_run_snapshots_settings_for_the_worker_thread(monkeypatch, app):
    """Tk variables are main-thread-only: CPython's _tkinter raises
    'main thread is not in main loop' when the worker reads them, so _run()
    must hand the worker plain values instead."""
    import types

    captured = {}

    def fake_run_job(model, source, output, **kwargs):
        captured.update(kwargs)
        captured["output"] = output
        return types.SimpleNamespace(processed=1, detections=0, elapsed=0.0, output=output)

    monkeypatch.setattr(gui.job, "run_job", fake_run_job)
    monkeypatch.setattr(gui.model_registry, "build",
                        lambda key, threads, decoder_mode: types.SimpleNamespace(key=key))
    monkeypatch.setattr(app, "_finish", lambda result, error: None)

    app.source = Path("clip-test.mp4")
    app.per_second_var.set(2.0)
    app.max_frames_var.set(3)
    app.save_frames_var.set(True)
    app.save_heads_var.set(False)
    app.export_video_var.set(False)
    app._run()
    assert app.worker is not None
    app.worker.join(10)
    for _ in range(20):
        app.update()

    assert captured["per_second"] == 2.0
    assert captured["max_frames"] == 3
    assert captured["save_frames"] is True and captured["save_heads"] is False
    assert captured["export_video"] is False
    logged = "\n".join(app._log_lines)
    assert "Traceback" not in logged, logged
    assert "main thread is not in main loop" not in logged
