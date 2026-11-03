"""Overlay drawing: crop rectangle and detection boxes on the original frame."""

from pipeline import MODEL_HEIGHT, MODEL_WIDTH, Crop, model_to_source

BOX_COLOR = (255, 60, 60)
SELECT_COLOR = (255, 235, 90)
CROP_COLOR = (255, 215, 0)
TEXT_BACK = (255, 60, 60)
TEXT_FORE = (255, 255, 255)
FADE_COLOR = (128, 128, 128)         # what unreselected areas are blended toward
DEFAULT_KEEP = 0.35                  # how much of the original survives outside the selected box


def dim_outside(image, box, keep: float = DEFAULT_KEEP):
    """Fade everything outside `box` (source-pixel centre/size), keeping `box` at 100%.

    `keep` is how much of the original survives outside the box: 1.0 = no fade.
    Blending toward a mid grey both drops saturation and lowers contrast, which
    reads as a pale ("淡色") background rather than a shadow.
    """
    from PIL import Image

    canvas = image.convert("RGB").copy()
    if box is None or keep >= 1.0:
        return canvas
    x, y, w, h = box[0], box[1], box[2], box[3]
    left = max(0, int(round(x - w / 2.0)))
    top = max(0, int(round(y - h / 2.0)))
    right = min(canvas.width, int(round(x + w / 2.0)))
    bottom = min(canvas.height, int(round(y + h / 2.0)))
    if right <= left or bottom <= top:
        return canvas
    faded = Image.blend(canvas, Image.new("RGB", canvas.size, FADE_COLOR), 1.0 - float(keep))
    out = faded.copy()
    out.paste(canvas.crop((left, top, right, bottom)), (left, top))
    return out


def _dashed_rectangle(draw, left, top, right, bottom, color, dash=10, gap=7, width=2):
    """Draw a dashed rectangle; Tk/PIL have no dashed outline primitive."""
    def dashed_line(x0, y0, x1, y1):
        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if length == 0:
            return
        step = dash + gap
        travelled = 0.0
        while travelled < length:
            remainder = min(dash, length - travelled)
            draw.line(
                [(x0 + (x1 - x0) * travelled / length, y0 + (y1 - y0) * travelled / length),
                 (x0 + (x1 - x0) * (travelled + remainder) / length, y0 + (y1 - y0) * (travelled + remainder) / length)],
                fill=color,
                width=width,
            )
            travelled += step

    dashed_line(left, top, right, top)
    dashed_line(right, top, right, bottom)
    dashed_line(right, bottom, left, bottom)
    dashed_line(left, bottom, left, top)


def map_records(records, crop: Crop):
    """Model-space records `[cx, cy, w, h, score, cls]` -> source-pixel boxes."""
    mapped = []
    for record in records:
        cx, cy, w, h, score, cls = record
        x, y, bw, bh = model_to_source(cx, cy, w, h, crop)
        mapped.append((x, y, bw, bh, int(score), int(cls)))
    return mapped


def _label_text(index: int, scored: dict | None) -> str:
    """`#1 · p 0.168` — the preview only says which box it is and how sure we are.

    Class, rank, token and slot are deliberately not drawn here: they are one
    click away in the detection list, and long labels made overlapping boxes
    unreadable.  `p` is the uncalibrated combined probability, so boxes from
    different slots stay comparable; `-` means no score data for that row.
    """
    combined = scored.get("combined") if scored else None
    if combined is None:
        return f"#{index} · p -"
    return f"#{index} · p {float(combined):.3f}"


def draw_overlay(image, mapped, crop: Crop, show_crop=True, show_labels=True, scale_for_display=None,
                 highlight=None, scored=None):
    """Return a copy of `image` with the crop rectangle and boxes drawn on it.

    `scale_for_display` scales the drawn line widths / label fonts so an
    enlarged preview stays legible.  `highlight` marks one box index as the
    selected one (bright, thicker) and hides every other box, so overlapping
    detections stop competing once the user has picked one; `scored` (from
    `score.annotate`) adds probability / token to each label.
    """
    from PIL import ImageDraw

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    factor = scale_for_display or 1.0
    line_width = max(1, int(round(2 * factor)))
    box_width = max(2, int(round(3 * factor)))

    if show_crop and crop.mode != "none":
        _dashed_rectangle(
            draw,
            crop.x,
            crop.y,
            crop.x + crop.width,
            crop.y + crop.height,
            CROP_COLOR,
            dash=max(6, int(10 * factor)),
            gap=max(4, int(7 * factor)),
            width=line_width,
        )

    font = _load_font(max(12, int(round(14 * factor))))
    for index, (x, y, w, h, score, cls) in enumerate(mapped):
        if highlight is None:
            selected = False
            color, width = BOX_COLOR, box_width
        else:
            if index != highlight:
                continue        # overlapping boxes disappear once one is selected
            selected = True
            color = SELECT_COLOR
            width = max(box_width, int(round(box_width * 1.6)))
        left = x - w / 2.0
        top = y - h / 2.0
        right = x + w / 2.0
        bottom = y + h / 2.0
        draw.rectangle([left, top, right, bottom], outline=color, width=width)
        if not show_labels:
            continue
        label = _label_text(index, scored[index] if scored and index < len(scored) else None)
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0] + 6
        text_height = box[3] - box[1] + 4
        # A box touching the right edge would otherwise clip the probability off
        # the end of the string, so slide the label left until it fits.  A box at
        # the top edge gets its label drawn just inside the box instead of above it.
        label_left = min(left, max(0.0, canvas.width - text_width))
        label_top = top - text_height
        if label_top < 0:
            label_top = min(max(0.0, bottom - text_height), top + width)
        draw.rectangle([label_left, label_top, label_left + text_width, label_top + text_height],
                       fill=color if selected else TEXT_BACK)
        draw.text((label_left + 3, label_top + 1), label, fill=TEXT_FORE, font=font)
    return canvas


def _load_font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans-Bold.ttf", "Arial.ttf", "Helvetica.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def model_grid_note(records) -> str:
    return f"{len(records)} candidate box(es) decoded from the model heads ({MODEL_WIDTH}x{MODEL_HEIGHT} space)"
