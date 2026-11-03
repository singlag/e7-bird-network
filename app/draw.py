"""Overlay drawing: crop rectangle and detection boxes on the original frame."""

from pipeline import MODEL_HEIGHT, MODEL_WIDTH, Crop, model_to_source

BOX_COLOR = (255, 60, 60)
CROP_COLOR = (255, 215, 0)
TEXT_BACK = (255, 60, 60)
TEXT_FORE = (255, 255, 255)


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


def draw_overlay(image, mapped, crop: Crop, show_crop=True, show_labels=True, scale_for_display=None):
    """Return a copy of `image` with the crop rectangle and boxes drawn on it.

    `scale_for_display` scales the drawn line widths / label fonts so an
    enlarged preview stays legible.
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
    for x, y, w, h, score, cls in mapped:
        left = x - w / 2.0
        top = y - h / 2.0
        right = x + w / 2.0
        bottom = y + h / 2.0
        draw.rectangle([left, top, right, bottom], outline=BOX_COLOR, width=box_width)
        if not show_labels:
            continue
        label = f"slot{cls} · {score}"
        box = draw.textbbox((left, top), label, font=font)
        text_height = box[3] - box[1] + 4
        draw.rectangle([left, top - text_height, left + (box[2] - box[0]) + 6, top], fill=TEXT_BACK)
        draw.text((left + 3, top - text_height + 1), label, fill=TEXT_FORE, font=font)
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
