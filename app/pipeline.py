"""Geometry adapter between a photo / video frame and the E7 bird model input.

The model consumes exactly 294912 unsigned bytes: one planar 3 x 256 x 384
red/green/blue image, i.e. a 3:2 frame.  The original `run.py` stretches any
image to 384 x 256, which distorts it.  This module instead centre-crops to
3:2 first, then resizes, and provides the inverse mapping so detection boxes
can be drawn on the *original* frame.

Crop rule (user-specified):
  * aspect <= 1.5  -> trim HEIGHT (shorten the frame to width/1.5)
  * aspect  > 1.5  -> trimming height would require *adding* pixels, which is
    impossible, so trim WIDTH instead.  The UI reports which one was used.
"""

from dataclasses import dataclass

MODEL_WIDTH = 384
MODEL_HEIGHT = 256
MODEL_BYTES = 3 * MODEL_HEIGHT * MODEL_WIDTH
TARGET_ASPECT = MODEL_WIDTH / MODEL_HEIGHT  # 1.5


@dataclass(frozen=True)
class Crop:
    """A centre crop rectangle in source pixel coordinates."""

    x: int
    y: int
    width: int
    height: int
    mode: str  # "trim_height" | "trim_width" | "none"

    @property
    def aspect(self) -> float:
        return self.width / self.height


def crop_to_target(width: int, height: int, target: float = TARGET_ASPECT) -> Crop:
    """Return the centred crop of (width, height) whose aspect ratio is `target`."""
    if width <= 0 or height <= 0:
        raise ValueError(f"frame must have positive size, got {width}x{height}")
    aspect = width / height
    if aspect == target:
        return Crop(0, 0, width, height, "none")
    if aspect < target:
        # Enough width relative to height: shorten the height.
        new_height = int(round(width / target))
        new_height = max(1, min(new_height, height))
        return Crop(0, (height - new_height) // 2, width, new_height, "trim_height")
    # Wider than 3:2: narrowing the width is the only physically valid option.
    new_width = int(round(height * target))
    new_width = max(1, min(new_width, width))
    return Crop((width - new_width) // 2, 0, new_width, height, "trim_width")


def encode(image, crop: Crop) -> bytes:
    """Crop `image` (PIL RGB) and resize it to the model's 3 x 256 x 384 u8 stream.

    The HWC -> CHW transpose goes through numpy because `Image.Transpose.CHW`
    was removed in newer Pillow releases; the byte layout is identical.
    """
    import numpy as np
    from PIL import Image  # local import so the geometry helpers stay dependency-free

    if crop.mode != "none":
        image = image.crop((crop.x, crop.y, crop.x + crop.width, crop.y + crop.height))
    if image.size != (MODEL_WIDTH, MODEL_HEIGHT):
        image = image.resize((MODEL_WIDTH, MODEL_HEIGHT), Image.Resampling.BILINEAR)
    if image.mode != "RGB":
        image = image.convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    return np.ascontiguousarray(array.transpose(2, 0, 1)).tobytes()


def model_to_source(cx: float, cy: float, w: float, h: float, crop: Crop):
    """Map a model-space box (384 x 256 pixels) back onto the source frame."""
    sx = crop.width / MODEL_WIDTH
    sy = crop.height / MODEL_HEIGHT
    return (cx * sx + crop.x, cy * sy + crop.y, w * sx, h * sy)


def describe_crop(width: int, height: int, crop: Crop) -> str:
    names = {"trim_height": "削高度 (trim height)", "trim_width": "削寬度 (trim width)",
             "none": "無需裁剪 (already 3:2)", "raw_nnin": "機身 NNIN 原圖 (no crop)"}
    return (
        f"{width}x{height} ({width / height:.3f}) -> {names[crop.mode]} -> "
        f"{crop.width}x{crop.height} @{crop.x},{crop.y} -> resize {MODEL_WIDTH}x{MODEL_HEIGHT}"
    )
