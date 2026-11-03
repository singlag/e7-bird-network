"""Crop geometry tests. Run directly:  python app/tests/test_crop.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import TARGET_ASPECT, crop_to_target, encode, model_to_source  # noqa: E402


def check(width, height, mode, expected_size):
    crop = crop_to_target(width, height)
    assert crop.mode == mode, f"{width}x{height}: mode {crop.mode} != {mode}"
    assert (crop.width, crop.height) == expected_size, f"{width}x{height}: {crop.width}x{crop.height} != {expected_size}"
    assert abs(crop.aspect - TARGET_ASPECT) < 0.02, f"{width}x{height}: aspect {crop.aspect}"
    print(f"  ok {width}x{height} -> {crop.mode} {crop.width}x{crop.height} @({crop.x},{crop.y})")


def main():
    print("crop_to_target:")
    check(1200, 900, "trim_height", (1200, 800))   # 4:3 -> trim height
    check(900, 1200, "trim_height", (900, 600))    # 3:4 portrait -> trim height
    check(1920, 1080, "trim_width", (1620, 1080))  # 16:9 -> impossible to trim height, trim width
    check(1000, 1000, "trim_height", (1000, 667))  # 1:1 -> trim height
    check(384, 256, "none", (384, 256))            # already 3:2
    check(768, 512, "none", (768, 512))            # already 3:2
    check(640, 480, "trim_height", (640, 427))     # 4:3 VGA
    check(256, 384, "trim_height", (256, 171))     # 2:3 portrait-ish
    check(4000, 3000, "trim_height", (4000, 2667))

    from PIL import Image

    print("encode:")
    image = Image.new("RGB", (1200, 900), (10, 20, 30))
    crop = crop_to_target(1200, 900)
    encoded = encode(image, crop)
    assert len(encoded) == 294912, f"encoded length {len(encoded)}"
    plane = 256 * 384
    assert encoded[0] == 10 and encoded[plane] == 20 and encoded[plane * 2] == 30, "CHW plane order"
    assert set(encoded[:plane]) == {10} and set(encoded[plane : plane * 2]) == {20}, "plane contents"
    print("  ok 1200x900 -> 294912 bytes, planar R/G/B planes at 0 / 98304 / 196608")

    print("model_to_source round trip:")
    # A box at the model-space centre maps to the crop centre in source pixels.
    x, y, w, h = model_to_source(192.0, 128.0, 48.0, 32.0, crop)
    assert abs(x - (crop.x + crop.width / 2)) < 1e-6, x
    assert abs(y - (crop.y + crop.height / 2)) < 1e-6, y
    scale = crop.width / 384.0
    assert abs(w - 48.0 * scale) < 1e-6 and abs(h - 32.0 * (crop.height / 256.0)) < 1e-6
    print(f"  ok model centre (192,128) -> source ({x:.1f},{y:.1f}), size ({w:.1f},{h:.1f})")

    # A box at the model-space origin maps to the crop rectangle's top-left corner.
    x0, y0, _, _ = model_to_source(0.0, 0.0, 1.0, 1.0, crop)
    assert (x0, y0) == (crop.x, crop.y)
    print("  ok model origin -> crop top-left")

    print("ALL CROP TESTS PASSED")


if __name__ == "__main__":
    main()
