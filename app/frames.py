"""Input sources: still photos and sampled video frames.

Video sampling (user-specified):
  * take one frame per `per_second` seconds -> stride `N = round(fps * per_second)`
  * process at most `min(60, round(fps))` frames by default; the UI exposes the
    value so a longer clip can be raised manually
"""

import math
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FPS = 25.0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".mpg", ".mpeg", ".3gp"}


@dataclass
class FrameJob:
    index: int  # original frame index in the video (0 for a still image)
    seconds: float
    image: object = None  # PIL image, filled on demand


@dataclass
class SourceInfo:
    kind: str  # "image" | "video"
    path: str
    width: int
    height: int
    fps: float | None = None
    duration: float | None = None
    total_frames: int | None = None
    sampled: int = 0
    stride: int = 1
    max_frames: int | None = None
    truncated: bool = False
    note: str = ""


def classify(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(f"unsupported file type: {suffix or path.name}")


def default_max_frames(fps: float) -> int:
    """Cap = min(60, fps): 60 frames normally, or the clip's fps when that is lower."""
    return max(1, min(60, int(round(fps))))


def probe(path: Path | str) -> SourceInfo:
    path = Path(path)
    kind = classify(path)
    if kind == "image":
        from PIL import Image

        with Image.open(path) as image:
            return SourceInfo("image", str(path), image.width, image.height, sampled=1)
    import imageio_ffmpeg

    meta = _ffmpeg_meta(path)
    fps = meta.get("fps") or 25.0
    duration = meta.get("duration") or 0.0
    return SourceInfo(
        "video",
        str(path),
        int(meta["size"][0]),
        int(meta["size"][1]),
        fps=float(fps),
        duration=float(duration),
        total_frames=int(round(fps * duration)) if duration else None,
    )


def _ffmpeg_meta(path: Path) -> dict:
    """Header metadata from the bundled ffmpeg (fps, duration, size), no decoding."""
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    meta = next(reader)
    reader.close()
    size = meta.get("size")
    if not size:
        raise RuntimeError(f"ffmpeg reported no video size for {path}")
    return {"fps": meta.get("fps") or 25.0, "duration": meta.get("duration") or 0.0, "size": size}


def sampling_plan(info: SourceInfo, per_second: float, max_frames: int | None) -> tuple[int, int]:
    """Return (stride, frames_to_process)."""
    if info.kind == "image":
        return 1, 1
    fps = float(info.fps or DEFAULT_FPS)
    stride = max(1, int(round(fps * per_second)))
    cap = max_frames if max_frames is not None else default_max_frames(fps)
    available = math.ceil((info.duration or 0.0) / per_second) if info.duration else 0
    if available == 0:
        available = 1
    return stride, max(1, min(cap, available))


def _frame_from_stream(raw, width: int, height: int):
    """ffmpeg rawvideo rgb24 frames are row-major (height, width, 3).

    imageio_ffmpeg >= 0.8 yields raw bytes; older releases yielded numpy arrays
    laid out (width, height, 3), so both shapes are accepted.
    """
    import numpy as np
    from PIL import Image

    if isinstance(raw, (bytes, bytearray, memoryview)):
        array = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(height, width, 3)
    else:
        array = np.asarray(raw, dtype=np.uint8)
        if array.shape == (width, height, 3):
            array = array.transpose(1, 0, 2)
    return Image.fromarray(array, "RGB")


def _frame_to_stream(image):
    """PIL image -> contiguous row-major (height, width, 3) uint8 array."""
    import numpy as np

    return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))


def iter_frames(path: Path | str, info: SourceInfo, stride: int, max_frames: int):
    """Yield sampled frames as PIL images, stopping after `max_frames`."""
    from PIL import Image

    path = Path(path)

    if info.kind == "image":
        with Image.open(path) as image:
            yield 0, 0.0, image.convert("RGB")
        return
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    next(reader)  # metadata dict
    emitted = 0
    index = 0
    fps = float(info.fps or DEFAULT_FPS)
    for raw in reader:
        if index % stride == 0:
            yield index, index / fps, _frame_from_stream(raw, info.width, info.height)
            emitted += 1
            if emitted >= max_frames:
                break
        index += 1
    reader.close()


def write_annotated_video(path: Path | str, width: int, height: int, fps: float, frames):
    """Encode already-annotated PIL frames into `path`.

    `macro_block_size=1` stops imageio_ffmpeg silently resizing frames (which
    would desynchronise from the size we declared); libx264 still needs even
    dimensions, so odd sizes are stepped down by one pixel and frames resized
    to match.
    """
    import imageio_ffmpeg
    from PIL import Image

    path = Path(path)
    out_width = width - (width % 2)
    out_height = height - (height % 2)
    writer = imageio_ffmpeg.write_frames(
        str(path), (out_width, out_height), fps=fps, codec="libx264",
        pix_fmt_in="rgb24", macro_block_size=1,
    )
    writer.send(None)
    for image in frames:
        if image.size != (out_width, out_height):
            image = image.resize((out_width, out_height), Image.Resampling.BILINEAR)
        writer.send(_frame_to_stream(image))
    writer.close()
