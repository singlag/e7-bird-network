"""Native candidate-decoder loader (Windows DLL / Linux .so / macOS .dylib).

Library search order:
  1. `E7_DECODER_LIBRARY` environment variable (absolute path to a built library)
  2. `app/vendor/wide-reference.<ext>` for the running platform, produced by
     `app/build_decoder.ps1` (Windows) or `app/build_decoder.sh` (Linux/macOS)
  3. the bundled `wide-reference.dll` at the repository root, Windows only

If nothing loadable is found, callers fall back to `decoder_port.PythonDecoder`.
The ABI mirrors `consumer.py`: `reference_consume(heads, len, ipu, len, Report*)`
plus `reference_mode()` returning 1 (round-to-nearest floating point).
"""

import ctypes
import os
import platform
import sys
from pathlib import Path

HEAD_BYTES = 0x6300
IPU_BYTES = 0x80000
MAX_OUTPUTS = 16

_SUFFIX = {"Windows": ".dll", "Darwin": ".dylib"}.get(platform.system(), ".so")


class Box(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("w", ctypes.c_float),
        ("h", ctypes.c_float),
        ("score", ctypes.c_uint16),
        ("cls", ctypes.c_uint16),
    ]


class Report(ctypes.Structure):
    _fields_ = [
        ("visited", ctypes.c_uint32),
        ("accepted", ctypes.c_uint32),
        ("outputs", ctypes.c_uint32),
        ("error_ordinal", ctypes.c_uint32),
        ("error_field", ctypes.c_uint32),
        ("initial", ctypes.c_uint16 * 24),
        ("ordinals", ctypes.c_uint32 * 32),
        ("candidates", Box * 32),
        # Declared flat (6 fields x 16 rows); same memory layout as `c_uint16 * 6 * 16`
        # but `report.result[row * 6 + field]` indexes a scalar instead of a sub-array.
        ("result", ctypes.c_uint16 * 96),
    ]


def candidate_library(repo_root: Path):
    override = os.environ.get("E7_DECODER_LIBRARY")
    if override:
        path = Path(override)
        if path.is_file():
            return path
    vendor = repo_root / "app" / "vendor" / f"wide-reference{_SUFFIX}"
    if vendor.is_file():
        return vendor
    bundled = repo_root / f"wide-reference{_SUFFIX}"
    if bundled.is_file() and platform.system() == "Windows":
        return bundled
    return None


class NativeDecoder:
    backend = "native"

    def __init__(self, library_path, ipu: bytes):
        if len(ipu) != IPU_BYTES:
            raise ValueError(f"consumer table must be {IPU_BYTES} bytes, got {len(ipu)}")
        self.path = Path(library_path)
        self.library = ctypes.CDLL(str(self.path))
        self.library.reference_mode.restype = ctypes.c_int
        if not self.library.reference_mode():
            raise RuntimeError(f"{self.path.name} requires round-to-nearest floating point")
        self.library.reference_consume.restype = ctypes.c_int
        self.library.reference_consume.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(Report),
        ]
        if ctypes.sizeof(Box) != 20 or ctypes.sizeof(Report) != 1028:
            raise RuntimeError("unexpected decoder struct layout")
        self._ipu = (ctypes.c_uint8 * IPU_BYTES).from_buffer_copy(ipu)

    def consume(self, heads: bytes) -> dict:
        if len(heads) != HEAD_BYTES:
            raise ValueError(f"head region must be {HEAD_BYTES} bytes, got {len(heads)}")
        raw = (ctypes.c_uint8 * HEAD_BYTES).from_buffer_copy(heads)
        report = Report()
        status = self.library.reference_consume(
            raw, HEAD_BYTES, self._ipu, IPU_BYTES, ctypes.byref(report)
        )
        outputs = int(report.outputs)
        accepted = int(report.accepted)
        return {
            "status": int(status),
            # ctypes scalar-array slices return bytes, so index element by element.
            "records": [[int(report.result[row * 6 + field]) for field in range(6)] for row in range(outputs)],
            "visited": int(report.visited),
            "accepted": accepted,
            "outputs": outputs,
            "error_field": int(report.error_field),
            "error_ordinal": int(report.error_ordinal),
            "initial": [int(report.initial[i]) for i in range(24)],
            "ordinals": [int(report.ordinals[i]) for i in range(accepted)],
            "candidates": [
                (float(report.candidates[i].x), float(report.candidates[i].y),
                 float(report.candidates[i].w), float(report.candidates[i].h),
                 int(report.candidates[i].score), int(report.candidates[i].cls))
                for i in range(accepted)
            ],
            "backend": f"native:{self.path.name}",
        }


def load_decoder(repo_root: Path, ipu: bytes, mode: str = "auto"):
    """Return a decoder with `.consume(heads)` and a `.backend` label."""
    from decoder_port import PythonDecoder

    if mode == "port":
        return PythonDecoder(ipu)
    path = candidate_library(repo_root)
    if path is None:
        if mode == "native":
            raise FileNotFoundError(
                "no native decoder library found; run app/build_decoder.ps1 (Windows) or "
                "app/build_decoder.sh (Linux/macOS), or choose decoder 'port'"
            )
        return PythonDecoder(ipu)
    try:
        return NativeDecoder(path, ipu)
    except OSError as error:  # wrong architecture / wrong platform build
        if mode == "native":
            raise
        print(f"[warn] native decoder {path} failed to load ({error}); using Python port", file=sys.stderr)
        return PythonDecoder(ipu)
