"""A Z8 NN body loaded from `models/` by content hash.

This adapter loads *and runs* the selected file: the emulator parses and
disassembles the body (model_bin full-body coverage through
core.PhysicalStream.from_body) and `infer()` replays frames through it.
None of these networks has a published decode contract (consumer table, rank
anchors), so they cannot produce decoder boxes — the job runs inference, saves
heads (`.npz`) and draws one research-only saliency frame per frame from the
studied objectness planes (see `saliency.py`); the GUI states this in the
model note.
"""

import sys
import time
from pathlib import Path

import model_files

if str(model_files.REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(model_files.REPO_ROOT))


class Z8NetModel:
    """Load-and-verify adapter; identity comes from the hash, not the name."""

    decodable = False

    def __init__(self, path: Path):
        meta = model_files.identify(path)
        if meta is None:
            raise ValueError(f"unidentified model body: {path}")
        self.key = meta["key"]
        self.name = meta["name"]
        self.role = meta["role"]
        self.runnable = meta["runnable"]
        self.path = path
        self.label = f"{self.name} — {path.name}"
        self.note = (
            f"Z8 body, identified by SHA-256 (file name is not used). Role: "
            f"{self.role}. Loading parses/disassembles the body. This network "
            "has no published decode contract, so the decoder never runs; the "
            "preview instead shows one research-only saliency frame built from "
            "the external-sample objectness-plane study (class column: SAL)."
        )
        self.stream = None

    def load(self, log=lambda message: None) -> None:
        if self.stream is not None:
            return
        started = time.monotonic()
        log(f"載入 {self.name}（{self.path.name}）…")
        from core import PhysicalStream  # heavy import kept local to the load

        self.stream = PhysicalStream.from_body(str(self.path))
        audit = self.stream.audit
        log(
            f"{self.name} 載入完成（{time.monotonic() - started:.1f} s）· "
            f"{audit.get('matrices', '?')} matrices · body verified by hash"
        )

    def status(self) -> dict:
        return {
            "model": self.name,
            "file": str(self.path),
            "loaded": self.stream is not None,
            "decodable": False,
            "role": self.role,
        }

    def infer(self, encoded: bytes, include_workspace: bool = False) -> dict:
        """Replay one u8 CHW frame through the parsed body.

        Same input conversion as `run_bodies.py` (u8 planar / 256 → f32).  The
        heads are returned as numpy arrays so `--save-heads` can write them;
        there is no decode contract, so no head buffer for a box decoder.
        """
        import hashlib
        import numpy as np
        import torch

        if self.stream is None:
            self.load()
        started = time.monotonic()
        pixels = torch.from_numpy(
            np.frombuffer(encoded, np.uint8).reshape(1, 3, 256, 384).astype(np.float32) / 256)
        with torch.inference_mode():
            outputs = self.stream(pixels)
        heads = {f"head{index}": tensor.detach().numpy()
                 for index, tensor in enumerate(outputs)}
        return {
            "heads": heads,
            "head_buffer": None,        # no published decode contract
            "input_sha256": hashlib.sha256(encoded).hexdigest(),
            "seconds": time.monotonic() - started,
        }
