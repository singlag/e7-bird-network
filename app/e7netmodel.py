"""A Z8 NN body loaded from `models/` by content hash.

This adapter proves the selected file loads: the emulator parses and disassembles
the body (model_bin full-body coverage through core.PhysicalStream.from_body).
None of these networks has a published decode contract (consumer table, rank
anchors), so they cannot produce detection boxes — running them for boxes is
disabled in the GUI with the reason stated.  Network replay through the
emulator itself is exercised by `run_bodies.py` / the GT analysis scripts.
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
            f"{self.role}. Loading parses/disassembles the body; box output is "
            "disabled because this network has no published decode contract."
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
