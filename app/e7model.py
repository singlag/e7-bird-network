"""NET5 / E7 bird model adapter.

The model itself is the untouched upstream package (`network.py` + `core.py` +
body + `commands.json`): original weights, original command order,
`weights_modified: false`.  The body is the Z8-extracted BIRD_body.bin placed
at the repo root as NET5_body.bin (byte-identical to the shipped
resource744.body.bin — same SHA-256; core.py falls back to the original when
NET5_body.bin is absent).  This adapter only wraps it for the GUI:
load once, feed 3:2 frames, return heads and decoded candidate boxes.
"""

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipeline  # noqa: E402  (app-local module)
from consumer import pack_heads  # noqa: E402  (upstream packer, unchanged)
from native_decoder import load_decoder  # noqa: E402


def consumer_table(root: Path = REPO_ROOT) -> bytes:
    """Assemble the 0x80000-byte consumer table from the manifest, verifying hashes."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    table = bytearray(0x80000)
    for row in manifest["consumer_tables"]:
        raw = (root / row["file"]).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"{row['file']} sha256 mismatch: expected {row['sha256']}, got {digest}")
        table[row["offset"] : row["offset"] + len(raw)] = raw
    return bytes(table)


class E7BirdModel:
    key = "z8-net5"
    label = "NET5 (BIRD)"
    decodable = True           # published decode contract: consumer table + ranks
    runnable = True
    note = (
        "Nikon E7 NET5 (BIRD) bird detector, original weights and command order. "
        "Body = Z8 BIRD_body.bin at the repo root as NET5_body.bin (same bytes as "
        "the shipped resource744.body.bin, SHA-256 verified). "
        "Input: u8 planar[3,256,384] (3:2). Output: fine[22,16,24] + coarse[33,8,12] "
        "half bit patterns, decoded to candidate boxes."
    )
    width = pipeline.MODEL_WIDTH
    height = pipeline.MODEL_HEIGHT
    crop_note = "非 3:2 輸入會先 center crop 去 3:2（優先削高度），再 bilinear resize 去 384x256"

    def __init__(self, threads: int = 4, decoder_mode: str = "auto"):
        self.threads = threads
        self.decoder_mode = decoder_mode
        self.network: object = None
        self.decoder: object = None
        self._ipu: bytes | None = None
        self._score_table: object = None

    def load(self, log=lambda message: None) -> None:
        if self.network is not None:
            return
        started = time.monotonic()
        log("載入模型（torch + NET5_body.bin）…")
        import torch  # imported here so the GUI can report a clear missing-dependency error

        torch.set_num_threads(self.threads)
        from network import E7BirdNetwork

        self.network = E7BirdNetwork(self.threads)
        self._ipu = consumer_table()
        self.decoder = load_decoder(REPO_ROOT, self._ipu, self.decoder_mode)
        log(f"模型載入完成（{time.monotonic() - started:.1f} s）· decoder={self.decoder.backend}")

    def encode(self, image) -> tuple:
        """Return (u8 CHW bytes, crop used)."""
        crop = pipeline.crop_to_target(image.width, image.height)
        return pipeline.encode(image, crop), crop

    def infer(self, encoded: bytes, include_workspace: bool = False) -> dict:
        started = time.monotonic()
        result = self.network.infer_planar(encoded, include_workspace=include_workspace)
        heads = result["heads"]
        buffer = pack_heads(heads["fine"].view("<f2"), heads["coarse"].view("<f2"))
        out = {
            "heads": heads,
            "head_buffer": buffer,
            "input_sha256": result["input_sha256"],
            "seconds": time.monotonic() - started,
        }
        if include_workspace:
            out["workspace"] = result["workspace"]
        return out

    def decode(self, head_buffer: bytes) -> dict:
        return self.decoder.consume(head_buffer)

    def score_table(self):
        """Consumer-table view used to recover token / probability / rank per row.

        Independent of which decoder backend ran: the arithmetic it exposes is the
        reference arithmetic both backends implement.
        """
        from decoder_port import PythonDecoder

        if self._score_table is None:
            self._score_table = PythonDecoder(self._ipu or consumer_table())
        return self._score_table

    def status(self) -> dict:
        import torch

        return {
            "model": self.label,
            "torch": torch.__version__,
            "threads": self.threads,
            "decoder": self.decoder.backend if self.decoder else "not loaded",
            "input": f"{self.width}x{self.height} u8 planar[3,256,384]",
            "repo": str(REPO_ROOT),
        }
