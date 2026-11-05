"""Model file discovery by content hash — file names carry no meaning here.

Model bodies live in `<repo>/models/`.  A file is identified by its SHA-256
against the verified Z8 body table (`references/model_backends_and_z8_models.md`
inventory; every hash measured in-session, never trusted from a sidecar).
The on-disk file name is irrelevant to identity — any name works — but it is
surfaced in the model list so the user can see which file a row loads.

Distribution: the firmware blobs are not committed to this repo; the user
copies them into `models/` from the Z8 extraction (~/Documents/Z8/...).
"""

import hashlib
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

# sha256 -> identity.  `runnable` = the body parses and replays through the
# emulator as a network; `decodable` = a published decode contract (consumer
# table / rank anchors) exists so it can produce detection boxes.  role_hypothesis
# is unverified (Z8 model IDs do not appear in firmware; hypothesis-level).
Z8_SOURCE = "extracted from Z8 firmware eg2070_0311b0.bi (inside Z_8_0311.dec)"

KNOWN_BODIES: dict[str, dict] = {
    "e8d371e77efc3035d0c633c8c90aa4851f1675e5abcf7999edb23adc6439722b": {
        "key": "z8-net5", "name": "NET5 (BIRD)",
        "runnable": True, "decodable": True, "role": "bird detector, proven",
        "built_in": True, "blocks": 889, "input": "384x256x3 u8",
        "head": "fine 22ch 16x24 + coarse 33ch 8x12 (engine-verified)",
        "source": "upstream package body (docs/model-provenance.json, shipped as "
                  "resource744.body.bin); at the repo root it is NET5_body.bin. "
                  "Same model as the Z8 extraction's BIRD_body.bin — "
                  "byte-identical, SHA-256 verified.",
        "data_from": "/home/garykan/Documents/Z8/analysis/nn_models/BIRD_body.bin",
    },
    "1c3e9753aaf23635b033ad15be3505ae1fdc9477e53363eb302e7954bc2fbbad": {
        "key": "z8-net1", "name": "NET1", "runnable": False, "decodable": False,
        "role": "hypothesis only; parsing fails on unresolved 0x3c8f1/0x3c921/0x3c933",
        "blocks": 889, "input": "384x256x3", "head": "12ch 8x8 coarse-only (axis-inferred)",
        "source": Z8_SOURCE,
    },
    "8b0a5ca5cf5c9aba3b11362766cd778f38948eb1de948968a9bee205796442aa": {
        "key": "z8-net2", "name": "NET2", "runnable": False, "decodable": False,
        "role": "hypothesis only; 32x32 input, parsing fails on unresolved ops",
        "blocks": 135, "input": "32x32 crop", "head": "32ch 32x2 (axis-inferred)",
        "source": Z8_SOURCE,
    },
    "373cd288cb60130ad74694635a0917906d60e0326eb674fd41210aa154f1d586": {
        "key": "z8-net3", "name": "NET3", "runnable": False, "decodable": False,
        "role": "hypothesis only; parsing fails on unresolved ops",
        "blocks": 192, "input": "32x32 crop", "head": "12ch 1x1 global classifier",
        "source": Z8_SOURCE,
    },
    "2cd4c959add30076c872a36d4275ebc8e102c469b405a9e034476c0872fac4d0": {
        "key": "z8-net4", "name": "NET4", "runnable": True, "decodable": False,
        "role": "bird hypothesis (strongest: 53% coverage, head layout matches)",
        "blocks": 709, "input": "384x256x3", "head": "24ch 16x30 (axis-inferred)",
        "source": Z8_SOURCE,
    },
    "67e8ff4a578200a9a94f67b27bf49a9e5f29497180d980d71812b72d2457dd34": {
        "key": "z8-net6", "name": "NET6", "runnable": True, "decodable": False,
        "role": "subject detector (hypothesis-level)",
        "blocks": 691, "input": "384x256x3", "head": "24ch 16x26 (axis-inferred)",
        "source": Z8_SOURCE,
    },
    "66a34f2456804699256b3c9d79da3bb241e2d5b617877b18948c37bdd898d32d": {
        "key": "z8-net7", "name": "NET7", "runnable": True, "decodable": False,
        "role": "scene classifier (hypothesis-level)",
        "blocks": 889, "input": "384x256x3", "head": "24ch 16x26 (axis-inferred)",
        "source": Z8_SOURCE,
    },
    "ba0574da3aa0e79f72e2e2d1c38f6e6ae80bd1c1ed27061acfc5dd44585407cc": {
        "key": "z8-net8", "name": "NET8", "runnable": False, "decodable": False,
        "role": "global classifier; parsing fails on unresolved ops",
        "blocks": 786, "input": "384x256x3", "head": "7ch 1x1 softmax global classifier",
        "source": Z8_SOURCE,
    },
    "fd35487517b86de748fd01b581b37a72bb33faefd0b1ac740a965eb70952ba2e": {
        "key": "z8-net9", "name": "NET9", "runnable": True, "decodable": False,
        "role": "hypothesis-level; pooling handler unverified, and GT "
                "enrichment scored it null-like (obs 6.61 < null mean 17.60)",
        "blocks": 1044, "input": "384x256x3",
        "head": "24ch 16x24 (axis-inferred); unique 2x2 pooling record family",
        "source": Z8_SOURCE,
    },
}


def table_for_key(key: str) -> tuple[str, dict] | tuple[None, None]:
    """(sha256, table entry) for a model key — the built-in entry included."""
    for sha, entry in KNOWN_BODIES.items():
        if entry["key"] == key:
            return sha, dict(entry)
    return None, None

# The built-in body sits at the repo root as NET5_body.bin (the Z8
# BIRD_body.bin, same bytes — SHA-256 verified).  It is not committed: it is a
# duplicate of the manifest-pinned resource744.body.bin, so a fresh checkout
# re-creates it from there under hash verification.  A matching file under
# models/ is a duplicate, not a second entry.
BUILT_IN_BODY = REPO_ROOT / "NET5_body.bin"
BUILT_IN_ORIGINAL = REPO_ROOT / "resource744.body.bin"


def ensure_builtin_body() -> Path:
    """Return the root NET5_body.bin, re-creating it from the original body."""
    if BUILT_IN_BODY.is_file():
        if sha256_of(BUILT_IN_BODY) in KNOWN_BODIES:
            return BUILT_IN_BODY
        BUILT_IN_BODY.unlink()
    raw = BUILT_IN_ORIGINAL.read_bytes()
    if sha256_bytes(raw) not in KNOWN_BODIES:
        raise ValueError("resource744.body.bin does not hash to the verified "
                         "NET5/BIRD body — refusing to create NET5_body.bin")
    BUILT_IN_BODY.write_bytes(raw)
    return BUILT_IN_BODY


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identify(path: Path) -> Optional[dict]:
    """Identity for a body file, or None if it is not a known Z8 body."""
    digest = sha256_of(path)
    meta = KNOWN_BODIES.get(digest)
    if meta is None:
        return None
    return {**meta, "path": path, "filename": path.name, "sha256": digest}


def discover(models_dir: Path = MODELS_DIR) -> list[dict]:
    """Scan `models_dir` for .bin bodies; one entry per distinct body.

    Files with a known hash collapse onto their identity (duplicates are
    reported under `extra_files`); unknown files are still listed so the user
    can see what is sitting in the folder, flagged as unidentified.
    """
    known: dict[str, dict] = {}
    unknown: list[dict] = []
    if not models_dir.is_dir():
        return []
    for path in sorted(models_dir.glob("*.bin")):
        meta = identify(path)
        if meta is None:
            unknown.append({"key": f"unknown:{path.name}", "name": path.name,
                            "runnable": False, "decodable": False,
                            "role": "unidentified file — its SHA-256 is not in "
                                    "the verified Z8 body table",
                            "path": path, "filename": path.name,
                            "sha256": sha256_of(path),
                            "unidentified": True})
            continue
        if meta.get("built_in"):
            entry = known.get(meta["key"])
            if entry is not None:
                entry["extra_files"].append(path.name)
            continue
        entry = known.get(meta["key"])
        if entry is None:
            meta["extra_files"] = []
            known[meta["key"]] = meta
        else:
            entry["extra_files"].append(path.name)
    order = ["z8-net1", "z8-net2", "z8-net3", "z8-net4", "z8-net6",
             "z8-net7", "z8-net8", "z8-net9"]
    out = [known[k] for k in order if k in known]
    out.extend(unknown)
    return out
