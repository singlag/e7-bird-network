"""Model discovery is by content hash — file names are decoration.

Run: pytest app/tests/test_model_files.py -q
"""

import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))
sys.path.insert(1, str(APP.parent))

import model_files  # noqa: E402
import model_registry  # noqa: E402

BODIES = Path.home() / "Documents/Z8/analysis/nn_models"

# a small, known body keeps the test quick but is hashed like any other
CANDIDATE = BODIES / "NET3_body.bin"


def test_identify_ignores_file_name(tmp_path):
    if not CANDIDATE.is_file():
        print(f"skip: {CANDIDATE} not present"); return
    alias = tmp_path / "i-called-this-anything.bin"
    alias.write_bytes(CANDIDATE.read_bytes())
    meta = model_files.identify(alias)
    assert meta is not None and meta["key"] == "z8-net3", meta
    assert meta["filename"] == "i-called-this-anything.bin"


def test_discover_lists_files_and_flags_unknown(tmp_path):
    if not CANDIDATE.is_file():
        print("skip: no Z8 body present"); return
    (tmp_path / "whatever-copy.bin").write_bytes((BODIES / "NET7_body.bin").read_bytes())
    (tmp_path / "NET7_body.bin").write_bytes((BODIES / "NET7_body.bin").read_bytes())
    (tmp_path / "junk.bin").write_bytes(b"\x00" * 4096)
    found = model_files.discover(tmp_path)
    by_key = {m["key"]: m for m in found}
    assert by_key["z8-net7"]["filename"] == "NET7_body.bin"
    assert by_key["z8-net7"]["extra_files"] == ["whatever-copy.bin"], \
        "same content collapses to one row, extra names reported"
    assert by_key["unknown:junk.bin"]["unidentified"]
    assert not by_key["unknown:junk.bin"]["runnable"]


def test_registry_labels_show_file_names():
    labels = dict(model_registry.labels())
    assert any("NET5_body.bin" in v for k, v in labels.items()
               if k == model_registry.default_key()), \
        "built-in NET5 row must name NET5_body.bin"
    builtin = model_registry.find(model_registry.default_key())
    assert builtin.runnable and builtin.decodable, "NET5 is the full model"
    for key, label in labels.items():
        if key.startswith("z8-net") and key != model_registry.default_key():
            assert " — " in label and label.split(" — ", 1)[1].endswith(".bin"), label
            assert model_registry.find(key).decodable is False


if __name__ == "__main__":
    import tempfile
    for fn in (test_identify_ignores_file_name, test_discover_lists_files_and_flags_unknown):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    test_registry_labels_show_file_names()
    print("MODEL DISCOVERY TESTS PASSED")
