#!/usr/bin/env python3
"""End-to-end regression: captured fixture input -> network heads -> wide decoder.

Chains the two halves that the upstream package verifies separately:

  fixtures/<case>/input-chw-u8.bin
      -> E7BirdNetwork heads            (must equal fixtures/<case>/expected.npz)
      -> pack_heads buffer              (upstream packer, unchanged)
      -> wide decoder                   (Python port, and the native library too
                                         when one is available)

The native side is optional: set E7_DECODER_LIBRARY to a built
wide-reference library (see build_decoder.sh / build_decoder.ps1) and the port
output is compared against it record by record.  Without it the test still
asserts the head bit-exactness and that the port produces a sane decode.

Run:  python app/tests/test_e2e_fixture.py [--case random] [--threads 4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parent
REPO_ROOT = APP_DIR.parent
for path in (APP_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402

from decoder_port import PythonDecoder  # noqa: E402
from native_decoder import candidate_library, load_decoder  # noqa: E402


def decode_with_port(buffer: bytes, ipu: bytes) -> dict:
    return PythonDecoder(ipu).consume(buffer)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="all", help="fixture name, or 'all'")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    index = json.loads((REPO_ROOT / "fixtures" / "index.json").read_text())
    cases = [c for c in index["cases"] if args.case == "all" or c["name"] == args.case]
    if not cases:
        print(f"no fixture case named {args.case!r}")
        return 1

    from e7model import E7BirdModel, consumer_table  # imported late: torch load is slow

    ipu = consumer_table(REPO_ROOT)
    model = E7BirdModel(threads=args.threads, decoder_mode="port")
    model.load(log=lambda message: print(f"  {message}"))

    library = candidate_library(REPO_ROOT)
    if library:
        native = load_decoder(REPO_ROOT, ipu, mode="native")
        print(f"native decoder = {Path(library).name}")
    else:
        native = None
        print("native decoder = none (set E7_DECODER_LIBRARY to compare)")

    failures = 0
    for case in cases:
        directory = REPO_ROOT / "fixtures" / case["name"]
        raw = (directory / "input-chw-u8.bin").read_bytes()
        result = model.infer(raw)
        heads = result["heads"]
        expected = np.load(directory / "expected.npz")

        head_mismatch = {
            key: int((heads[key] != expected[key]).sum()) for key in ("fine", "coarse")
        }
        for key, count in head_mismatch.items():
            if count:
                failures += 1
                print(f"FAIL {case['name']}: head {key} differs in {count} values")

        decoded = decode_with_port(result["head_buffer"], ipu)
        want_records = case.get("expected_records", [])
        print(
            f"{case['name']}: heads exact={all(v == 0 for v in head_mismatch.values())} "
            f"status={decoded['status']} visited={decoded['visited']} "
            f"accepted={decoded['accepted']} records={len(decoded['records'])} "
            f"(fixture expects {len(want_records)})"
        )
        if decoded["status"] != 0:
            failures += 1
            print(f"FAIL {case['name']}: decoder status {decoded['status']}")
        if decoded["records"] != want_records:
            failures += 1
            print(f"FAIL {case['name']}: decoded records != fixtures/index.json expected_records")
            print(f"  got   ={decoded['records']}")
            print(f"  wants ={want_records}")

        if native is not None:
            other = native.consume(result["head_buffer"])
            keys = ("status", "visited", "accepted", "outputs", "records", "initial", "ordinals")
            for key in keys:
                if other[key] != decoded[key]:
                    failures += 1
                    print(f"FAIL {case['name']}: port != native on {key}\n  port  ={decoded[key]}\n  native={other[key]}")

    if failures:
        print(f"E2E FAILURES: {failures}")
        return 1
    print(f"E2E OK: {len(cases)} fixture case(s) through network and wide decoder")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
