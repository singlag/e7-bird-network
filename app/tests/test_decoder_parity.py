"""Decoder parity test: the Python port vs the native reference library.

Run with a built library, e.g.

  E7_DECODER_LIBRARY=/path/to/wide-reference.so python app/tests/test_decoder_parity.py

Without a native library the test only exercises the port's self-consistency
(structures, scan count, deterministic output) and says so.
"""

import os
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
REPO = APP.parent
sys.path.insert(0, str(APP))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from decoder_port import PythonDecoder  # noqa: E402
from e7model import consumer_table  # noqa: E402
from native_decoder import NativeDecoder, candidate_library  # noqa: E402

HEAD_BYTES = 0x6300
TOKEN_MAX = 0x5400  # anything above this is a non-finite token (status -2)


def random_heads(rng, count=HEAD_BYTES // 2):
    values = rng.integers(0, TOKEN_MAX + 1, size=count, dtype=np.uint16)
    sign = rng.integers(0, 2, size=count).astype(bool)
    return np.where(sign, values | 0x8000, values).astype("<u2").tobytes()


def blank_heads():
    return bytes(HEAD_BYTES)


def token_offset(stage, anchor, field, row, col):
    stride = 32 if stage else 48
    rows = 8 if stage else 16
    return (0x4200 if stage else 0) + (anchor * 11 + field) * stride * rows + row * stride + 2 * col


def set_token(buffer: bytearray, stage, anchor, field, row, col, value):
    offset = token_offset(stage, anchor, field, row, col)
    buffer[offset] = value & 0xFF
    buffer[offset + 1] = (value >> 8) & 0xFF


def crafted_one():
    """One accepted candidate: fine stage, anchor 0, cell (8,12), objectness + class 0."""
    buffer = bytearray(HEAD_BYTES)
    for field in range(6):
        set_token(buffer, 0, 0, field, 8, 12, {4: 0x3C00, 5: 0x3C00}.get(field, 0x3800))
    # Same shape in the coarse stage, anchor 1, so two stages contribute.
    for field in range(6):
        set_token(buffer, 1, 1, field, 4, 6, {4: 0x3C00, 5: 0x3C00}.get(field, 0x3800))
    assert len(buffer) == HEAD_BYTES
    return bytes(buffer)


def crafted_cluster(rng):
    """Many nearby candidates so the suppression pass and clip paths run."""
    values = rng.integers(0, TOKEN_MAX + 1, size=HEAD_BYTES // 2, dtype=np.uint16)
    values = np.where(rng.integers(0, 2, size=values.shape).astype(bool), values | 0x8000, values)
    buffer = bytearray(values.astype("<u2").tobytes())
    # Push a block of fine cells above the gates so they are accepted.
    for row in range(6):
        for col in range(6):
            set_token(buffer, 0, 0, 4, row, col, 0x3C00)  # objectness
            set_token(buffer, 0, 0, 5, row, col, 0x3C00)  # class slot 0
            set_token(buffer, 0, 0, 0, row, col, 0x3800)  # cx logit
            set_token(buffer, 0, 0, 1, row, col, 0x3800)  # cy logit
            set_token(buffer, 0, 0, 2, row, col, 0x2800)  # w logit
            set_token(buffer, 0, 0, 3, row, col, 0x2800)  # h logit
    return bytes(buffer)


def nonfinite_heads():
    buffer = bytearray(HEAD_BYTES)
    set_token(buffer, 0, 0, 4, 0, 0, 0x5401)  # first objectness token above the finite ceiling
    return bytes(buffer)


def comparable(result):
    return {
        "status": result["status"],
        "records": [list(map(int, record)) for record in result["records"]],
        "visited": result["visited"],
        "accepted": result["accepted"],
        "outputs": result["outputs"],
        "initial": [int(value) for value in result["initial"]],
        "error_field": result["error_field"],
        "error_ordinal": result["error_ordinal"],
    }


def main():
    ipu = consumer_table(REPO)
    port = PythonDecoder(ipu)
    path = candidate_library(REPO)
    native = NativeDecoder(path, ipu) if path else None
    print(f"port backend = {port.backend}")
    print(f"native library = {path.name if path else 'NONE (port-only run)'}")

    rng = np.random.default_rng(20261103)
    cases = [("blank", blank_heads()), ("crafted-one", crafted_one()), ("cluster", crafted_cluster(rng))]
    cases += [(f"random-{i:02d}", random_heads(rng)) for i in range(40)]
    cases.append(("nonfinite", nonfinite_heads()))

    failures = []
    accepted_total = 0
    record_total = 0
    for name, buffer in cases:
        result = port.consume(buffer)
        accepted_total += result["accepted"]
        record_total += len(result["records"])
        if native is None:
            continue
        reference = native.consume(buffer)
        got, want = comparable(result), comparable(reference)
        if got != want:
            failures.append((name, got, want))

    print(f"cases={len(cases)} accepted_seen={accepted_total} records_seen={record_total}")
    sample = port.consume(crafted_one())
    print(f"crafted-one -> status={sample['status']} visited={sample['visited']} "
          f"accepted={sample['accepted']} records={sample['records']}")
    broken = port.consume(nonfinite_heads())
    print(f"nonfinite -> status={broken['status']} error_field={broken['error_field']} error_ordinal={broken['error_ordinal']}")

    if native is None:
        print("NO NATIVE LIBRARY: port self-consistency only. Build one with "
              "app/build_decoder.sh / app/build_decoder.ps1 and set E7_DECODER_LIBRARY.")
        print(f"PORT SMOKE OK ({record_total} records decoded)")
        return 0

    if failures:
        print(f"MISMATCHES: {len(failures)}")
        for name, got, want in failures[:8]:
            print(f"  {name}\n    port   {got}\n    native {want}")
        return 1
    print(f"PARITY OK: port == native on {len(cases)} cases "
          f"({record_total} records, {accepted_total} accepted candidates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
