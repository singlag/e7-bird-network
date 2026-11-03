#!/usr/bin/env bash
# Build the native wide-reference decoder library (Linux / macOS).
#
# Windows normally does NOT need this: the package already ships
# `wide-reference.dll` (x64) at the repository root, and app/native_decoder.py
# loads it automatically.  Use this script when you want the native backend on
# Linux/macOS, or a locally built copy to compare against the Python port.
#
#   ./app/build_decoder.sh   ->  app/vendor/wide-reference.so  (.dylib on macOS)
#
# The upstream sources target Windows and are NOT edited: two host-only shims
# are applied to copies under .local/decoder-build/
#   1. drop `__declspec(dllexport)` — a Windows storage-class specifier that
#      gcc/clang on Linux and macOS do not accept
#   2. relax `_Static_assert(sizeof(long)==4, ...)` to `4 || 8`, because the
#      archived qsort swaps elements as `long` words only when
#      `sizeof(Record) % sizeof(long) == 0`.  A 20-byte Record on a 64-bit host
#      (long = 8) fails that test, so it falls back to byte swapping — the same
#      20 bytes move, so the sorted order, and hence the tie order for equal
#      scores, is unchanged.  app/tests/test_decoder_parity.py proves that
#      against this build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/.local/decoder-build"
OUT_DIR="$ROOT/app/vendor"
mkdir -p "$BUILD" "$OUT_DIR"

case "$(uname -s)" in
  Darwin) OUT="$OUT_DIR/wide-reference.dylib" ;;
  *)      OUT="$OUT_DIR/wide-reference.so" ;;
esac

for src in wide_reference.c qsort-reference.c; do
  sed -e 's/__declspec(dllexport)//g' \
      -e 's#_Static_assert(sizeof(long)==4, #_Static_assert(sizeof(long)==4||sizeof(long)==8, #' \
      "$ROOT/$src" > "$BUILD/$src"
done

CC="${CC:-cc}"
"$CC" -O2 -std=c11 -fPIC -shared -o "$OUT" "$BUILD/wide_reference.c" "$BUILD/qsort-reference.c" -lm

echo "built: $OUT"
echo "parity check against the Python port:"
echo "  E7_DECODER_LIBRARY=$OUT python app/tests/test_decoder_parity.py"
