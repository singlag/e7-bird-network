"""Pure-Python port of `wide_reference.c`, the E7 candidate decoder.

The bundled `wide-reference.dll` is Windows x64, so this module reproduces the
archived consumer op-for-op on any platform.  It follows the C source line by
line: same head offsets, same binary32 round-to-nearest-even step at every
primitive (`add` / `mul` / `divide` each round once, no fused multiply-add),
same 1000-point rank curves, same suppression on the *copied* candidate array,
same integer geometry in `clip`.

Records are `[cx, cy, w, h, score, cls]` in 384 x 256 model pixels, with
`cx, cy` the box centre — exactly what the library's `result` rows contain.

Tie order is reproduced too: the sort is an index-by-index port of the
archived `qsort-reference.c` (BSD qsort, Bentley & McIlroy), so equal scores
end up in the same order as the reference library.
"""

import struct

import numpy as np

F = np.float32

HEAD_BYTES = 0x6300
IPU_BYTES = 0x80000
MAX_CANDIDATES = 32
MAX_OUTPUTS = 16

TOKEN_OFFSET = 0x4EE48
CURVE_OFFSET = 0x4EE78
OBJGATE_OFFSET = 0x66578
SCOREGATE_OFFSET = 0x6657C
FACTOR_OFFSET = 0x66580
FLOOR_OFFSET = 0x66584
ANCHOR_W_OFFSET = 0x665A2
ANCHOR_H_OFFSET = 0x665A4
CURVE_POINTS = 1000
CURVE_BYTES = 4 * CURVE_POINTS
COARSE_BASE = 0x4200
SCALED_LIMIT = 2147483648.0


def bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def float_to_bits(value) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _f32(bits: int) -> np.float32:
    return F(bits_to_float(bits))


def add(a, b) -> np.float32:
    return F(F(a) + F(b))


def mul(a, b) -> np.float32:
    return F(F(a) * F(b))


def divide(a, b) -> np.float32:
    return F(F(a) / F(b))


def half(value: int) -> np.float32:
    """`reference_half`: the stored uint16 is a sign / exponent / mantissa pattern."""
    exponent = (value >> 10) & 31
    mantissa = value & 1023
    significand = float(mantissa + 1024 if exponent else mantissa)
    scaled = F(np.ldexp(F(significand), exponent - 25 if exponent else -24))
    return F(-scaled) if value & 32768 else scaled


def polynomial(value) -> np.float32:
    """`reference_polynomial`: e**x via degree-4 polynomial plus exact 2**n scaling."""
    value = F(value)
    scaled = mul(value, _f32(0x3FB8AA3B))  # log2(e)
    ordinal = int(float(scaled))  # C (int32_t): truncation toward zero
    residual = add(value, -mul(F(ordinal), _f32(0x3F317218)))  # value - ordinal*ln2
    r2 = mul(residual, residual)
    r4 = mul(r2, r2)
    high = add(mul(residual, _f32(0x394D4A8B)), _f32(0x3AB91BCB))
    middle = add(mul(residual, _f32(0x3C0897A6)), _f32(0x3D2AA390))
    low = add(mul(residual, _f32(0x3E2AAA6A)), _f32(0x3F00000A))
    linear = add(mul(residual, _f32(0x3F800001)), F(1.0))
    body = add(
        mul(add(mul(high, r2), middle), r4),
        add(mul(low, r2), linear),
    )
    return F(bits_to_float((float_to_bits(body) + ((ordinal << 23) & 0xFFFFFFFF)) & 0xFFFFFFFF))


def sigmoid(value) -> np.float32:
    return divide(F(1.0), add(F(1.0), polynomial(-value)))


def read_head(heads: bytes, stage: int, anchor: int, field: int, row: int, col: int) -> int:
    """`reference_read_head`: anchor/field planes, pitch 48 B (fine) / 32 B (coarse)."""
    stride = 32 if stage else 48
    rows = 8 if stage else 16
    offset = (COARSE_BASE if stage else 0) + (anchor * 11 + field) * stride * rows + row * stride + 2 * col
    return heads[offset] | (heads[offset + 1] << 8)


def rank(curve, value: float, low: int) -> int:
    """`reference_rank`: binary search over one 1000-point calibration curve."""
    high = CURVE_POINTS - 1
    if low > high:
        return CURVE_POINTS
    while low <= high:
        middle = int((low + high) / 2)  # C integer division truncates toward zero
        item = curve[middle]
        if value > item:
            low = middle + 1
        elif item > value:
            high = middle - 1
        else:
            return middle
    return min(high + 1, CURVE_POINTS)


def _i32(bits: int) -> int:
    return struct.unpack("<i", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def plus32(a: int, b: int) -> int:
    return _i32((a & 0xFFFFFFFF) + (b & 0xFFFFFFFF))


def minus32(a: int, b: int) -> int:
    return _i32((a & 0xFFFFFFFF) - (b & 0xFFFFFFFF))


def coordinate(value, scale: int) -> int:
    """`reference_coordinate`: mul, add half, clamp to int32, truncate toward zero."""
    scaled = mul(F(value), F(float(scale)))
    biased = add(scaled, F(0.5 if scaled >= 0 else -0.5))
    if biased >= SCALED_LIMIT:
        return 2147483647
    if biased <= -SCALED_LIMIT:
        return -2147483648
    return int(biased)



def _by_score_desc(a, b) -> int:
    """The archived comparator: `by_score` in `qsort-reference.c`."""
    return int(b[4]) - int(a[4])


def _insertion_sort(items, start: int, n: int, cmp) -> None:
    """BSD insertion pass: `for (pm = a+es; ...)` in `qsort-reference.c`."""
    for pm in range(start + 1, start + n):
        pl = pm
        while pl > start and cmp(items[pl - 1], items[pl]) > 0:
            items[pl], items[pl - 1] = items[pl - 1], items[pl]
            pl -= 1


def _med3(items, a: int, b: int, c: int, cmp) -> int:
    if cmp(items[a], items[b]) < 0:
        if cmp(items[b], items[c]) < 0:
            return b
        return c if cmp(items[a], items[c]) < 0 else a
    if cmp(items[b], items[c]) > 0:
        return b
    return a if cmp(items[a], items[c]) < 0 else c


def archived_qsort(items, n: int, cmp=_by_score_desc, start: int = 0) -> None:
    """The archived BSD/Bentley-McIlroy qsort, ported index-by-index.

    `qsort-reference.c` walks the array with byte pointers; here every pointer
    becomes an element index (`ptr / es`), so the pivot choice, the equal-value
    gathering and the two `vecswap` moves happen in the same order.  Swapping by
    element instead of by `long`/`char` blocks only changes speed, never order,
    which is what decides the tie order for equal scores.
    """
    while True:
        swap_cnt = 0
        if n < 7:
            _insertion_sort(items, start, n, cmp)
            return
        pm = start + n // 2
        if n > 7:
            pl = start
            pn = start + n - 1
            if n > 40:
                d = n // 8
                pl = _med3(items, pl, pl + d, pl + 2 * d, cmp)
                pm = _med3(items, pm - d, pm, pm + d, cmp)
                pn = _med3(items, pn - 2 * d, pn - d, pn, cmp)
            pm = _med3(items, pl, pm, pn, cmp)
        items[start], items[pm] = items[pm], items[start]
        pa = pb = start + 1
        pc = pd = start + n - 1
        while True:
            while pb <= pc:
                r = cmp(items[pb], items[start])
                if r > 0:
                    break
                if r == 0:
                    swap_cnt = 1
                    items[pa], items[pb] = items[pb], items[pa]
                    pa += 1
                pb += 1
            while pb <= pc:
                r = cmp(items[pc], items[start])
                if r < 0:
                    break
                if r == 0:
                    swap_cnt = 1
                    items[pc], items[pd] = items[pd], items[pc]
                    pd -= 1
                pc -= 1
            if pb > pc:
                break
            items[pb], items[pc] = items[pc], items[pb]
            swap_cnt = 1
            pb += 1
            pc -= 1
        if swap_cnt == 0:
            _insertion_sort(items, start, n, cmp)
            return
        end = start + n
        r = min(pa - start, pb - pa)
        items[start : start + r], items[pb - r : pb] = items[pb - r : pb], items[start : start + r]
        r = min(pd - pc, end - pd - 1)
        items[pb : pb + r], items[end - r : end] = items[end - r : end], items[pb : pb + r]
        left = pb - pa
        if left > 1:
            archived_qsort(items, left, cmp, start)
        right = pd - pc
        if right > 1:
            # The C version iterates instead of recursing on the right partition.
            start = end - right
            n = right
            continue
        return


def clip(candidate) -> list | None:
    """`reference_clip`: centre/size in model space, rejecting degenerate boxes.

    Returns `[cx, cy, w, h, score, cls]` exactly like the library's `result` row.
    """
    cx, cy, w, h, score, cls = candidate
    x = coordinate(cx, 384)
    y = coordinate(cy, 256)
    w = coordinate(w, 384)
    h = coordinate(h, 256)
    left = minus32(x, int(w / 2))
    top = minus32(y, int(h / 2))
    right = plus32(left, w)
    bottom = plus32(top, h)
    if right > 383:
        w = minus32(w, minus32(right, 383))
        x = plus32(minus32(383, w), int(w / 2))
    if left < 0:
        w = plus32(w, left)
        x = int(w / 2)
    if bottom > 255:
        h = minus32(h, minus32(bottom, 255))
        y = plus32(minus32(255, h), int(h / 2))
    if top < 0:
        h = plus32(h, top)
        y = int(h / 2)
    if (
        (x & 0xFFFFFFFF) > 383
        or ((w - 1) & 0xFFFFFFFF) > 383
        or (y & 0xFFFFFFFF) > 255
        or ((h - 1) & 0xFFFFFFFF) > 255
        or score > 1000
    ):
        return None
    return [x, y, w, h, score, cls]


class PythonDecoder:
    """Cross-platform stand-in for `wide-reference.dll`."""

    backend = "python-port"

    def __init__(self, ipu: bytes):
        if len(ipu) != IPU_BYTES:
            raise ValueError(f"consumer table must be {IPU_BYTES} bytes, got {len(ipu)}")
        self.ipu = ipu
        self.tokens = [self._u16(TOKEN_OFFSET + 2 * k) for k in range(24)]
        self.curves = [
            list(struct.unpack("<1000f", ipu[CURVE_OFFSET + CURVE_BYTES * k : CURVE_OFFSET + CURVE_BYTES * (k + 1)]))
            for k in range(24)
        ]
        self.objectness_gate = bits_to_float(self._u32(OBJGATE_OFFSET))
        self.score_gate = bits_to_float(self._u32(SCOREGATE_OFFSET))
        self.overlap_factor = bits_to_float(self._u32(FACTOR_OFFSET))

    def _u16(self, offset: int) -> int:
        return struct.unpack_from("<H", self.ipu, offset)[0]

    def _u32(self, offset: int) -> int:
        return struct.unpack_from("<I", self.ipu, offset)[0]

    def initial(self) -> list:
        """The 24 rank seeds the library writes into `Report.initial`."""
        seeds = []
        for token in range(24):
            cls = (self.tokens[token] >> 8) & 15
            if cls > 5:
                cls = 5
            seeds.append(rank(self.curves[token], self.score_gate, self._u16(FLOOR_OFFSET + 2 * cls) - 1))
        return seeds

    def consume(self, heads: bytes) -> dict:
        if len(heads) != HEAD_BYTES:
            raise ValueError(f"head region must be {HEAD_BYTES} bytes, got {len(heads)}")
        seeds = self.initial()
        candidates: list = []
        ordinals: list = []
        visited = 0
        ordinal = 0
        objectness_gate = self.objectness_gate
        score_gate = self.score_gate

        for stage in (0, 1):
            anchors = 3 if stage else 2
            rows = 8 if stage else 16
            columns = 12 if stage else 24
            for anchor in range(anchors):
                for row in range(rows):
                    for col in range(columns):
                        visited += 1
                        raw = read_head(heads, stage, anchor, 4, row, col)
                        if (raw & 32767) > 0x5400:
                            return {
                                "status": -2,
                                "records": [],
                                "visited": visited,
                                "accepted": 0,
                                "outputs": 0,
                                "error_field": 4,
                                "error_ordinal": ordinal,
                                "initial": seeds,
                                "ordinals": [],
                                "candidates": [],
                                "backend": self.backend,
                            }
                        objectness = sigmoid(half(raw))
                        if objectness_gate > objectness:
                            ordinal += 1
                            continue
                        token = (stage << 4) + (2 if stage else 0) + anchor
                        best = 0
                        cls = 0
                        rejected = False
                        for category in range(6):
                            wanted = (category << 8) + token
                            slot = next((k for k in range(24) if self.tokens[k] == wanted), None)
                            if slot is None:
                                continue
                            raw = read_head(heads, stage, anchor, 5 + category, row, col)
                            if (raw & 32767) > 0x5400:
                                return {
                                    "status": -2,
                                    "records": [],
                                    "visited": visited,
                                    "accepted": len(candidates),
                                    "outputs": 0,
                                    "error_field": 5 + category,
                                    "error_ordinal": ordinal,
                                    "initial": seeds,
                                    "ordinals": ordinals,
                                    "candidates": list(candidates),
                                    "backend": self.backend,
                                }
                            combined = mul(sigmoid(half(raw)), objectness)
                            if score_gate > float(combined):
                                continue
                            found = rank(self.curves[slot], float(combined), seeds[slot] - 1)
                            if found > best:
                                best = found
                                cls = category
                        if not best:
                            ordinal += 1
                            continue
                        geometry = []
                        for field in range(4):
                            raw = read_head(heads, stage, anchor, field, row, col)
                            if (raw & 32767) > 0x5400:
                                return {
                                    "status": -2,
                                    "records": [],
                                    "visited": visited,
                                    "accepted": len(candidates),
                                    "outputs": 0,
                                    "error_field": field,
                                    "error_ordinal": ordinal,
                                    "initial": seeds,
                                    "ordinals": ordinals,
                                    "candidates": list(candidates),
                                    "backend": self.backend,
                                }
                            geometry.append(half(raw))
                        index = (2 if stage else 0) + anchor
                        cx = divide(add(sigmoid(geometry[0]), F(float(col))), F(12.0 if stage else 24.0))
                        cy = divide(add(sigmoid(geometry[1]), F(float(row))), F(8.0 if stage else 16.0))
                        width = divide(
                            mul(F(float(self._u16(ANCHOR_W_OFFSET + 4 * index))), polynomial(geometry[2])),
                            F(384.0),
                        )
                        height = mul(
                            mul(F(float(self._u16(ANCHOR_H_OFFSET + 4 * index))), polynomial(geometry[3])),
                            _f32(0x3B800000),
                        )
                        ordinals.append(ordinal)
                        candidates.append((float(cx), float(cy), float(width), float(height), int(best), cls))
                        ordinal += 1
                        if len(candidates) == MAX_CANDIDATES:
                            break
                    if len(candidates) == MAX_CANDIDATES:
                        break
                if len(candidates) == MAX_CANDIDATES:
                    break
            if len(candidates) == MAX_CANDIDATES:
                break

        # Suppression runs on the memcpy'd copy (`sorted`), then `active` counts
        # that same copy, then the copy is sorted by score descending.  A box
        # zeroed by suppression can therefore still occupy one of the first
        # `active` slots — reproducing it is part of bit-exactness.
        working = list(candidates)
        factor = self.overlap_factor
        for i in range(len(working)):
            if not working[i][4]:
                continue
            for j in range(i + 1, len(working)):
                if not working[j][4]:
                    continue
                a = working[i]
                b = working[j]
                if a[5] != b[5] and not ((a[5] == 0 and b[5] == 4) or (a[5] == 4 and b[5] == 0)):
                    continue
                dx = abs(add(F(a[0]), F(-b[0])))
                dy = abs(add(F(a[1]), F(-b[1])))
                tx = mul(add(mul(F(factor), F(a[2])), mul(F(factor), F(b[2]))), F(0.5))
                ty = mul(add(mul(F(factor), F(a[3])), mul(F(factor), F(b[3]))), F(0.5))
                if not (tx > dx and ty > dy):
                    continue
                if (a[5] == 0 and b[5] == 4) or (a[5] == b[5] and b[4] > a[4]):
                    working[i] = (a[0], a[1], a[2], a[3], 0, a[5])
                    break
                working[j] = (b[0], b[1], b[2], b[3], 0, b[5])

        active = sum(1 for box in working if box[4])
        archived_qsort(working, len(working))  # same tie order as the archived qsort

        records = []
        for box in working[:active]:
            clipped = clip(box)
            if clipped is not None:
                records.append(clipped)
                if len(records) == MAX_OUTPUTS:
                    break

        return {
            "status": 0,
            "records": records,
            "visited": visited,
            "accepted": len(candidates),
            "outputs": len(records),
            "error_field": 0,
            "error_ordinal": 0,
            "initial": seeds,
            "ordinals": ordinals,
            "candidates": list(candidates),
            "backend": self.backend,
        }
