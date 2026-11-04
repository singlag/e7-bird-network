"""Nikon NEF RAW reader — stdlib only, one file per call, mmap so big RAWs stay light.

Pulls three things out of a NEF that the rest of the app displays or compares:

  * AFInfo2  — the maker-note record (0x00b7 on Z8/Z9, "0400" layout) holding the
    one AF box the camera locked onto, in AFImage space (full-size stored raster,
    top-left origin).  The camera records its *chosen* box only; the model may
    emit several candidates.
  * Orientation — IFD0 0x0112: how the stored raster maps to the display image.
    The preview JPEG and the NNIN are stored unrotated (landscape); the display
    image and the AF box are both rotated with this tag so they line up.
  * NNIN — the 384x256 u8 grayscale strip the camera fed to its own AF network
    (SubIFD with Compression=1, ImageWidth=384, ImageHeight=256).
  * Preview JPEG — the largest embedded JPEG (JpgFromRaw when present, else the
    IFD0 thumbnail), used as the display frame.

NEF IFDs are reachable through the header IFD0, its ExifIFD, and further IFDs
whose pointer tags sit in unpredictable places on Nikon files, so discovery
brute-scans the header region for structurally valid IFDs instead of trusting
one chain.
"""

import io
import mmap
import struct
from dataclasses import dataclass, field

TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 2, 7: 1, 8: 4, 9: 8, 10: 8}
AFINFO2_TAG = 0x00B7          # maker-note AFInfo2 on Z8/Z9 ("0400" layout)
AFINFO2_OLD = 0x0036          # older layouts, same field offsets past the version
MN_TAGS = (0x927C, 0x0125)    # maker note inside ExifIFD (v2) or classic
MODEL_WIDTH, MODEL_HEIGHT = 384, 256
# exiftool's %aFAreaModeCD Z-series lookup (values seen in AFInfo2 byte 5)
AF_MODES = {
    192: "Pinpoint", 193: "Single", 194: "Dynamic", 195: "Wide (S)", 196: "Wide (L)",
    197: "Auto", 198: "Auto (People)", 199: "Auto (Animal)",
    200: "Normal-area AF", 201: "Wide-area AF", 202: "Face-priority AF", 203: "Subject-tracking AF",
    204: "Dynamic Area (S)", 205: "Dynamic Area (M)", 206: "Dynamic Area (L)",
    207: "3D-tracking", 208: "Wide (C1/C2)",
}
# PIL transpose that turns the stored raster into the display image per EXIF tag
ORIENT_TRANSPOSE = {3: "ROTATE_180", 6: "ROTATE_270", 8: "ROTATE_90",
                    2: "FLIP_LEFT_RIGHT", 4: "FLIP_TOP_BOTTOM"}


@dataclass
class NefMeta:
    path: str
    orientation: int = 1
    af: dict | None = None            # x, y, w, h, iw, ih, mode, mode_name, version, focus_result
    nnin: bytes | None = None         # 384x256 u8 grayscale NN input strip
    preview: tuple | None = None      # (offset, length) of the chosen JPEG
    source_size: tuple = (0, 0)       # AFImage space (stored raster)
    note: str = ""


def _read_ifd(buf, off):
    """Return (entries dict, next pointer). entries[tag] = (ft, c, sz, data_off).

    A pointer into non-IFD data usually decodes as an absurd entry count, so
    anything over 64 entries is rejected outright instead of being parsed.
    """
    if off < 0 or off + 2 > len(buf):
        return {}, 0
    n = struct.unpack_from("<H", buf, off)[0]
    if n > 128:                      # Z8 maker notes carry ~77 entries; garbage pointers are wilder
        return {}, 0
    entries = {}
    for i in range(n):
        e = off + 2 + i * 12
        if e + 12 > len(buf):
            return {}, 0
        tag, ft, c = struct.unpack_from("<HHI", buf, e)
        sz = TYPE_SIZE.get(ft, 1) * c
        src = struct.unpack_from("<I", buf, e + 8)[0] if sz > 4 else e + 8
        entries[tag] = (ft, c, sz, src)
    nxt = struct.unpack_from("<I", buf, off + 2 + n * 12)[0] if n else 0
    return entries, nxt


def _valid_ifd(entries, limit):
    if not 4 <= len(entries) <= 48:
        return False
    prev = -1
    for tag, (ft, c, sz, src) in entries.items():
        if ft not in TYPE_SIZE or c == 0 or sz > 10_000_000 or src + (sz if sz > 4 else 0) > limit + 4_000_000:
            return False
        if tag <= prev:
            return False
        prev = tag
    return True


def _u32(buf, entries, tag):
    """Single SHORT/LONG entry as an int (inline when it fits, pointed when it doesn't)."""
    e = entries.get(tag)
    if e is None:
        return None
    ft, c, sz, src = e
    if ft not in (3, 4) or c != 1:
        return None
    return struct.unpack_from("<I", buf, src)[0]


def _inline_u16(buf, entries, tag):
    e = entries.get(tag)
    if e is None or e[0] != 3 or e[1] != 1:
        return None
    return struct.unpack_from("<H", buf, e[3])[0]


def _find_ifds(buf):
    """IFDs reachable through the classic Nikon chain: header IFD0, SubIFDs arrays,
    ExifIFD, next pointers.  NEF lays out IFD0 -> SubIFD (JpgFromRaw) -> SubIFD1..3
    (RAW, preview, NNIN), so following pointers beats scanning."""
    seen, out, queue = set(), [], []
    head = struct.unpack_from("<I", buf, 4)[0]
    queue.append(head)
    while queue:
        off = queue.pop()
        if not off or off in seen or off >= len(buf) - 2:
            continue
        seen.add(off)
        entries, nxt = _read_ifd(buf, off)
        if not entries:
            continue
        out.append((off, entries))
        subs = entries.get(0x0132)          # SubIFDs pointer array
        if subs and subs[0] == 4 and subs[2] <= 64:
            for i in range(subs[1]):
                queue.append(struct.unpack_from("<I", buf, subs[3] + i * 4)[0])
        if 0x8769 in entries:               # ExifIFD (holds the Nikon maker note)
            queue.append(_u32(buf, entries, 0x8769))
        if nxt:
            queue.append(nxt)
    return out


def _scan_ifds(buf, limit):
    """Brute-scan a bounded header region for valid IFDs.

    NEF repurposes tag 0x0132 as text, so SubIFDs (RAW strip, preview, NNIN)
    are not always reachable by pointer; this scan finds them.  A two-unpack
    prefilter keeps it under a second even over 400 KB.
    """
    found = []
    end = min(len(buf), limit)
    for off in range(6, end, 2):
        n = struct.unpack_from("<H", buf, off)[0]
        if not 4 <= n <= 96:
            continue
        t0 = struct.unpack_from("<H", buf, off + 2)[0]
        t1 = struct.unpack_from("<H", buf, off + 14)[0]
        if t1 <= t0:
            continue
        entries, _ = _read_ifd(buf, off)
        if _valid_ifd(entries, limit):
            found.append((off, entries))
    return found


def _afinfo2(buf, mn_base, record_off, sz):
    if sz < 0x4b:
        return None
    d = buf[mn_base + record_off: mn_base + record_off + sz]
    version = d[:4].decode("ascii", "replace")
    iw, ih = struct.unpack_from("<HH", d, 0x3E)
    ax, ay = struct.unpack_from("<HH", d, 0x42)
    aw, ah = struct.unpack_from("<HH", d, 0x46)
    if iw == 0 or ih == 0 or aw == 0 or ah == 0:
        return None
    mode = d[5]
    return {
        "x": int(ax), "y": int(ay), "w": int(aw), "h": int(ah),
        "iw": int(iw), "ih": int(ih),
        "detection": int(d[4]), "mode": int(mode),
        "mode_name": AF_MODES.get(int(mode), f"mode {mode}"),
        "coordinates_available": int(d[7]), "focus_result": int(d[0x4A]),
        "version": version,
    }


def probe_nef(path):
    """Extract orientation / AF box / NNIN / preview pointer from one NEF."""
    with open(path, "rb") as handle:
        buf = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if buf[:2] != b"II":
                raise ValueError(f"{path}: only little-endian NEF supported ({buf[:2]!r})")
            ifd0, _ = _read_ifd(buf, struct.unpack_from("<I", buf, 4)[0])
            orientation = _inline_u16(buf, ifd0, 0x0112) or 1

            af = None
            note = ""
            exif_off = _u32(buf, ifd0, 0x8769)
            if exif_off:
                exif, _ = _read_ifd(buf, exif_off)
                mn = next((exif[t] for t in MN_TAGS if t in exif), None)
                if mn and buf[mn[3]:mn[3] + 6] == b"Nikon\x00":
                    base = mn[3] + 10          # TIFF base inside the v2 maker note
                    mn_ifd = struct.unpack_from("<I", buf, base + 4)[0]
                    entries, _ = _read_ifd(buf, base + mn_ifd)
                    record = entries.get(AFINFO2_TAG) or entries.get(AFINFO2_OLD)
                    if record:
                        af = _afinfo2(buf, base, record[3], record[2])
                    if af is None:
                        note = "maker note has no usable AFInfo2 record"
                else:
                    note = "no Nikon v2 maker note"
            else:
                note = "no ExifIFD"

            nnin = None
            preview = None
            best_len = 0

            def harvest(entries):
                nonlocal nnin, preview, best_len
                w = _u32(buf, entries, 0x0100)
                h = _u32(buf, entries, 0x0101)
                comp = _inline_u16(buf, entries, 0x0103)
                spp = _inline_u16(buf, entries, 0x0115)
                if nnin is None and comp == 1 and w == MODEL_WIDTH and h == MODEL_HEIGHT and spp == 1:
                    count = _u32(buf, entries, 0x0117)
                    strip = _u32(buf, entries, 0x0111)
                    if count == MODEL_WIDTH * MODEL_HEIGHT and strip and strip + count <= len(buf):
                        nnin = bytes(buf[strip:strip + count])
                if comp == 6 and 0x0201 in entries and 0x0202 in entries:
                    start = _u32(buf, entries, 0x0201)
                    length = _u32(buf, entries, 0x0202)
                    if start is not None and length is not None and start + length <= len(buf) \
                            and buf[start:start + 2] == b"\xff\xd8" and length > best_len:
                        preview, best_len = (start, length), length

            for _, entries in _find_ifds(buf):
                harvest(entries)
            if nnin is None:                      # pointer chain dead-end: header-region scan
                for _, entries in _scan_ifds(buf, 400_000):
                    harvest(entries)
            if preview is None and 0x0201 in ifd0 and 0x0202 in ifd0:
                start = _u32(buf, ifd0, 0x0201)
                length = _u32(buf, ifd0, 0x0202)
                if start and length and start + length <= len(buf) and buf[start:start + 2] == b"\xff\xd8":
                    preview = (start, length)
            if preview is None:
                raise ValueError(f"{path}: no embedded preview JPEG found")

            size = (af["iw"], af["ih"]) if af else (0, 0)
            return NefMeta(path=str(path), orientation=orientation, af=af, nnin=nnin,
                           preview=preview, source_size=size, note=note)
        finally:
            buf.close()


def preview_image(path, meta=None):
    """Decode the embedded preview and rotate it into display orientation."""
    if meta is None:
        meta = probe_nef(path)
    if meta.preview is None:
        raise ValueError(f"NEF has no embedded preview: {path}")
    from PIL import Image

    with open(path, "rb") as handle:
        buf = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        data = bytes(buf[meta.preview[0]:meta.preview[0] + meta.preview[1]])
        buf.close()
    image = Image.open(io.BytesIO(data))
    own = image.getexif().get(0x0112, 1) or 1
    key = ORIENT_TRANSPOSE.get(own if own != 1 else meta.orientation)
    if key:
        image = getattr(image, key) if key == "ROTATE_180" else image.transpose(getattr(Image.Transpose, key))
    return image.convert("RGB")


def display_size(meta, preview_wh):
    """Display raster size = stored raster rotated the same way the preview is."""
    w, h = preview_wh
    return (h, w) if meta.orientation in (5, 6, 7, 8) else (w, h)


def _stored_corner(x, y, orientation, iw, ih):
    """One corner of a stored-raster rect -> continuous display coords (unrotated space)."""
    if orientation == 2:
        return iw - x, y
    if orientation == 3:
        return iw - x, ih - y
    if orientation == 4:
        return x, ih - y
    if orientation == 6:                      # stored -> display: rotate 90 CW
        return ih - y, x
    if orientation == 8:                      # stored -> display: rotate 90 CCW
        return y, iw - x
    return x, y


def _rot_scales(orientation, iw, ih, dw, dh):
    """Display-pixel scales per stored axis: rotated rasters swap which axis is which."""
    if orientation in (5, 6, 7, 8):
        return dh / iw, dw / ih               # stored x maps to display v, stored y to display u
    return dw / iw, dh / ih


def _stored_rect_to_display(x, y, w, h, orientation, iw, ih, dw, dh):
    u1, v1 = _stored_corner(x, y, orientation, iw, ih)
    u2, v2 = _stored_corner(x + w, y + h, orientation, iw, ih)
    su, sv = _rot_scales(orientation, iw, ih, dw, dh)
    return min(u1, u2) * su, min(v1, v2) * sv, abs(u2 - u1) * su, abs(v2 - v1) * sv


def af_display_box(meta, display_wh):
    """Camera AF box in display pixels, corner form: {x, y, w, h, mode_name}.

    Same convention `draw.draw_overlay` uses for `af_box` (top-left corner,
    width, height) — the maker note stores a rectangle, not a centre.
    """
    af, iw, ih = meta.af, meta.source_size[0], meta.source_size[1]
    if not af or iw <= 0 or ih <= 0:
        return None
    x, y, w, h = _stored_rect_to_display(af["x"], af["y"], af["w"], af["h"],
                                         meta.orientation, iw, ih, display_wh[0], display_wh[1])
    return {"x": x, "y": y, "w": w, "h": h,
            "mode": af["mode"], "mode_name": af["mode_name"]}


def model_box_display(cx, cy, w, h, meta, display_wh, score, cls):
    """Model-space box (384x256, camera NNIN space = stored raster) -> display pixels."""
    iw, ih = meta.source_size
    if iw <= 0 or ih <= 0:
        return None
    x, y, bw, bh = _stored_rect_to_display(cx - w / 2.0, cy - h / 2.0, w, h,
                                           meta.orientation, iw, ih, display_wh[0], display_wh[1])
    return (x + bw / 2.0, y + bh / 2.0, bw, bh, int(score), int(cls))


def encode_input(meta):
    """The camera's own NNIN as the model's 3x256x384 planar stream (gray x3 RGB)."""
    if meta.nnin is None:
        raise ValueError(f"{meta.path}: NEF has no 384x256 NNIN strip (older model?)")
    return meta.nnin * 3
