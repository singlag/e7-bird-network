"""Disassemble an E7/Z8 NN ``.body.bin`` into the event stream ``core.py`` runs.

Why this exists
---------------
``commands.json`` is the published, verified event list for resource744 (BIRD).
Other NN bodies found in the Z8 dump (NET4/6/7/9…) have no such file, so their
events are rebuilt from the body itself. The op sizes and operand layouts below
were checked two ways:

1. against resource744's ``commands.json`` (every control op it contains, and all
   114 of its activation tables reproduce from the body alone);
2. by whole-body coverage: BIRD, NET4, NET6 and NET7 disassemble with zero
   unconsumed words, and every DMA operand is field-sane.

Op map (major = high byte of the first word)
-------------------------------------------
  0x4c  4 B   tile config: h = op&255, w = words[1]&4095, stride = 2 if bit 12
  0x70  2 B   edge/pool config (``edge`` in core.py)
  0x44 16 B   per-batch activation config for the 16 lanes (see decode_activation)
  0x44  2 B   family member: identity activation (0x4480)
  0x20 12 B   DMA input plane -> bank0 lane
  0x22 12 B   DMA work -> bank0 lane
  0x30 12 B   DMA bank0 -> head        0x31 12 B   DMA bank1 -> head
  0x32 12 B   DMA bank0 -> work       0x33 12 B   DMA bank1 -> work
  0x68/0x69 6 B  2x2 max-pool bank0/bank1 -> the other bank
  0x40  2 B   no-op (0x4080); op 0x0000/0x0001 also no-ops
  matrix block: 6-word header + 512 B of f16 weights + optional 32 B bias

Note on 0x33: resource744 never emits it, so its size is *inferred* from its
five siblings in the same DMA family (all 12 B, identical operand layout) and
from core.py already executing major 51. NET4/NET6 confirm it: 64 records each,
all field-sane, landing in the same work-buffer region as the 0x32 copies.

NET9 adds a pooling variant family: 3-word headers of the 0xA000 family with
bit 7 set (0xA080/0xA180/0xA280/0xA380), each followed by fp16 lane constants
(0x3400 = 0.25). This loader recognises them as control records (NET9 now
disassembles with zero unconsumed words, and the four records of each group
carry the 2x2 subpixel shift bits in their operands), but the semantics are
unproven: core.py refuses to execute the family until the firmware
interpreter is read out of eg2070, so NET9 is reported NOT RUN rather than
silently approximated.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

# exact op -> size in bytes, then family default sizes for members never seen
EXACT_SIZES = {0x0000: 2, 0x0001: 2, 0x4080: 2, 0x4400: 16, 0x4480: 2, 0x68a0: 6, 0x69a0: 6}
FAMILY_SIZES = {0x00: 2, 0x20: 12, 0x21: 12, 0x22: 12, 0x23: 12, 0x30: 12, 0x31: 12,
                0x32: 12, 0x33: 12, 0x34: 2, 0x40: 2, 0x44: 2, 0x4c: 4, 0x68: 6, 0x69: 6,
                0x70: 2}
MATRIX_FLAGS_MASK = 0xA700          # bits seen in every body: 15,13 + 10,9,8
POOLING_FLAGS_MASK = 0xA780         # NET9 pooling variant: matrix flags + bit 7
DMA_MAJORS = (0x20, 0x22, 0x30, 0x31, 0x32, 0x33)
HEAD_MAJORS = (0x30, 0x31)


def _is_matrix_header(w0: int) -> bool:
    return (w0 & 0xE000) == 0xA000 and not (w0 & ~MATRIX_FLAGS_MASK & 0xFFFF)


def _is_pooling_header(w0: int) -> bool:
    """NET9-only pooling record header: the 0xA000 family with bit 7 set.

    A 3-word record (flags, two operands), followed by fp16 lane constants
    (0x3400 = 0.25) that parse as ordinary 2-byte 0x34 ops. NET9 has 16 of
    them in two clusters, in groups of four whose operands carry the
    {0,1}x{0,1} subpixel shift bits — a 2x2-shift pooling/interp stage.
    BIRD/NET4/NET6/NET7 contain zero of these words, so recognizing the
    family changes nothing for the already-verified bodies. The runtime
    refuses to execute them until the firmware interpreter is decoded.
    """
    return (w0 & 0xE000) == 0xA000 and bool(w0 & 0x0080) \
        and not (w0 & ~POOLING_FLAGS_MASK & 0xFFFF)


def decode_activation(words: list[int]) -> list[list[int]]:
    """16 lanes of [kind, sign, shift] from a 16-byte 0x44 config op.

    Verified against all 114 activation tables in resource744's commands.json.
    Lanes are packed MSB-first: kind 2 bits/lane, sign 1 bit/lane, shift 4 bits/lane.
    """
    payload = b''.join(w.to_bytes(2, 'little') for w in words[1:])
    kind_bits = int.from_bytes(payload[0:4], 'big')
    sign_bits = int.from_bytes(payload[4:6], 'big')
    shift_bits = int.from_bytes(payload[6:14], 'big')
    return [[(kind_bits >> (2 * (15 - lane))) & 3, (sign_bits >> (15 - lane)) & 1,
             (shift_bits >> (4 * (15 - lane))) & 15] for lane in range(16)]


def size_of(op: int) -> int | None:
    return EXACT_SIZES.get(op, FAMILY_SIZES.get(op >> 8))


def disassemble(body: bytes) -> tuple[list, dict, dict, list]:
    """Walk the program: events, {batch: activation lanes}, geometry, unresolved words.

    An activation epoch (the "batch" key that conv ops look their activation up
    by) starts at every matrix block whose header declares a bias table, flag
    bit 0x200: the bias belongs to the block group, so a bias-bearing block is
    the first tile of a new tensor. Numbering those blocks from 0 reproduces all
    2,941 published batch ids of resource744 with no mismatch.

    A 0x44-family op sets the activation of the epoch it appears in (the
    activation is applied after the conv group has run), so epochs with no 0x44
    op, or whose op is the 0x4480 form, keep the identity lanes [0, 0, 0].
    """
    u16 = memoryview(body).cast('H')
    events: list = []
    activations: dict[int, list] = {}
    batch: int | None = None
    unresolved: list[tuple[int, int]] = []
    pooling: list[int] = []
    pos = 0
    while pos < len(u16):
        w0 = u16[pos]
        if _is_matrix_header(w0):
            # 3-word block header (flags, address lo/hi) then the fp16 payload.
            has_bias = bool(w0 & 0x200)
            if pos + 3 + 256 + (16 if has_bias else 0) > len(u16):
                break
            if has_bias:
                batch = 0 if batch is None else batch + 1
                activations[batch] = [[0, 0, 0]] * 16
            offset = pos * 2 + 6
            bias = offset + 512 if has_bias else None
            # Event offset is the block header, matching the published event
            # list; the weight payload offset stays in the block dict.
            # 3-word block header: flags, then the destination/source address as
            # a little-endian lo/hi word pair. That is the opposite word order
            # to the DMA command operands in core.py (high word first); reading
            # it the other way round swaps ox/oy against sx/sy and moves every
            # block, which is only visible once outputs are compared.
            events.append((pos * 2, 'matrix', {
                'block': {'header_body_offset': hex(pos * 2), 'data_body_offset': hex(offset),
                          'flags': hex(w0), 'address': hex(u16[pos + 1] | u16[pos + 2] << 16),
                          'matrix_bytes': 512,
                          'bias_body_offset': hex(bias) if bias is not None else None},
                'batch': batch, 'layer': None,
                'weight': torch.from_numpy(np.frombuffer(body, '<f2', 256, offset).astype(np.float32).reshape(16, 16)),
                'bias': torch.from_numpy(np.frombuffer(body, '<f2', 16, bias).astype(np.float32))
                        if bias is not None else None}))
            pos += 3 + 256 + (16 if has_bias else 0)
            continue
        if _is_pooling_header(w0):
            if pos + 3 > len(u16):
                unresolved.append((pos * 2, w0))
                break
            events.append((pos * 2, 'control', {'offset': pos * 2, 'op': hex(w0), 'size': 6,
                                                'raw': body[pos * 2:pos * 2 + 6].hex(),
                                                'words': [w0, u16[pos + 1], u16[pos + 2]]}))
            pooling.append(pos * 2)
            pos += 3
            continue
        size = size_of(w0)
        if size is None or pos + size // 2 > len(u16):
            unresolved.append((pos * 2, w0))
            pos += 1
            continue
        words = [u16[pos + k] for k in range(size // 2)]
        major = w0 >> 8
        if major == 0x44 and size == 16 and batch is not None:
            # Only the 16-byte form carries an activation payload; the 2-byte
            # 0x4480 form has none and leaves the epoch's table at identity
            # (both cases verified against resource744's 114 published tables).
            activations[batch] = decode_activation(words)
        events.append((pos * 2, 'control', {'offset': pos * 2, 'op': hex(w0), 'size': size,
                                           'raw': body[pos * 2:pos * 2 + size].hex(),
                                           'words': words}))
        pos += size // 2
    geometry = head_geometry(body, events)
    audit = {'failures': [] if not unresolved else [f'{len(unresolved)} unresolved words'],
             'batches': len(activations),
             'matrices': sum(1 for _, k, _ in events if k == 'matrix'),
             'pooling_records': len(pooling),
             'unresolved': unresolved[:20]}
    return events, activations, audit, geometry


def head_geometry(body: bytes, events: list) -> dict:
    """Head-buffer size and stage shapes, read from the head-write DMAs.

    Each stage groups the head DMAs by tile geometry: planes = number of head
    writes, row pitch = DMA pitch, w = tile width. Ordered by buffer position.
    For resource744 this reproduces the published fine[22,16,24] /
    coarse[33,8,12] layout with a 25344-byte head region exactly.
    """
    u16 = memoryview(body).cast('H')
    hw = None
    stages: dict[tuple[int, int, int], dict] = {}
    work_end = 0
    for pos, kind, v in events:
        if kind == 'matrix':
            continue
        words = v['words']
        major = words[0] >> 8
        if major == 0x4c:
            hw = (words[0] & 255, words[1] & 4095, 2 if words[1] & 4096 else 1)
        elif major in DMA_MAJORS:
            h, w, stride = hw
            addr = words[1] << 16 | words[2]
            pitch = words[3]
            if major >= 0x32:
                work_end = max(work_end, addr // 2 + h * (pitch // 2))
            if major in HEAD_MAJORS:
                key = (h, w, pitch // 2)
                st = stages.setdefault(key, {'h': h, 'w': w, 'pitch': pitch // 2, 'planes': 0,
                                             'base': addr // 2, 'end': addr // 2})
                st['planes'] += 1
                st['base'] = min(st['base'], addr // 2)
                st['end'] = max(st['end'], addr // 2 + h * (pitch // 2))
    ordered = sorted(stages.values(), key=lambda s: s['base'])
    return {'head_halves': max((s['end'] for s in ordered), default=0),
            'stages': ordered, 'work_halves': work_end}


def describe(path: Path) -> dict:
    """Structural summary of one body, for listing/diagnostics."""
    body = Path(path).read_bytes()
    events, activations, audit, geometry = disassemble(body)
    ops: dict[str, int] = {}
    for _, kind, v in events:
        key = 'matrix' if kind == 'matrix' else v['op']
        ops[key] = ops.get(key, 0) + 1
    return {'path': str(path), 'body_size': len(body), 'sha256': hashlib.sha256(body).hexdigest(),
            'audit': {k: v for k, v in audit.items() if k != 'unresolved'},
            'unresolved_words': len(audit['unresolved']), 'ops': ops, 'geometry': geometry}
