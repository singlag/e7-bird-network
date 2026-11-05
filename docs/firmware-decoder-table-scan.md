# Firmware decoder-table carve scan (Route A data-region step)

Date: 2026-11-05. Target: `/home/garykan/Documents/Z8/analysis/modules/eg2070_0311b0.bi`
(Z8 v3.11 NN module, 81 MB) plus `li2070/ex2070/vr2070` modules and the raw
`Z_8_0311.bin`, `ZR__0111.bin`, `Z_50_0260.bin`, `Z_6_0380.bin` images.

## Method (no CEVA-XM6 disassembler needed)

The published host decoder (`wide_reference.c`) reads the per-model decision
tables from the task `ipu` blob at fixed contract offsets:

| item | ipu offset | BIRD verified value |
|---|---|---|
| rank header, 24 u16 `(class<<8|token)` words | `0x4ee48` (=323144) | classes 0–5 × tokens {0,1,18,19,20} |
| 24 rank curves, fp32 × 1000 | `0x4ee78 + 4000*k` | matches published `rank-tables.bin` |
| objectness gate f32 | `0x66578` | 0.1 |
| score gate f32 | `0x6657c` | 0.1 |
| per-class reference f32 | `0x66580` | 0.8 |
| per-class rank u16 ref ×6 | `0x66584` | 300 |
| anchor config u16 | `0x6658C–0x665A4` | 0x3ad/0x321 + 60/80/160/140/200 |

Scan: every 4-byte-aligned position in each file was tested for the fp32 gate
pair at +0x66578/+0x6657c in (0.02, 1], then full token-struct validation at
+0x4ee48 (words with class ≤5, token ∈ {0,1,18,19,20}, 24 distinct), then curve
sanity (24 × fp32/1000, finite, values in [0,1]).

## Result — exactly two complete decoder contracts in eg2070

1. **`ipu base 0x1dd8bfc` = NET5/BIRD** — rank table byte-identical to the
   published `rank-tables.bin` (96,048 B) and gate struct identical to
   `gate-anchor-tables.bin` (obj 0.1 / score 0.1 / ref 0.8 / rank-ref 300×6).
   This re-confirms the published tables are a direct carve of the firmware,
   validating the Route A carving approach without any chip ISA.
2. **`ipu base 0x1da0f88` = second model, unpublished** — complete contract:
   - tokens: 6 classes × {0,1,18,19} (2 anchors/stage/class, 24 curves),
   - all 24 rank curves intact fp32/1000, finite, values in [0,1],
   - gate at +0x66578 is a **variant layout**: fp32 thresholds
     `[0.49977, 0.54131, 0.84180, 0.92512]` followed by fp32 anchor-grid
     values (0.01 / 0.01001 ramps), i.e. a newer decoder-contract version
     (fp32 anchors instead of BIRD-style u16 pixel anchors).
   - `0x1a961e4` holds a duplicate copy of the same fp32 threshold pair;
     a token-struct copy also exists at `0x1a9def4` inside the CDNN runtime
     data section.
   - ~1,085 dword pointers into the template region come from the command-stream
     band `0x1ac0000–0x1cc0000` (task definitions ahead of the NET1 container).
   - Geometry hypothesis: matches a 2-anchor-per-stage, 6-class, 2-stage net
     (24ch-style heads: NET7/NET9 class); **not validated against any ground
     truth — treat as hypothesis only.**

3. **No other complete contract exists anywhere** in eg2070 or the other Z8
   modules (relaxed vocab-≥20 near-miss scan: 701 hits, all fp32 curve ramps).
   The raw camera images (`Z_8_0311.bin`, `ZR__0111.bin`, …) are encrypted
   containers: 0 hits. So NET1/2/3/4/6/8/9 decode tables are not stored
   statically in the module — they must be assembled at task-load time by the
   CDNN runtime (or live in a module we do not have).

## Carve artifacts (local only, NOT committed — firmware-derived data)

- `/home/garykan/.hermes/cache/scratch/ipu_found/ipu_1dd8bfc_{rank,gate,template}.bin`
- `/home/garykan/.hermes/cache/scratch/ipu_found/ipu_1da0f88_{rank,gate,template}.bin`
- Scan scripts: `dig_tables7.py` (strict contract), `dig_tables8.py` (all files + relaxed).

## Consequences for NET4/6/7/9 boxes

- Route A feasibility is proven at the table level: the decode contract is
  recoverable from firmware data regions by the fixed-offset fingerprint.
- BIRD/NET5 parity is now double-anchored (published file == firmware carve).
- The second template gives a testable decoder contract (0.5/0.54/0.84/0.925
  thresholds + 24 curves) — usable only as a labelled hypothesis experiment
  against NET7/NET9 heads, pending a device-captured ground truth to confirm
  which model it belongs to and which gate-layout version its decoder uses.
- Unlocking NET4/6/7/9 boxes honestly still requires the per-model decision
  captures (Route A device dump) or user-labelled shots.
