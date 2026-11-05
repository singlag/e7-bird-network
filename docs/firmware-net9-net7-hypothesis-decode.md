# NET7 / NET9 hypothesis-grade decode experiment — template #2 thresholds/curves

> **STATUS: UNVERIFIED (未驗證).** Every box produced in this experiment is a
> hypothesis artifact. None of it has ground-truth parity, so nothing here may
> be presented as a validated detection box, and nothing was wired into the GUI.

Follow-up to `firmware-decoder-table-scan.md`: the second contract carved from
`eg2070_0311b0.bi` at `0x1da0f88` ("template #2") was applied to the NET9 / NET7
stage outputs to see whether its gate/curve/anchor tables could decode those
heads into boxes.

## Inputs

- **Contract** `ipu_1da0f88_template.bin` (0x80000): token table `class<<8|token`
  for classes 0..5 × tokens 0..3 at `+0x4EE48`; 24 fp32 rank curves (1000 pts,
  `/1000`-style) at `+0x4EE78`; gates `obj 0.49977 / score 0.54131 / factor
  0.84180` at `+0x66578/0x6657C/0x66580`; fp32 scale grid (≈0.0100, 16 floats)
  at `+0x66588`. The `+0x66584` floor slots are not u16 floors here → rank
  low bound set to 0 (itself a hypothesis).
- **Heads**: NET9 `[24,16,24]/[24,8,12]` and NET7 `[26,16,24]/[26,8,12]` fp16
  stage tensors, replayed per golden NEF through `PhysicalStream`
  (`gt_child2`-style CPU inference). NET7 carries **26 planes = 2 anchors × 13
  fields**; NET9 is 2 × 12. All numerics reuse the bit-exact `decoder_port`
  primitives (`half/polynomial/sigmoid/rank/clip/archived_qsort`) — no
  reimplementation.

## Hypothesis matrix

| axis | variants |
|---|---|
| plane→(anchor,field) | **A** anchor-major `an*F+fd` · **B** field-major `fd*2+an` |
| geometry scale | **poly** `w = grid·polynomial(g)` · **sigm** `w = grid·sigmoid(g)` |
| anchor scale pairs | first 4 pairs of the 0x66588 grid (P0) / strided pairing (P1) |

Golden check: record centre vs camera AF-area rect, 4 orientation variants.

## Results

| hypothesis | obj/score gate passes | candidates | clipped records | golden hits |
|---|---|---|---|---|
| NET9-A | ~2 total | 0 | 0 | 0 |
| NET7-A | 0 | 0 | 0 | 0 |
| NET9-B + poly | obj 7359 s0 / 897 s1; cls0-score 4710/348 (all files) | 186 | **0** (clip rejects all) | 0 |
| NET7-B + poly | obj/score 907/290 | 0 (rank never > 0) | 0 | 0 |
| NET9-B + sigm | same gate passes | 1380 boxes | 1380 | 44 |
| NET7-B + sigm | gate passes exist | 0 | 0 | 0 |

Interpretation:

1. **Mapping A is dead** (gate passes ≈ 0): planes 4/5 are not objectness
   logits. **Field-major B is the right plane layout** — objectness plane
   `fd*2+anchor` passes the 0.49977 gate on 53% of stage-0 cells.
2. **B + poly geometry**: field planes hold large logits (stage absmax ≈ 62),
   `polynomial()` explodes → every candidate rejected by `clip` → 0 boxes.
3. **B + sigm geometry**: 1380 boxes across 11 NET9 files, but every box is a
   degenerate ~4×1 px square (the 0.01 scale grid × 384 px) and rank saturates
   at 999/1000 (curve direction/low semantics wrong). 44 golden hits vs ≈21
   expected by chance (AF area ≈ 0.4% of frame) — ~2× chance, **inconclusive**,
   and the boxes are unusable as detections.
4. **NET7** produced zero candidates under every variant: its 13th field per
   anchor shifts the class-plane indexing, so class ranks never clear.

## Sanity calibration (validated part)

- `rank` port parity vs `decoder_port`: 0 mismatches.
- BIRD trusted path (`E7BirdModel.infer(nnin*3, workspace)` → 0x6300 head
  buffer → `PythonDecoder(ipu_1dd8bfc)`): `status 0`, 2 accepted, 1 record.
  The golden miss on that file is the known BIRD AF-outlier from the
  enrichment report, not a harness bug. The experiment harness itself works.

## Verdict

Template #2's thresholds/curves **do not decode NET7/NET9 into validated
boxes**. Missing unknowns: true field ordering (which planes are cx/cy/w/h vs
extras), anchor-scale semantics of the 0x66588 grid, and rank-curve direction.
NET boxes remain blocked until the CEVA-XM6 object-code zone yields the real
consumer (Route B disassembly) or a field-order search over richer hypotheses
is run against more golden files. No GUI changes made.
