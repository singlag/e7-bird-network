"""Run the Z8 NN bodies through the emulator and prove it reproduces BIRD exactly.

Usage: python run_bodies.py [body ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

from core import PhysicalStream, torch

FIXTURE = Path('fixtures/random/input-chw-u8.bin')


def image():
    raw = FIXTURE.read_bytes()
    return torch.from_numpy(np.frombuffer(raw, np.uint8).reshape(1, 3, 256, 384).astype(np.float32) / 256)


def stats(t):
    a = t.numpy().astype(np.float32)
    return (tuple(t.shape), round(float(np.abs(a).mean()), 4), round(float(a.max()), 3),
            'nonfinite' if not np.isfinite(a).all() else 'finite')


def main(bodies):
    x = image()
    reference = PhysicalStream()
    print(f'resource744 via commands.json: {reference.audit}')
    with torch.inference_mode():
        ref = reference(x)
    print('  heads', [stats(t) for t in ref])
    entries = {'resource744 (BIRD, from body)': 'resource744.body.bin'}
    for i, p in enumerate(bodies):
        entries[Path(p).stem] = p
    for label, path in entries.items():
        try:
            model = PhysicalStream.from_body(path)
        except ValueError as exc:
            print(f'{label}: NOT RUN — {exc}')
            continue
        with torch.inference_mode():
            try:
                out = model(x)
            except ValueError as exc:
                print(f'{label}: PARSED OK, NOT RUN — replay refused: {exc}')
                continue
        head = ' | '.join(f'{t.shape[1]} planes {tuple(t.shape[2:])}' for t in out)
        print(f'{label}: {model.audit["matrices"]} matrices, {model.audit["batches"]} epochs, '
              f'head {model.head_halves * 2} B -> {head}; values {stats(torch.cat([t.reshape(t.shape[0], -1) for t in out], 1))[1:]}')
        if label.startswith('resource744'):
            print('  bit-identical to the published BIRD heads:',
                  all(torch.equal(a, b) for a, b in zip(ref, out)))


if __name__ == '__main__':
    default = sorted(Path('/home/garykan/Documents/Z8/analysis/nn_models').glob('NET*_body.bin'))
    main(sys.argv[1:] or [str(p) for p in default])
