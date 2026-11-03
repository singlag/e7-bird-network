"""Selectable models for the GUI / CLI.

Keeping the list here means the model dropdown can grow without touching the
GUI: add a `ModelSpec` and its factory.  Today there is exactly one model.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    note: str
    factory: Callable[..., object]


def _build_e7(threads: int = 4, decoder_mode: str = "auto"):
    from e7model import E7BirdModel

    return E7BirdModel(threads=threads, decoder_mode=decoder_mode)


REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="e7-bird-resource744",
        label="E7 Bird Network (resource744)",
        note="Nikon E7 resource744 bird detector · 3:2 384x256 input · candidate boxes with native rank score",
        factory=_build_e7,
    ),
)


def specs() -> tuple[ModelSpec, ...]:
    return REGISTRY


def labels() -> list[tuple[str, str]]:
    return [(spec.key, spec.label) for spec in REGISTRY]


def default_key() -> str:
    return REGISTRY[0].key


def find(key: str) -> ModelSpec:
    for spec in REGISTRY:
        if spec.key == key:
            return spec
    raise KeyError(f"unknown model key: {key} (available: {', '.join(s.key for s in REGISTRY)})")


def build(key: str, threads: int = 4, decoder_mode: str = "auto"):
    return find(key).factory(threads=threads, decoder_mode=decoder_mode)
