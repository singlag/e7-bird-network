"""Available detection models.

Built-in: the upstream E7 bird network from this repo — untouched package
(`network.py` + `core.py` + `resource744.body.bin` + `commands.json`), original
weights, original command order, `weights_modified: false`.  Its body file name
is fixed by the upstream manifest pins; everything else in the list is found by
scanning `<repo>/models/` **by content hash** — the on-disk file name carries
no meaning and can be anything.  The model list shows the file name so the
user can see which file each row loads; selecting a model makes the program
load that exact file immediately (GUI: `_model_changed` → `model.load()`).

Z8 NET bodies are distribution-restricted firmware blobs: the user copies them
into `models/` from the Z8 extraction.  Their roles are hypothesis-level
(references/model_backends_and_z8_models.md), never claimed as fact, and none
has a published decode contract, so they load/verify but cannot produce boxes.
"""

import e7model
import e7netmodel
import model_files
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ModelSpec:
    key: str
    label: str
    note: str
    factory: Optional[Callable[..., object]]
    decodable: bool = True     # can produce detection boxes (published decode contract)
    runnable: bool = True      # loads / runs through the emulator at all


def _built_in_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            key=e7model.E7BirdModel.key,
            label=f"{e7model.E7BirdModel.label} — {model_files.BUILT_IN_BODY.name}",
            note=e7model.E7BirdModel.note,
            factory=lambda **kw: e7model.E7BirdModel(**kw),
        ),
    ]


def _discovered_specs() -> list[ModelSpec]:
    specs = []
    for meta in model_files.discover():
        if meta.get("built_in"):
            continue
        extra = (
            f" (also found as: {', '.join(meta['extra_files'])})"
            if meta.get("extra_files") else ""
        )
        if meta.get("unidentified"):
            note = ("這個 .bin 唔喺 verified Z8 body 表內（SHA-256 冇匹配）— "
                    "只列出，唔會 load。")
            factory = None
        elif meta["decodable"]:
            note = meta["role"]
            factory = lambda path=meta["path"], **kw: e7netmodel.Z8NetModel(path)
        else:
            note = meta["role"] + (
                "" if meta["runnable"] else "（emulator 未能完整 parse，只列出）")
            factory = (
                (lambda path=meta["path"], **kw: e7netmodel.Z8NetModel(path))
                if meta["runnable"] else None)
        specs.append(ModelSpec(
            key=meta["key"],
            label=f"{meta['name']} — {meta['filename']}{extra}",
            note=note,
            factory=factory,
            decodable=False,
            runnable=meta["runnable"] and not meta.get("unidentified", False),
        ))
    return specs


def registry() -> dict[str, ModelSpec]:
    out = {}
    for spec in _built_in_specs() + _discovered_specs():
        out[spec.key] = spec
    return out


def default_key() -> str:
    return e7model.E7BirdModel.key


def labels() -> list[tuple[str, str]]:
    return [(key, spec.label) for key, spec in registry().items()]


def find(key: str) -> ModelSpec:
    reg = registry()
    if key not in reg:
        known = ", ".join(reg) or "(無)"
        raise ValueError(f"unknown model: {key} (known: {known})")
    return reg[key]


def build(key: str, **kwargs) -> object:
    spec = find(key)
    if spec.factory is None:
        raise ValueError(f"model cannot load: {spec.label} — {spec.note}")
    return spec.factory(**kwargs)
