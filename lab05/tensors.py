"""A weight tensor, small enough to reason about and dependency-free.

Provided complete in both trees. No torch, no numpy — the same choice Lab 04
made and for a related reason: a mask is not a fact about a framework, and the
arithmetic that decides which weights go is arithmetic you should be able to do
without one.

## Why the weights are generated rather than shipped

`models/*.json` describes tensors by name and shape and carries a seed. The
values are produced here, by a linear congruential generator with the constants
written down in `_LCG_*` below, so that every machine that runs this lab gets
the same floats to eight decimal places and two students comparing
`sparsity.json` are comparing the same experiment. Shipping the weights as
literals would have been the same thing in more bytes and would have made
`models/build_weights.py` a file nobody could check.

The generator is not a good source of random numbers and does not need to be.
It needs to be identical everywhere, which it is, and to produce a tensor whose
channels have visibly different norms, which is what `_channel_scale` is for:
without it every output channel of a uniform tensor has the same expected
L2 norm, channel pruning becomes a tie-break over noise, and `tests/` would be
asserting on which way a float comparison happened to fall.

## What a Tensor is not

It is not a torch tensor and it is not trying to be. It is a shape and a flat
tuple in row-major (C) order, which is the layout every framework this course
touches uses for a convolution weight `[C_out, C_in, kH, kW]`. The one
consequence that matters for `prune.py`: output channel `c` occupies the
contiguous run of `stride = C_in·kH·kW` values starting at `c·stride`, so
"remove a channel" is "delete a contiguous slice", which is why structured
pruning produces a tensor a kernel can use and unstructured pruning does not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = "1.0"

# Bytes per element. `int8` is here because Lecture 06 arrives in a week and
# `sparsity.json` will be read next to a quantised model's; nothing in Lab 05
# uses it.
DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}

# One index per stored value, four bytes wide. A real CSR layout also carries a
# row pointer array, which is small and which this model ignores — and the
# docstring on `prune.bytes_stored` says so, because an accounting that quietly
# rounds in its own favour is the thing this lab is about.
INDEX_BYTES = 4

# glibc's constants. Chosen because they are the ones a student can look up and
# check this file against, not because they are good.
_LCG_A = 1103515245
_LCG_C = 12345
_LCG_M = 2 ** 31


class TensorError(ValueError):
    """A description that cannot be turned into a tensor.

    Raised, not returned. Everything else in this lab reports a problem as a
    finding with a status; this is the one class of problem where there is
    nothing to report about, because the object under discussion does not
    exist. `sweep.py` catches it and exits 3.
    """


@dataclass(frozen=True)
class Tensor:
    """A named dense weight tensor in row-major order.

    Frozen on purpose. Every operation in `prune.py` returns a new Tensor
    rather than modifying one, so a report can hold the before and the after
    of the same weight and neither can be the other one by accident.
    """

    name: str
    shape: tuple[int, ...]
    data: tuple[float, ...]
    dtype: str = "fp32"

    def __post_init__(self) -> None:
        if not self.shape or any(d <= 0 for d in self.shape):
            raise TensorError(f"{self.name}: shape {self.shape} is not a shape")
        if self.dtype not in DTYPE_BYTES:
            raise TensorError(f"{self.name}: unknown dtype {self.dtype!r}")
        if len(self.data) != elements(self.shape):
            raise TensorError(
                f"{self.name}: shape {self.shape} wants {elements(self.shape)} "
                f"values, got {len(self.data)}"
            )

    @property
    def parameters(self) -> int:
        """How many numbers this tensor has room for.

        A property of the shape and of nothing else. It does not change when
        values are set to zero, which is the sentence the whole lab is about.
        """
        return elements(self.shape)

    @property
    def channels(self) -> int:
        """Output channels — the size of axis 0."""
        return self.shape[0]

    @property
    def channel_stride(self) -> int:
        """How many values one output channel occupies."""
        return elements(self.shape[1:]) if len(self.shape) > 1 else 1


def elements(shape: Sequence[int]) -> int:
    """The product of a shape."""
    n = 1
    for d in shape:
        n *= int(d)
    return n


def dtype_bytes(dtype: str) -> int:
    """Bytes per element, or a TensorError naming the ones that exist."""
    try:
        return DTYPE_BYTES[dtype]
    except KeyError:
        raise TensorError(
            f"unknown dtype {dtype!r}; known: {sorted(DTYPE_BYTES)}"
        ) from None


def _lcg(seed: int) -> Iterable[float]:
    """An endless stream of floats in (-1, 1), identical on every machine."""
    x = seed % _LCG_M
    while True:
        x = (_LCG_A * x + _LCG_C) % _LCG_M
        yield (x / _LCG_M) * 2.0 - 1.0


def _channel_scale(c: int, total: int) -> float:
    """A per-output-channel multiplier, increasing with the channel index.

    Without this the channels of a uniformly generated tensor have the same
    expected norm and the structured sweep becomes a coin toss between
    near-identical scores. With it, channel 0 is the smallest and channel
    C-1 the largest, every time, on every machine — so `tests/` can assert
    which channels a 50% channel prune keeps instead of asserting how many.

    It is also, incidentally, true of real networks more often than not:
    channels differ in scale, which is the entire premise of Network
    Slimming (Lecture 05 slide 28).
    """
    return 0.20 + 0.90 * (c + 1) / total


def materialise(name: str, shape: Sequence[int], seed: int,
                dtype: str = "fp32") -> Tensor:
    """Build the tensor `models/*.json` describes.

    Deterministic in `(shape, seed)` and in nothing else. Two tensors in the
    same model get different seeds — see `load_model` — so that a network is
    not four copies of one weight.
    """
    shape = tuple(int(d) for d in shape)
    n = elements(shape)
    stride = elements(shape[1:]) if len(shape) > 1 else 1
    gen = _lcg(seed)
    vals = []
    for i in range(n):
        v = next(gen)
        vals.append(round(v * _channel_scale(i // stride, shape[0]), 8))
    return Tensor(name=name, shape=shape, data=tuple(vals), dtype=dtype)


def load_model(path: str | Path) -> dict[str, Any]:
    """Read a model description and materialise its tensors.

    Returns `{"name", "source", "dtype", "tensors": [Tensor, ...], "notes"}`.
    Raises TensorError on anything malformed, including a description whose
    `schema_version` this file does not know — a silent version skew in a file
    that decides what the weights *are* would make two students' reports
    incomparable without either of them being wrong.
    """
    path = Path(path)
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise TensorError(f"{path}: not JSON — {exc}") from None
    except OSError as exc:
        raise TensorError(f"{path}: cannot read — {exc}") from None

    if not isinstance(doc, dict):
        raise TensorError(f"{path}: top level is {type(doc).__name__}, want object")
    got = doc.get("schema_version")
    if got != SCHEMA_VERSION:
        raise TensorError(
            f"{path}: schema_version {got!r}, this package speaks "
            f"{SCHEMA_VERSION!r}"
        )
    for key in ("name", "seed", "tensors"):
        if key not in doc:
            raise TensorError(f"{path}: no {key!r}")
    if not isinstance(doc["tensors"], list) or not doc["tensors"]:
        raise TensorError(f"{path}: 'tensors' must be a non-empty list")

    dtype = doc.get("dtype", "fp32")
    dtype_bytes(dtype)
    base = int(doc["seed"])
    tensors = []
    for i, spec in enumerate(doc["tensors"]):
        if not isinstance(spec, dict) or "name" not in spec or "shape" not in spec:
            raise TensorError(f"{path}: tensor {i} needs 'name' and 'shape'")
        shape = spec["shape"]
        if not isinstance(shape, list) or not shape:
            raise TensorError(f"{path}: tensor {spec.get('name')} has no shape")
        # The seed is offset by the tensor's position, so the tensors of one
        # model differ from each other and the model as a whole still depends
        # on one number in the file.
        tensors.append(materialise(spec["name"], shape, base + 7919 * i, dtype))

    return {
        "name": doc["name"],
        "source": doc.get("source", str(path)),
        "dtype": dtype,
        "seed": base,
        "tensors": tensors,
        "notes": list(doc.get("notes", [])),
    }


def total_parameters(tensors: Sequence[Tensor]) -> int:
    """Parameters across a list of tensors. A shape fact, not a value fact."""
    return sum(t.parameters for t in tensors)