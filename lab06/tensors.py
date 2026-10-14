from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = "1.0"

# Bytes per stored element. `int4` is here because Lecture 06 slide 26 draws the
# ladder past eight bits and a reader of this table should be able to see where
# it goes; nothing in Lab 06 quantizes to four.
DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}

# What one scale and one zero point cost on disk. A scale is a float32 and a
# zero point is an int32 in every framework this course touches, which is why
# per-channel quantization is not free and why `quantize.effective_bits` is a
# number worth printing. TensorRT stores both per channel for a convolution.
SCALE_BYTES = 4
ZERO_BYTES = 4

# glibc's constants. Chosen because they are the ones a student can look up and
# check this file against, not because they are good.
_LCG_A = 1103515245
_LCG_C = 12345
_LCG_M = 2 ** 31

# How many activation classes the shipped samples are drawn from, and how much
# wider the widest is than the narrowest. Four and about five: enough that a
# one-class calibration set clips visibly, few enough that a student can hold
# the whole picture.
CLASSES = 4


class TensorError(ValueError):
    """A description that cannot be turned into a tensor or a sample.

    Raised, not returned. Everything else in this lab reports a problem as a
    finding with a status; this is the one class of problem where there is
    nothing to report about, because the object under discussion does not
    exist. `precision.py` catches it and exits 3.
    """


@dataclass(frozen=True)
class Tensor:
    """A named dense weight tensor in row-major order.

    Frozen on purpose. Every operation in `quantize.py` returns a new object
    rather than modifying one, so a report can hold the tensor and its
    dequantized twin at once and neither can be the other by accident.
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
        """How many numbers this tensor holds. A property of the shape."""
        return elements(self.shape)

    @property
    def channels(self) -> int:
        """Output channels — the size of axis 0."""
        return self.shape[0]

    @property
    def channel_stride(self) -> int:
        """How many values one output channel occupies."""
        return elements(self.shape[1:]) if len(self.shape) > 1 else 1


@dataclass(frozen=True)
class Sample:
    """One recorded activation vector, at one layer, for one input.

    `cls` is which of the `CLASSES` input classes produced it. Real activation
    records do not carry a label and do not need one; this one does, because
    the lab's central exercise is choosing a calibration set and a student
    cannot choose a set they cannot describe.
    """

    layer: str
    cls: int
    data: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.data:
            raise TensorError(f"{self.layer}: an empty activation sample")
        if not 0 <= self.cls < CLASSES:
            raise TensorError(
                f"{self.layer}: class {self.cls} is not in 0..{CLASSES - 1}"
            )


def elements(shape: Sequence[int]) -> int:
    """The product of a shape."""
    n = 1
    for d in shape:
        n *= int(d)
    return n


def dtype_bytes(dtype: str) -> float:
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

    Lab 05 introduced this so that structured pruning was not a coin toss over
    near-identical norms. It does a second job here and the second job is the
    reason it survived into Lab 06 unchanged: with it, the output channels of a
    tensor have genuinely different ranges, so one scale for the whole tensor is
    set by the widest channel and is too coarse for the narrowest. That is
    Lecture 06 slide 23, and without `_channel_scale` the per-tensor and
    per-channel errors in this lab would differ in the fourth decimal place and
    the comparison would be a rounding artefact.
    """
    return 0.20 + 0.90 * (c + 1) / total


def _class_scale(cls: int) -> float:
    """How wide class `cls`'s activations are, relative to class 0.

    1.0, 2.0, 3.5, 5.0. Class 0 is the narrow one, so "calibrate on class 0
    only" is the pathological choice and `tests/` can assert on it by name.

    The spread is not gentle. A realistic spread would be a few tens of per
    cent and would produce a realistic accuracy loss, which on a model this
    small would be inside the noise of everything else the lab measures. The
    caricature is chosen so the mechanism is visible in one table; the handout
    says it is a caricature and the report's provenance records the factor.
    """
    return (1.0, 2.0, 3.5, 5.0)[cls % CLASSES]


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


def materialise_samples(layer: str, width: int, count: int,
                        seed: int) -> tuple[Sample, ...]:
    """`count` activation samples of `width` values each, cycling the classes.

    Sample `i` belongs to class `i % CLASSES`, so a `count` that is a multiple
    of `CLASSES` gives a balanced set. `make_fixtures.py` relies on that and
    `build_weights.py` writes a multiple.

    Activations are one-sided. Everything in this lab's model description sits
    behind a ReLU, so the samples are `abs()` of the generator's output scaled
    by the class width, and their minimum is at or near zero. That is not a
    detail: an asymmetric quantizer earns its zero point on exactly this
    distribution and a symmetric one throws away half its codes, which is
    Lecture 06 slides 19 and 20 in one array.
    """
    gen = _lcg(seed)
    out: list[Sample] = []
    for i in range(int(count)):
        cls = i % CLASSES
        scale = _class_scale(cls)
        vals = tuple(round(abs(next(gen)) * scale, 8) for _ in range(int(width)))
        out.append(Sample(layer=layer, cls=cls, data=vals))
    return tuple(out)


def load_model(path: str | Path) -> dict[str, Any]:
    """Read a model description, materialise its tensors and its samples.

    Returns `{"name", "source", "dtype", "seed", "tensors", "activations",
    "notes"}`. Raises TensorError on anything malformed, including a
    description whose `schema_version` this file does not know — a silent
    version skew in the file that decides what the weights *are* would make two
    students' reports incomparable without either of them being wrong.
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

    acts: list[Sample] = []
    for j, spec in enumerate(doc.get("activations", [])):
        if not isinstance(spec, dict):
            raise TensorError(f"{path}: activations[{j}] is not an object")
        for key in ("layer", "width", "samples"):
            if key not in spec:
                raise TensorError(f"{path}: activations[{j}] has no {key!r}")
        if int(spec["width"]) <= 0 or int(spec["samples"]) <= 0:
            raise TensorError(
                f"{path}: activations[{j}] wants a positive width and count"
            )
        acts.extend(
            materialise_samples(
                str(spec["layer"]), int(spec["width"]), int(spec["samples"]),
                base + 104729 * (j + 1),
            )
        )

    return {
        "name": doc["name"],
        "source": doc.get("source", str(path)),
        "dtype": dtype,
        "seed": base,
        "tensors": tensors,
        "activations": tuple(acts),
        "notes": list(doc.get("notes", [])),
    }


def total_parameters(tensors: Sequence[Tensor]) -> int:
    """Parameters across a list of tensors. A shape fact, not a value fact."""
    return sum(t.parameters for t in tensors)


def samples_for(activations: Sequence[Sample], layer: str) -> tuple[Sample, ...]:
    """Every shipped sample recorded at `layer`, in the order they were made."""
    return tuple(s for s in activations if s.layer == layer)


def layers(activations: Sequence[Sample]) -> tuple[str, ...]:
    """The layer names that have activation samples, first-seen order."""
    seen: list[str] = []
    for s in activations:
        if s.layer not in seen:
            seen.append(s.layer)
    return tuple(seen)