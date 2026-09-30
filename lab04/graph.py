from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable



DTYPE_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5,
}

PARAMETRIC = frozenset({"conv", "linear", "bn"})


@dataclass(frozen=True)
class Layer:
    name: str
    kind: str
    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    kernel: tuple[int, int] | None = None
    stride: int = 1
    groups: int = 1
    bias: bool = False
    weight_dtype: str = "fp32"
    act_dtype: str = "fp32"
    reads: tuple[str, ...] = ()

    def elements(self, shape: tuple[int, ...]) -> int:
        n = 1
        for d in shape:
            n *= d
        return n

    @property
    def in_elements(self) -> int:
        return self.elements(self.in_shape)

    @property
    def out_elements(self) -> int:
        return self.elements(self.out_shape)

    @property
    def out_spatial(self) -> int:
        if len(self.out_shape) == 3:
            return self.out_shape[1] * self.out_shape[2]
        return 1


@dataclass
class Graph:
    name: str
    input_shape: tuple[int, ...]
    layers: list[Layer]
    precision: str = "fp32"
    source: str = "unattributed"
    notes: str = ""
    _by_name: dict[str, Layer] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_name = {ly.name: ly for ly in self.layers}

    def __iter__(self) -> Iterable[Layer]:
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)

    def get(self, name: str) -> Layer | None:
        return self._by_name.get(name)

    def conditions(self) -> dict[str, Any]:
        """The fields that must match before two graphs may be ranked together."""
        return {
            "input_shape": list(self.input_shape),
            "precision": self.precision,
        }

    def validate(self) -> list[str]:
        problems: list[str] = []
        prev: Layer | None = None
        seen: set[str] = set()

        for ly in self.layers:
            if ly.kind not in {
                "conv", "linear", "pool", "relu", "add", "bn", "flatten",
            }:
                problems.append(f"{ly.name}: unknown layer kind {ly.kind!r}")
            for label, dt in (("weight", ly.weight_dtype), ("act", ly.act_dtype)):
                if dt not in DTYPE_BYTES:
                    problems.append(f"{ly.name}: unknown {label} dtype {dt!r}")

            if not ly.reads and prev is not None and ly.in_shape != prev.out_shape:
                problems.append(
                    f"{ly.name}: input {ly.in_shape} does not match "
                    f"{prev.name} output {prev.out_shape}"
                )

            for target in ly.reads:
                if target not in seen:
                    problems.append(
                        f"{ly.name}: reads {target!r}, which is not an earlier layer"
                    )

            if ly.kind == "add" and len(ly.reads) == 2:
                shapes = {self.get(t).out_shape for t in ly.reads if self.get(t)}
                if len(shapes) > 1:
                    problems.append(
                        f"{ly.name}: add over tensors of different shapes {sorted(shapes)}"
                    )

            if ly.kind == "conv":
                c_in = ly.in_shape[0]
                c_out = ly.out_shape[0]
                if ly.groups < 1 or c_in % ly.groups or c_out % ly.groups:
                    problems.append(
                        f"{ly.name}: groups={ly.groups} does not divide "
                        f"C_in={c_in} / C_out={c_out}"
                    )
                if len(ly.in_shape) == 3 and len(ly.out_shape) == 3 and ly.stride:
                    want = -(-ly.in_shape[1] // ly.stride)
                    if abs(ly.out_shape[1] - want) > 1:
                        problems.append(
                            f"{ly.name}: output H={ly.out_shape[1]} is not "
                            f"consistent with input H={ly.in_shape[1]} at "
                            f"stride {ly.stride}"
                        )

            seen.add(ly.name)
            prev = ly

        return problems


# --- loading ----------------------------------------------------------------


def load_graph(path: str | Path) -> Graph:
    """Read one model description from JSON."""
    raw = json.loads(Path(path).read_text())
    layers = [
        Layer(
            name=d["name"],
            kind=d["kind"],
            in_shape=tuple(d["in_shape"]),
            out_shape=tuple(d["out_shape"]),
            kernel=tuple(d["kernel"]) if d.get("kernel") else None,
            stride=int(d.get("stride", 1)),
            groups=int(d.get("groups", 1)),
            bias=bool(d.get("bias", False)),
            weight_dtype=d.get("weight_dtype", raw.get("precision", "fp32")),
            act_dtype=d.get("act_dtype", raw.get("precision", "fp32")),
            reads=tuple(d.get("reads", ())),
        )
        for d in raw["layers"]
    ]
    return Graph(
        name=raw["name"],
        input_shape=tuple(raw["input_shape"]),
        layers=layers,
        precision=raw.get("precision", "fp32"),
        source=raw.get("source", "unattributed"),
        notes=raw.get("notes", ""),
    )


def load_graphs(directory: str | Path) -> list[Graph]:
    """Every `*.json` in a directory, in sorted filename order."""
    return [load_graph(p) for p in sorted(Path(directory).glob("*.json"))]


def load_bench(path: str | Path) -> dict[str, Any] | None:
    """Read one Lab 03 record, or None if it is absent or not JSON.

    Returning None rather than raising is the whole reason this is a function.
    A missing benchmark record is the normal state of the world in Stage A —
    you have not run anything yet — and every function downstream has to
    produce a report anyway, with the latency column honestly unknown. A loader
    that raises turns "I have not measured this yet" into a crash.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# --- findings ---------------------------------------------------------------


def unknown(source: str, why: str) -> dict[str, Any]:
    return {"value": None, "source": source, "status": "unknown", "detail": why}


def computed(value: Any, source: str, **extra: Any) -> dict[str, Any]:
    return {"value": value, "source": source, "status": "computed", **extra}


def measured(value: Any, source: str, **extra: Any) -> dict[str, Any]:
    return {"value": value, "source": source, "status": "measured", **extra}


def is_answered(finding: dict[str, Any] | None) -> bool:
    return bool(finding) and finding.get("status") in {"computed", "measured"}


def dtype_bytes(name: str) -> float:
    if name not in DTYPE_BYTES:
        raise KeyError(f"unknown dtype {name!r}; known: {sorted(DTYPE_BYTES)}")
    return DTYPE_BYTES[name]