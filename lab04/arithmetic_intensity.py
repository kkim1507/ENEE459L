from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DTYPE_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5,
}

FLOPS_PER_MAC = 2
BN_PARAMS_PER_CHANNEL = 2
BN_BUFFERS_PER_CHANNEL = 2
BUFFER_DTYPE = "fp32"


@dataclass(frozen=True)
class Layer:
    name: str
    kind: str
    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    kernel: tuple[int, int] | None = None
    groups: int = 1
    bias: bool = False
    weight_dtype: str = "fp32"
    act_dtype: str = "fp32"
    reads: tuple[str, ...] = ()

    @staticmethod
    def _elements(shape: tuple[int, ...]) -> int:
        result = 1
        for dimension in shape:
            result *= dimension
        return result

    @property
    def in_elements(self) -> int:
        return self._elements(self.in_shape)

    @property
    def out_elements(self) -> int:
        return self._elements(self.out_shape)


@dataclass
class Graph:
    name: str
    input_shape: tuple[int, ...]
    layers: list[Layer]
    precision: str = "fp32"
    source: str = "unattributed"

    def __iter__(self):
        return iter(self.layers)


def dtype_bytes(name: str) -> float:
    if name not in DTYPE_BYTES:
        raise KeyError(f"unknown dtype {name!r}; known: {sorted(DTYPE_BYTES)}")
    return DTYPE_BYTES[name]


def _layer_parameters(ly: Layer) -> int:
    if ly.kind == "conv":
        kernel_height, kernel_width = ly.kernel or (1, 1)
        parameter_count = (
            ly.out_shape[0]
            * (ly.in_shape[0] // ly.groups)
            * kernel_height
            * kernel_width
        )
        return parameter_count + (ly.out_shape[0] if ly.bias else 0)

    if ly.kind == "linear":
        parameter_count = ly.out_shape[0] * ly.in_shape[0]
        return parameter_count + (ly.out_shape[0] if ly.bias else 0)

    if ly.kind == "bn":
        return BN_PARAMS_PER_CHANNEL * ly.out_shape[0]

    return 0


def computed(value: float, message: str, **extra: Any) -> dict[str, Any]:
    return {
        "type": "computed",
        "value": value,
        "message": message,
        **extra,
    }

# BEGINNING OF ASSIGNMENT 2

def _layer_macs(ly: Layer) -> int:
    if ly.kind == "conv":
        kernel_height, kernel_width = ly.kernel or (1, 1)
        input_channels_per_group = ly.in_shape[0] // ly.groups
        return (
            ly.out_elements
            * input_channels_per_group
            * kernel_height
            * kernel_width
        )

    if ly.kind == "linear":
        return ly.in_elements * ly.out_elements

    return 0


def arithmetic_intensity(graph: Graph) -> dict[str, Any]:
    layers = list(graph)
    producers = {ly.name: ly for ly in layers}
    per_layer: dict[str, dict[str, int | float]] = {}
    total_flops = 0
    total_bytes = 0.0

    for index, ly in enumerate(layers):
        if ly.reads:
            producer = producers.get(ly.reads[0])
            if producer is None:
                raise ValueError(
                    f"{ly.name}: input producer {ly.reads[0]!r} was not found"
                )
            input_dtype = producer.act_dtype
        elif index > 0:
            input_dtype = layers[index - 1].act_dtype
        else:
            input_dtype = graph.precision

        input_count = ly.in_elements * (2 if ly.kind == "add" and len(ly.reads) == 2 else 1)
        input_bytes = input_count * dtype_bytes(input_dtype)
        weight_bytes = _layer_parameters(ly) * dtype_bytes(ly.weight_dtype)
        if ly.kind == "bn":
            buffer_elements = BN_BUFFERS_PER_CHANNEL * ly.out_shape[0]
            weight_bytes += buffer_elements * dtype_bytes(BUFFER_DTYPE)

        output_bytes = ly.out_elements * dtype_bytes(ly.act_dtype)
        layer_bytes = input_bytes + weight_bytes + output_bytes
        flops = _layer_macs(ly) * FLOPS_PER_MAC
        intensity = round(flops / layer_bytes, 4) if layer_bytes else 0.0

        per_layer[ly.name] = {
            "flops": flops,
            "bytes": layer_bytes,
            "intensity": intensity,
        }
        total_flops += flops
        total_bytes += layer_bytes

    total_intensity = round(total_flops / total_bytes, 4) if total_bytes else 0.0
    message = f"{graph.name}: FLOPs over bytes, every byte counted once"

    result = computed(
        value=total_intensity,
        message=message,
        per_layer=per_layer,
        total_flops=total_flops,
        total_bytes=total_bytes,
        flops_per_mac=FLOPS_PER_MAC,
        cache_reuse_assumed=False,
        note="no fusion and no cache reuse; both would lower the byte column",
    )
    output_path = Path(__file__).with_name("arithmetic_intensity_results.json")
    output_path.write_text(json.dumps(result, indent=2))
    return result

# Synthetic graph for testing, same as one from main.py.

def build_synthetic_graph() -> Graph:
    """Build the same ResNet-like graph used by main.py."""
    layers = [
        Layer(
            name="conv1",
            kind="conv",
            in_shape=(3, 32, 32),
            out_shape=(16, 32, 32),
            kernel=(3, 3),
            bias=False,
            weight_dtype="fp32",
            act_dtype="fp32",
        ),
        Layer(
            name="bn1",
            kind="bn",
            in_shape=(16, 32, 32),
            out_shape=(16, 32, 32),
            weight_dtype="fp32",
        ),
        Layer(
            name="relu1",
            kind="relu",
            in_shape=(16, 32, 32),
            out_shape=(16, 32, 32),
        ),
        Layer(
            name="res_conv",
            kind="conv",
            in_shape=(16, 32, 32),
            out_shape=(16, 32, 32),
            kernel=(3, 3),
            bias=False,
            reads=("relu1",),
        ),
        Layer(
            name="add1",
            kind="add",
            in_shape=(16, 32, 32),
            out_shape=(16, 32, 32),
            reads=("relu1", "res_conv"),
        ),
        Layer(
            name="pool",
            kind="pool",
            in_shape=(16, 32, 32),
            out_shape=(16, 1, 1),
        ),
        Layer(
            name="flatten",
            kind="flatten",
            in_shape=(16, 1, 1),
            out_shape=(16,),
        ),
        Layer(
            name="fc",
            kind="linear",
            in_shape=(16,),
            out_shape=(10,),
            bias=True,
            weight_dtype="fp32",
            act_dtype="fp32",
        ),
    ]
    return Graph(
        name="toy_resnet",
        input_shape=(3, 32, 32),
        layers=layers,
        precision="fp32",
        source="synthetic_test",
    )

# Create report

if __name__ == "__main__":
    arithmetic_intensity(build_synthetic_graph())
    output_path = Path(__file__).with_name("arithmetic_intensity_results.json")
    print(f"Arithmetic intensity results saved to {output_path.resolve()}")
