from __future__ import annotations

from typing import Any

from complexity import (
    BN_BUFFERS_PER_CHANNEL,
    BUFFER_DTYPE,
    FLOPS_PER_MAC,
    _layer_parameters,
)
from graph import Graph, Layer, computed, dtype_bytes


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

    return computed(
        total_intensity,
        message,
        type="computed",
        message=message,
        per_layer=per_layer,
        total_flops=total_flops,
        total_bytes=total_bytes,
        flops_per_mac=FLOPS_PER_MAC,
        cache_reuse_assumed=False,
        note="no fusion and no cache reuse; both would lower the byte column",
    )
