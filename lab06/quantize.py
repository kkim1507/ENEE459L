from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from pathlib import Path
import json

from tensors import (
    SCALE_BYTES,
    ZERO_BYTES,
    Sample,
    Tensor,
    TensorError,
    load_model,
    dtype_bytes,
    elements,
    total_parameters,
    materialise_samples,
)

from quantize_helper import (
    _channel_slice,
    _qmin_qmax,
    _range_of,
    _round_half_up,
    _scale_zero,
    CALIBRATIONS,
    CLIP_METHODS,
    DEFAULT_BITS,
    DEFAULT_PERCENTILE,
    GRANULARITIES,
    MSE_CANDIDATES,
    PRECISIONS,
)


# helper class and functions to carry the quantized tensor and its metadata around together
@dataclass(frozen=True)
class Quantized:
    name: str
    shape: tuple[int, ...]
    codes: tuple[int, ...]
    scales: tuple[float, ...]
    zeros: tuple[int, ...]
    bits: int
    granularity: str
    symmetric: bool
    clipped: int = 0

    @property
    def parameters(self) -> int:
        return elements(self.shape)

    @property
    def channels(self) -> int:
        return self.shape[0]

    @property
    def channel_stride(self) -> int:
        return elements(self.shape[1:]) if len(self.shape) > 1 else 1

    @property
    def scale_count(self) -> int:
        """How many (scale, zero) pairs this tensor carries on disk."""
        return len(self.scales)

    @property
    def clipped_fraction(self) -> float:
        return round(self.clipped / len(self.codes), 8) if self.codes else 0.0

def to_fp16(t: Tensor) -> Tensor:
    out = []
    for v in t.data:
        if not -65504.0 <= v <= 65504.0:
            raise TensorError(
                f"{t.name}: {v} is outside binary16's range; FP16 would store "
                "infinity and every number after this one would be a lie"
            )
        out.append(round(struct.unpack("e", struct.pack("e", float(v)))[0], 8))
    return Tensor(name=t.name, shape=t.shape, data=tuple(out), dtype="fp16")


def as_tensor(name: str, values: Sequence[float]) -> Tensor:
    vals = tuple(float(v) for v in values)
    if not vals:
        raise TensorError(f"{name}: no values")
    return Tensor(name=name, shape=(len(vals),), data=vals, dtype="fp32")


def select_calibration(activations: Sequence[Sample],
                       policy: str = "all") -> tuple[float, ...]:
    acts = tuple(activations)
    if policy == "all":
        chosen = acts
    elif policy == "two_class":
        chosen = tuple(s for s in acts if s.cls in (0, 1))
    elif policy == "first_8":
        chosen = acts[:8]
    elif policy == "one_class":
        chosen = tuple(s for s in acts if s.cls == 0)
    else:
        raise TensorError(
            f"unknown calibration policy {policy!r}; known: {list(CALIBRATIONS)}"
        )
    if not chosen:
        raise TensorError(f"calibration policy {policy!r} selected no samples")
    return tuple(v for s in chosen for v in s.data)


########################################
# start filling in the code from here.
#########################################

def quantize_tensor(t: Tensor, bits: int = DEFAULT_BITS,
                    granularity: str = "per_tensor", symmetric: bool = True,
                    clip: str = "minmax",
                    percentile: float = DEFAULT_PERCENTILE,
                    given_range: tuple[float, float] | None = None) -> Quantized:
    if granularity not in GRANULARITIES:
        raise TensorError(
            f"unknown granularity {granularity!r}; known: {list(GRANULARITIES)}"
        )
    if clip not in CLIP_METHODS:
        raise TensorError(f"unknown clip method {clip!r}; known: {list(CLIP_METHODS)}")
    qmin, qmax = _qmin_qmax(bits, symmetric)
    if granularity == "per_channel":
        groups = [_channel_slice(t, c) for c in range(t.channels)]
    else:
        groups = [(0, len(t.data))]
    codes = [0] * len(t.data)
    scales: list[float] = []
    zeros: list[int] = []
    clipped = 0
    for lo, hi in groups:
        if given_range is not None:
            rlo, rhi = float(given_range[0]), float(given_range[1])
        else:
            rlo, rhi = _range_of(t.data[lo:hi], clip, percentile, bits)
        s, z = _scale_zero(rlo, rhi, bits, symmetric)
        scales.append(s)
        zeros.append(z)
        for i in range(lo, hi):
            q = _round_half_up(t.data[i] / s) + z
            if q < qmin or q > qmax:
                clipped += 1
            codes[i] = max(qmin, min(qmax, q))
    return Quantized(
        name=t.name, shape=tuple(t.shape), codes=tuple(codes),
        scales=tuple(scales), zeros=tuple(zeros), bits=int(bits),
        granularity=granularity, symmetric=bool(symmetric), clipped=clipped,
    )


def dequantize(qt: Quantized) -> Tensor:
    stride = qt.channel_stride if qt.granularity == "per_channel" else len(qt.codes)
    if stride <= 0:
        raise TensorError(f"{qt.name}: a group of {stride} values is not a group")
    out = []
    for i, q in enumerate(qt.codes):
        g = i // stride
        if g >= len(qt.scales):
            raise TensorError(
                f"{qt.name}: value {i} falls in group {g} and there are "
                f"{len(qt.scales)} scales"
            )
        out.append(round(qt.scales[g] * (int(q) - qt.zeros[g]), 8))
    return Tensor(name=qt.name, shape=qt.shape, data=tuple(out), dtype="fp32")



def quantization_error(before: Sequence[Tensor],
                       after: Sequence[Tensor]) -> dict[str, float]:
    b = list(before)
    a = list(after)
    if len(b) != len(a):
        raise TensorError(f"{len(b)} tensors against {len(a)}")
    num = 0.0
    den = 0.0
    worst = 0.0
    for x, y in zip(b, a):
        if len(x.data) != len(y.data):
            raise TensorError(
                f"{x.name}: {len(x.data)} values against {len(y.data)}"
            )
        for u, v in zip(x.data, y.data):
            d = u - v
            num += d * d
            den += u * u
            worst = max(worst, abs(d))
    frob = math.sqrt(num)
    return {
        "frobenius": round(frob, 8),
        "relative": round(frob / math.sqrt(den), 8) if den > 0.0 else 0.0,
        "max_abs": round(worst, 8),
    }



def bytes_stored(tensors: Sequence[Tensor], precision: str = "fp32",
                 granularity: str = "per_tensor") -> int:
    if precision not in PRECISIONS:
        raise TensorError(
            f"unknown precision {precision!r}; known: {list(PRECISIONS)}"
        )
    if granularity not in GRANULARITIES:
        raise TensorError(
            f"unknown granularity {granularity!r}; known: {list(GRANULARITIES)}"
        )
    per = dtype_bytes(precision)
    total = 0.0
    for t in tensors:
        total += t.parameters * per
        if precision == "int8":
            groups = t.channels if granularity == "per_channel" else 1
            total += groups * (SCALE_BYTES + ZERO_BYTES)
    return int(total)


def effective_bits(tensors: Sequence[Tensor], precision: str = "fp32",
                   granularity: str = "per_tensor") -> float:
    n = total_parameters(tensors)
    if n <= 0:
        raise TensorError("a model with no parameters has no bits per parameter")
    return round(bytes_stored(tensors, precision, granularity) * 8.0 / n, 4)


def precision_row(model_name: str, config: Mapping[str, Any],
                  tensors: Sequence[Tensor],
                  calibration_values: Sequence[float],
                  all_values: Sequence[float]) -> dict[str, Any]:
    label = str(config.get("label") or config.get("precision") or "")
    precision = str(config.get("precision", "fp32"))
    if precision not in PRECISIONS:
        raise TensorError(
            f"unknown precision {precision!r}; known: {list(PRECISIONS)}"
        )
    granularity = str(config.get("granularity", "per_tensor"))
    symmetric = bool(config.get("symmetric", True))
    clip = str(config.get("clip", "minmax"))
    percentile = float(config.get("percentile", DEFAULT_PERCENTILE))
    calibration = str(config.get("calibration", "all"))
    if not label:
        raise TensorError("a configuration with no label cannot be plotted")

    if precision == "int8":
        qs = [quantize_tensor(t, DEFAULT_BITS, granularity, symmetric, clip,
                              percentile) for t in tensors]
        after = [dequantize(q) for q in qs]
        scale_count = sum(q.scale_count for q in qs)
    elif precision == "fp16":
        after = [to_fp16(t) for t in tensors]
        scale_count = 0
    else:
        after = list(tensors)
        scale_count = 0

    acts = as_tensor("activations", all_values)
    if precision == "int8":
        rlo, rhi = _range_of(calibration_values, clip, percentile, DEFAULT_BITS)
        qa = quantize_tensor(acts, DEFAULT_BITS, "per_tensor", False, clip,
                             percentile, given_range=(rlo, rhi))
        act_after = dequantize(qa)
        clipped = qa.clipped_fraction
    else:
        rlo, rhi = min(acts.data), max(acts.data)
        act_after = to_fp16(acts) if precision == "fp16" else acts
        clipped = 0.0

    return {
        "model": str(model_name),
        "label": label,
        "precision": precision,
        "granularity": granularity if precision == "int8" else "n/a",
        "symmetric": symmetric if precision == "int8" else None,
        "calibration": calibration if precision == "int8" else "not calibrated",
        "clip": clip if precision == "int8" else "n/a",
        "parameters": total_parameters(tensors),
        "weight_error": quantization_error(tensors, after)["relative"],
        "activation_error": quantization_error([acts], [act_after])["relative"],
        "clipped_fraction": clipped,
        "activation_range": [round(float(rlo), 8), round(float(rhi), 8)],
        "scale_count": scale_count,
        "bytes_stored": bytes_stored(tensors, precision, granularity),
        "effective_bits": effective_bits(tensors, precision, granularity),
    }


def pareto_front(points: Sequence[Mapping[str, Any]],
                 objectives: Mapping[str, float]) -> tuple[int, ...]:
    pts = list(points)
    if not pts:
        raise TensorError("no points to take a front over")
    if not objectives:
        raise TensorError("a front with no objectives is every point")
    for name in objectives:
        for k, p in enumerate(pts):
            if p.get(name) is None:
                raise TensorError(f"point {k} has no {name!r} to compare on")
    front: list[int] = []
    for i, a in enumerate(pts):
        beaten = False
        for j, b in enumerate(pts):
            if i == j:
                continue
            better = False
            worse = False
            for name, sense in objectives.items():
                d = (float(b[name]) - float(a[name])) * float(sense)
                if d > 0.0:
                    better = True
                elif d < 0.0:
                    worse = True
            if better and not worse:
                beaten = True
                break
        if not beaten:
            front.append(i)
    return tuple(front)


if __name__ == "__main__":
    json_path = Path("model.json")

    model_spec = load_model(json_path)
    model_tensors = model_spec["tensors"]
    model_name = model_spec["name"]
    total_params = total_parameters(model_tensors)

    test_tensor = model_tensors[0]  # conv1: shape (20, 3, 3, 3)

    raw_samples = materialise_samples(layer="conv3", width=40, count=32, seed=model_spec["seed"] + 100)
    all_act_vals = tuple(v for s in raw_samples for v in s.data)
    calib_all = select_calibration(raw_samples, policy="all")
    calib_one_cls = select_calibration(raw_samples, policy="one_class")

    # -----------------------------------------------------------------
    # 1. Test quantize_tensor
    # -----------------------------------------------------------------
    print("\n[1/7] Testing quantize_tensor...")
    q_pt = quantize_tensor(test_tensor, bits=8, granularity="per_tensor", symmetric=True)
    q_pc = quantize_tensor(test_tensor, bits=8, granularity="per_channel", symmetric=True)

    assert len(q_pt.codes) == test_tensor.parameters
    assert q_pt.scale_count == 1
    assert q_pc.scale_count == test_tensor.channels
    print(f"  ✓ Per-tensor quantized:  {len(q_pt.codes)} codes, {q_pt.scale_count} scale")
    print(f"  ✓ Per-channel quantized: {len(q_pc.codes)} codes, {q_pc.scale_count} scales")

    # -----------------------------------------------------------------
    # 2. Test dequantize
    # -----------------------------------------------------------------
    print("\n[2/7] Testing dequantize...")
    deq_pt = dequantize(q_pt)
    deq_pc = dequantize(q_pc)

    assert deq_pt.shape == test_tensor.shape
    assert len(deq_pt.data) == len(test_tensor.data)
    assert deq_pc.shape == test_tensor.shape
    print(f"  ✓ Reconstructed tensor shape: {deq_pt.shape}, dtype: {deq_pt.dtype}")

    # -----------------------------------------------------------------
    # 3. Test quantization_error
    # -----------------------------------------------------------------
    print("\n[3/7] Testing quantization_error...")
    err_pt = quantization_error([test_tensor], [deq_pt])
    err_pc = quantization_error([test_tensor], [deq_pc])

    assert err_pt["relative"] >= 0.0 and err_pc["relative"] >= 0.0
    print(f"  ✓ Per-tensor relative error:  {err_pt['relative']:.6f} (max_abs: {err_pt['max_abs']:.6f})")
    print(f"  ✓ Per-channel relative error: {err_pc['relative']:.6f} (max_abs: {err_pc['max_abs']:.6f})")

    # -----------------------------------------------------------------
    # 4. Test bytes_stored
    # -----------------------------------------------------------------
    print("\n[4/7] Testing bytes_stored...")
    bytes_fp32 = bytes_stored(model_tensors, precision="fp32")
    bytes_fp16 = bytes_stored(model_tensors, precision="fp16")
    bytes_int8_pt = bytes_stored(model_tensors, precision="int8", granularity="per_tensor")
    bytes_int8_pc = bytes_stored(model_tensors, precision="int8", granularity="per_channel")

    assert bytes_fp32 > bytes_fp16 > bytes_int8_pt
    assert bytes_int8_pc >= bytes_int8_pt  # per-channel carries scale/zero per output channel
    print(f"  ✓ FP32 bytes:            {bytes_fp32} B")
    print(f"  ✓ FP16 bytes:            {bytes_fp16} B")
    print(f"  ✓ INT8 per-tensor bytes: {bytes_int8_pt} B")
    print(f"  ✓ INT8 per-channel bytes:{bytes_int8_pc} B")

    # -----------------------------------------------------------------
    # 5. Test effective_bits
    # -----------------------------------------------------------------
    print("\n[5/7] Testing effective_bits...")
    eff_fp32 = effective_bits(model_tensors, precision="fp32")
    eff_fp16 = effective_bits(model_tensors, precision="fp16")
    eff_int8_pt = effective_bits(model_tensors, precision="int8", granularity="per_tensor")
    eff_int8_pc = effective_bits(model_tensors, precision="int8", granularity="per_channel")

    assert eff_fp32 == 32.0
    assert eff_fp16 == 16.0
    assert eff_int8_pc > 8.0  # accounts for scales and zeros overhead
    print(f"  ✓ FP32 effective bits:            {eff_fp32:.4f}")
    print(f"  ✓ FP16 effective bits:            {eff_fp16:.4f}")
    print(f"  ✓ INT8 per-tensor effective bits: {eff_int8_pt:.4f}")
    print(f"  ✓ INT8 per-channel effective bits:{eff_int8_pc:.4f}")

    # -----------------------------------------------------------------
    # 6. Test precision_row
    # -----------------------------------------------------------------
    print("\n[6/7] Testing precision_row...")
    configs = [
        {"label": "FP32 Baseline", "precision": "fp32"},
        {"label": "FP16", "precision": "fp16"},
        {
            "label": "INT8 Per-Tensor (All)",
            "precision": "int8",
            "granularity": "per_tensor",
            "symmetric": False,
            "clip": "minmax",
            "calibration": "all",
        },
        {
            "label": "INT8 Per-Channel (All)",
            "precision": "int8",
            "granularity": "per_channel",
            "symmetric": True,
            "clip": "minmax",
            "calibration": "all",
        },
        {
            "label": "INT8 Per-Channel (Class 0 Only)",
            "precision": "int8",
            "granularity": "per_channel",
            "symmetric": True,
            "clip": "minmax",
            "calibration": "one_class",
        },
    ]

    rows = []
    for cfg in configs:
        calib_vals = calib_one_cls if cfg.get("calibration") == "one_class" else calib_all
        row = precision_row(
            model_name=model_name,
            config=cfg,
            tensors=model_tensors,
            calibration_values=calib_vals,
            all_values=all_act_vals,
        )
        rows.append(row)
        print(f"  ✓ {row['label']:<32} | W_Err: {row['weight_error']:<8.6f} | Act_Err: {row['activation_error']:<8.6f} | Bits: {row['effective_bits']:<6.3f}")

    # -----------------------------------------------------------------
    # 7. Test pareto_front & Save Output
    # -----------------------------------------------------------------
    print("\n[7/7] Testing pareto_front...")
    # Minimize effective_bits and minimize weight_error (sense = -1.0)
    objectives = {
        "effective_bits": -1.0,
        "weight_error": -1.0,
    }
    front_indices = pareto_front(rows, objectives)

    assert len(front_indices) > 0
    print(f"  ✓ Pareto optimal candidate indices: {front_indices}")
    for idx in front_indices:
        p = rows[idx]
        print(f"    * Index {idx}: {p['label']} (Bits: {p['effective_bits']}, Weight Err: {p['weight_error']})")

    # Output to JSON
    report_path = Path("quantization_test_results.json")
    results = {
        "model": model_name,
        "total_parameters": total_params,
        "objectives": objectives,
        "pareto_front_indices": list(front_indices),
        "configurations": rows,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Saved] Full output written to {report_path.resolve()}")


