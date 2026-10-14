from __future__ import annotations

import math


import math
from dataclasses import dataclass
from typing import Sequence

from tensors import (
    Tensor,
    TensorError,
)

# The three the lab builds. `int4` exists in `tensors.DTYPE_BYTES` and is not
# here, because a four-bit build of a model this small is Lecture 06 slide 26's
# territory and not a thing this lab measures.
PRECISIONS = ("fp32", "fp16", "int8")

# Where a scale is allowed to live. Group quantization (UNC Lec 9 p. 31) is the
# next rung and is deliberately absent: it needs a group size argument, a second
# level of scales, and an afternoon this lab does not have.
GRANULARITIES = ("per_tensor", "per_channel")

# How the clip is chosen. Lecture 06 slide 35 draws all three.
CLIP_METHODS = ("minmax", "percentile", "mse")

# Which samples a calibration policy shows the calibrator. `select_calibration`
# below is provided complete; the policies are the lab's menu of ways to be
# wrong, and `one_class` is the one that is wrong on purpose.
CALIBRATIONS = ("all", "two_class", "first_8", "one_class")

DEFAULT_BITS = 8
DEFAULT_PERCENTILE = 99.9

# How many clip candidates the MSE search tries, evenly spaced from 1/32 of the
# observed range up to all of it. Thirty-two is enough to land within a per cent
# of the continuous optimum on this data and small enough that the search is
# instant. The real method (OCTAV, UNC Lec 9 p. 40) solves for the optimum with
# Newton-Raphson; a grid is the honest version of that for a lab, and the number
# is a named constant so a report can say which grid it used.
MSE_CANDIDATES = 32

def _round_half_up(x: float) -> int:
    """`x` to the nearest integer, halves going up, negatives included.

    Python's `round` is banker's rounding and gets this wrong in both
    directions: `round(0.5)` is 0 and `round(2.5)` is 2. `int(x + 0.5)`
    truncates towards zero and gets it wrong for negatives, which is where the
    zero point lives.
    """
    return int(math.floor(float(x) + 0.5))


def _qmin_qmax(bits: int, symmetric: bool) -> tuple[int, int]:
    """The signed integer range `bits` bits can hold under this convention.

    Asymmetric keeps every code: `-2^(b-1) .. 2^(b-1)-1`, which for eight bits
    is -128..127 and for two bits is -2..1. Symmetric gives up the most negative
    one so that the range is centred on zero: `-(2^(b-1)-1) .. 2^(b-1)-1`, which
    for eight bits is -127..127.
    """
    b = int(bits)
    if b < 2:
        raise TensorError(f"{b} bits cannot carry a range")
    if symmetric:
        return -(2 ** (b - 1)) + 1, 2 ** (b - 1) - 1
    return -(2 ** (b - 1)), 2 ** (b - 1) - 1


def _scale_zero(rmin: float, rmax: float, bits: int,
                symmetric: bool) -> tuple[float, int]:
    qmin, qmax = _qmin_qmax(bits, symmetric)
    lo, hi = float(min(rmin, rmax)), float(max(rmin, rmax))
    if symmetric:
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    else:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi == lo:
        return 1.0, 0
    s = (hi - lo) / (qmax - qmin)
    if symmetric:
        return s, 0
    z = _round_half_up(qmin - lo / s)
    return s, max(qmin, min(qmax, z))


def _channel_slice(t: Tensor, c: int) -> tuple[int, int]:
    if not 0 <= c < t.channels:
        raise TensorError(f"{t.name}: no channel {c} in {t.channels}")
    stride = t.channel_stride
    return c * stride, (c + 1) * stride


def _range_of(values: Sequence[float], method: str = "minmax",
              percentile: float = DEFAULT_PERCENTILE,
              bits: int = DEFAULT_BITS) -> tuple[float, float]:
    vals = [float(v) for v in values]
    if not vals:
        raise TensorError("no values to take a range over")
    if method not in CLIP_METHODS:
        raise TensorError(f"unknown clip method {method!r}; known: {list(CLIP_METHODS)}")
    if method == "minmax":
        return min(vals), max(vals)
    if method == "percentile":
        if not 0.0 < float(percentile) <= 100.0:
            raise TensorError(f"percentile {percentile} is not in (0, 100]")
        order = sorted(vals)
        n = len(order)
        lo_i = min(n - 1, int(math.floor((100.0 - float(percentile)) / 200.0 * n)))
        hi_i = max(0, n - 1 - lo_i)
        return order[min(lo_i, hi_i)], order[max(lo_i, hi_i)]
    lo0, hi0 = min(vals), max(vals)
    qmin, qmax = _qmin_qmax(bits, False)
    levels = float(qmax - qmin)
    best: tuple[float, float, float] | None = None
    for k in range(1, MSE_CANDIDATES + 1):
        f = float(k) / MSE_CANDIDATES
        lo, hi = lo0 * f, hi0 * f
        if hi <= lo:
            continue
        step = (hi - lo) / levels
        err = sum((v - min(hi, max(lo, v))) ** 2 for v in vals)
        err += len(vals) * step * step / 12.0
        if best is None or err < best[0]:
            best = (err, lo, hi)
    if best is None:
        raise TensorError("the MSE search found no usable clip")
    return best[1], best[2]