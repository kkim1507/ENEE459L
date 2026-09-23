from __future__ import annotations

import statistics
from typing import Any

from bench import Bench, measured, read_first, read_text, unknown

import json

# A sample is still warm-up while it exceeds the settled rate by this fraction.
WARMUP_TOL = 0.5

# How many samples must sit strictly above a quantile before that quantile is an
# estimate rather than "the biggest number we saw, wearing a hat".
MIN_SAMPLES_ABOVE = 5

# Percentiles the record carries, in the order the schema lists them.
PERCENTILES = (50, 95, 99)

# The widest gap between neighbouring measurements, as a multiple of the typical
# gap, beyond which the sample is treated as coming from two populations.
MULTIMODAL_GAP_RATIO = 20.0

# Neither side of that gap is a mode unless it holds at least this fraction.
MIN_MODE_FRACTION = 0.10

# Below this many retained samples, modality is not a question worth answering.
MIN_SAMPLES_FOR_MODALITY = 20

# How far the last third of a run may drift from the first third, relative to
# the run's own median, before the run is not one population either.
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12

THERMAL_ZONES = "sys/devices/virtual/thermal"

POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)

GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
)

CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"


# ===========================================================================
# 1. The loop
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    bench.workload.synchronize()
    samples = []

    for _ in range(repeats):
        start = bench.clock()
        bench.workload.run()
        bench.workload.synchronize()
        end = bench.clock()
        samples.append((end - start) / 1_000_000.0)

    return samples


def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    if len(samples) < 4:
        return unknown(
            "second half median of samples",
            "too few samples to identify a settled rate",
        )

    settled_rate = statistics.median(samples[len(samples) // 2 :])
    if settled_rate <= 0:
        return unknown(
            "second half median of samples",
            "settled median must be greater than zero",
        )

    threshold = settled_rate * (1 + WARMUP_TOL)
    discarded = 0
    for sample in samples:
        if sample <= threshold:
            break
        discarded += 1

    return measured(
        discarded,
        "leading prefix above (1 + 0.5) x median of the run's second half",
        settled_rate_ms=round(settled_rate, 4),
        threshold_ms=round(threshold, 4),
        tolerance=WARMUP_TOL,
        retained=len(samples) - discarded,
    )



def summarize(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }

    ordered = sorted(samples)
    n = len(ordered)

    def percentile(percentile: int) -> float:
        h = (n - 1) * percentile / 100
        index = int(h)
        if index == n - 1:
            return ordered[index]
        fraction = h - index
        return ordered[index] + fraction * (ordered[index + 1] - ordered[index])

    return {
        "n": n,
        "mean": round(statistics.fmean(ordered), 4),
        "std": round(statistics.stdev(ordered), 4) if n > 2 else 0.0,
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
        "p50": round(percentile(50), 4),
        "p95": round(percentile(95), 4),
        "p99": round(percentile(99), 4),
    }

def is_multimodal(samples: list[float]) -> dict[str, Any]:
    sample_count = len(samples)
    if sample_count < MIN_SAMPLES_FOR_MODALITY:
        return unknown(
            "widest trimmed gap / median trimmed gap",
            "not enough samples to assess modality",
        )

    ordered = sorted(samples)
    trim_count = int(sample_count * 0.05)
    trimmed = ordered[trim_count : sample_count - trim_count]
    gaps = [right - left for left, right in zip(trimmed, trimmed[1:])]
    typical_gap = statistics.median(gaps)
    if typical_gap <= 0:
        return unknown(
            "widest trimmed gap / median trimmed gap",
            "timer resolution is too coarse to distinguish adjacent samples",
        )

    widest_gap_index = max(range(len(gaps)), key=gaps.__getitem__)
    widest_gap = gaps[widest_gap_index]
    gap_ratio = widest_gap / typical_gap

    split_index = trim_count + widest_gap_index
    left_samples = ordered[: split_index + 1]
    right_samples = ordered[split_index + 1 :]
    left_count = len(left_samples)
    right_count = len(right_samples)
    modes = [
        {
            "n": left_count,
            "share": round(left_count / sample_count, 4),
            "median_ms": round(statistics.median(left_samples), 4),
        },
        {
            "n": right_count,
            "share": round(right_count / sample_count, 4),
            "median_ms": round(statistics.median(right_samples), 4),
        },
    ]

    return measured(
        gap_ratio >= MULTIMODAL_GAP_RATIO
        and left_count >= sample_count * MIN_MODE_FRACTION
        and right_count >= sample_count * MIN_MODE_FRACTION,
        "widest trimmed gap >= 20.0x the median gap, with >= 10% of samples on each side",
        gap_ratio=round(gap_ratio, 4),
        widest_gap_ms=round(widest_gap, 4),
        typical_gap_ms=round(typical_gap, 4),
        modes=modes,
    )

# ===========================================================================
# 7. The clock ceiling the run happened under
# ===========================================================================


def probe_power_state(bench: Bench) -> dict[str, Any]:
    result = bench.runner(["nvpmodel", "-q"])
    if not result.ok or result.returncode != 0:
        detail = result.error or f"command exited with status {result.returncode}"
        return unknown("nvpmodel -q", f"could not query power mode: {detail}")

    lines = [line.strip() for line in result.stdout.splitlines()]
    mode_line_index = next(
        (index for index, line in enumerate(lines) if "NV Power Mode:" in line),
        None,
    )
    if mode_line_index is None or mode_line_index + 1 >= len(lines):
        return unknown("nvpmodel -q", "power mode name or mode ID was not found")

    mode_name = lines[mode_line_index].split("NV Power Mode:", 1)[1].strip()
    try:
        mode_index = int(lines[mode_line_index + 1])
    except ValueError:
        return unknown("nvpmodel -q", "power mode ID was not an integer")
    if not mode_name:
        return unknown("nvpmodel -q", "power mode name was empty")

    minimum = read_text(bench.telemetry, CPUFREQ_MIN)
    maximum = read_text(bench.telemetry, CPUFREQ_MAX)
    clock_source = f"{CPUFREQ_MIN} vs {CPUFREQ_MAX}"
    if minimum is None or maximum is None:
        clocks = unknown(clock_source, "one or both CPU frequency limits could not be read")
        jetson_clocks = None
    else:
        minimum = minimum.strip()
        maximum = maximum.strip()
        clocks = measured(
            f"scaling_min_freq={minimum}, scaling_max_freq={maximum}",
            clock_source,
        )
        jetson_clocks = minimum == maximum

    return measured(
        mode_name,
        "nvpmodel -q",
        mode_index=mode_index,
        jetson_clocks=jetson_clocks,
        jetson_clocks_source=clocks,
    )



def probe_telemetry(bench: Bench) -> dict[str, Any]:
    thermal_root = bench.telemetry / THERMAL_ZONES
    temperatures = []
    zones_read = 0
    for zone_dir in sorted(thermal_root.glob("thermal_zone*/")):
        raw_temperature = read_text(
            bench.telemetry,
            f"{THERMAL_ZONES}/{zone_dir.name}/temp",
        )
        if raw_temperature is None:
            continue
        try:
            millidegrees = int(raw_temperature)
        except ValueError:
            continue
        if millidegrees <= -1000:
            continue

        temperature = millidegrees / 1000.0
        zone_name = read_text(
            bench.telemetry,
            f"{THERMAL_ZONES}/{zone_dir.name}/type",
        ) or zone_dir.name
        temperatures.append((temperature, zone_name))
        zones_read += 1

    if temperatures:
        highest_temperature, hottest_zone = max(temperatures, key=lambda item: item[0])
        temperature_record = measured(
            round(highest_temperature, 4),
            f"{THERMAL_ZONES}/*/temp",
            zone=hottest_zone,
            zones_read=zones_read,
        )
    else:
        temperature_record = unknown(
            f"{THERMAL_ZONES}/*/temp",
            "no valid thermal zone temperatures could be read",
        )

    power_source = read_first(bench.telemetry, POWER_RAIL_CANDIDATES)
    if power_source is None:
        power_record = unknown(
            " | ".join(POWER_RAIL_CANDIDATES),
            "none of the documented INA3221 rail paths could be read",
        )
    else:
        power_path, raw_power = power_source
        try:
            power_record = measured(int(raw_power), power_path)
        except ValueError:
            power_record = unknown(power_path, "INA3221 power value was not an integer")

    gpu_source = read_first(bench.telemetry, GPU_LOAD_CANDIDATES)
    if gpu_source is None:
        gpu_record = unknown(
            " | ".join(GPU_LOAD_CANDIDATES),
            "none of the documented GPU load paths could be read",
        )
    else:
        gpu_path, raw_load = gpu_source
        try:
            gpu_record = measured(
                round(int(raw_load) / 10.0, 4),
                gpu_path,
                units="per-mille / 10",
            )
        except ValueError:
            gpu_record = unknown(gpu_path, "GPU load value was not an integer")

    return {
        "temperature_c": temperature_record,
        "power_mw": power_record,
        "gpu_utilization_percent": gpu_record,
    }

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
    # env = Bench.real()
    # out = find_warmup_boundary(samples)
    # print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Bench.real()

    # get your samples
    samples = run_timed_iterations(env, repeats=100)

    # testing measurments and probes
    report = {
        "warmup_boundary": find_warmup_boundary(samples),
        "summarize_setup": summarize(samples),
        "is_multimodal": is_multimodal(samples),
        "probe_power_state": probe_power_state(env),
        "probe_telemetry": probe_telemetry(env),
    }

    # save samples
    path = "samples_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)

    # save report
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)