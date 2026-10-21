from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0"
LAB = "lab07"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_UNKNOWN = 2
EXIT_MALFORMED = 3

REQUIRED = (
    "schema_version",
    "lab",
    "generated",
    "camera",
    "conditions",
    "runs",
    "findings",
    "verdicts",
    "provenance",
    "notes",
)

REQUIRED_CAMERA = ("sensor", "width", "height", "framerate", "modes", "source")

# What a run must carry before it is a run. `pipeline` is the description,
# `bytes_per_frame` is arithmetic over that description, and `frames` and
# `intervals` are the two things only a board can supply -- which is why both
# are allowed to be null and neither is allowed to be absent.
REQUIRED_RUN = (
    "label",
    "pipeline",
    "stages",
    "bytes_per_frame",
    "frames",
    "intervals",
    "scene",
)

# The conditions a Lab 07 run has to state before its numbers mean anything.
# `scene` is the new one and it is last on purpose: it is the field the shared
# harness does not have, and a reader scanning this tuple should hit the four
# familiar ones first and then notice a fifth.
FIXED_CONDITIONS = ("power_mode", "sensor_mode", "warmup_frames", "duration_s",
                    "scene")

VALID_STATUS = ("computed", "measured", "unknown")
VALID_RESULT = ("pass", "fail", "unknown")

VERDICTS = (
    "frames_are_not_black",
    "moving_the_work_moved_the_bytes",
    "moving_the_work_moved_the_rate",
    "rate_holds_its_shape",
    "no_frames_vanished",
    "both_runs_saw_the_same_scene",
)

# Which verdicts cannot be settled without a board and a camera. Named here
# rather than inferred, so `check_report.py` and `report.py` cannot drift apart
# about it. Note that `moving_the_work_moved_the_bytes` is NOT in this list:
# the byte count is arithmetic over a pipeline description, it is settled on a
# laptop on the bus, and Stage B asks for the prediction before Stage C is run.
NEEDS_A_BOARD = (
    "frames_are_not_black",
    "moving_the_work_moved_the_rate",
    "rate_holds_its_shape",
    "no_frames_vanished",
    "both_runs_saw_the_same_scene",
)

# How much fewer bytes the moved pipeline has to shift before
# `moving_the_work_moved_the_bytes` counts it as having moved anything. Five per
# cent. The arithmetic on Lecture 07 slides 29 and 37 comes out at 3.75x, so a
# correct implementation clears this by a factor of fifteen; the margin exists
# so that a student who moves one small stage and nothing else gets a pass with
# a number next to it rather than a pass on rounding.
BYTES_MARGIN = 0.05

# How much faster the moved pipeline has to run before
# `moving_the_work_moved_the_rate` passes. Five per cent, which is Lab 03's
# threshold and is here for Lab 03's reason: two rates a few per cent apart on
# a warm board are the same rate. It is deliberately the same number as
# BYTES_MARGIN and deliberately not the same test -- one is arithmetic and one
# is a measurement, and watching them disagree is Stage D's paragraph.
RATE_MARGIN = 0.05

# How far the 95th percentile inter-frame interval may sit above the median
# before `rate_holds_its_shape` calls the stream bursty. Fifty per cent: a
# stream at a steady 30 fps has p50 and p95 within a millisecond or two of
# 33.33, and a stream that stalls for one frame time shows p95 at roughly twice
# p50. Halfway between those is a threshold that a real jitter pattern crosses
# and float noise does not.
SHAPE_MARGIN = 0.50

# How far two runs' frame statistics may differ before
# `both_runs_saw_the_same_scene` stops believing they saw the same thing. Ten
# per cent of full scale on the mean, which is wide enough to survive the 3A
# loops re-converging between two runs and narrow enough that a light switched
# on, a lens capped, or a camera pointed somewhere else does not get through.
#
# There is no principled derivation of 0.10 available to us. It is a stated
# convention, it is on the face of the report, and Stage D is where a student
# argues with it. That is better than a threshold nobody can see.
SCENE_MARGIN = 0.10

# A frame is black when nothing in it is above this, on a 0-1 scale. Not zero.
# Lecture 07 slide 41: a real scene in a dark room has a maximum well above
# zero, and a sensor returning literal zeros everywhere is returning no signal
# rather than a dark picture. One part in 255 is the smallest thing an 8-bit
# frame can say, so anything at or below it is silence.
BLACK_MAX = 1.0 / 255.0

# And a frame is *suspicious* when it has signal but no structure: a maximum
# above BLACK_MAX and a spread below this. A uniformly grey frame at 0.5 is not
# a picture of anything either, and it is what a sensor stuck in a test-pattern
# mode or an ISP with no statistics will hand you.
FLAT_SPREAD = 0.02


class ReportError(ValueError):
    """A report that cannot be validated. `capture.py` catches it, exits 3."""


# ---------------------------------------------------------------------------
# findings
# ---------------------------------------------------------------------------

def computed(value: Any, source: str, **extra: Any) -> dict[str, Any]:
    """A number that came out of arithmetic over a pipeline description."""
    return {"value": value, "source": source, "status": "computed", **extra}


def measured(value: Any, source: str, **extra: Any) -> dict[str, Any]:
    """A number that came off a board, under stated conditions."""
    return {"value": value, "source": source, "status": "measured", **extra}


def unknown(source: str, detail: str, **extra: Any) -> dict[str, Any]:
    """No value, a reason, and the same shape as the other two.

    The shape matters. A caller that has to check whether a key exists before it
    can read a status will eventually forget, and the failure mode is a report
    that looks complete.
    """
    return {
        "value": None,
        "source": source,
        "status": "unknown",
        "detail": detail,
        **extra,
    }


def is_answered(finding: Any) -> bool:
    """True when a finding carries a value somebody established."""
    return (
        isinstance(finding, dict)
        and finding.get("status") in ("computed", "measured")
        and finding.get("value") is not None
    )


def verdict(result: str, detail: str, source: str) -> dict[str, str]:
    """One of pass, fail, unknown, with a sentence and an attribution."""
    if result not in VALID_RESULT:
        raise ReportError(f"verdict result {result!r} not in {VALID_RESULT}")
    return {"result": result, "detail": detail, "source": source}


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def validate(report: Any) -> None:
    """Raise ReportError unless `report` is one. Returns nothing on success."""
    if not isinstance(report, dict):
        raise ReportError(f"top level is {type(report).__name__}, want object")
    for key in REQUIRED:
        if key not in report:
            raise ReportError(f"no {key!r}")
    if report["schema_version"] != SCHEMA_VERSION:
        raise ReportError(
            f"schema_version {report['schema_version']!r}, "
            f"this package speaks {SCHEMA_VERSION!r}"
        )
    if report["lab"] != LAB:
        raise ReportError(f"lab {report['lab']!r}, want {LAB!r}")

    cam = report["camera"]
    if not isinstance(cam, dict):
        raise ReportError("'camera' is not an object")
    for key in REQUIRED_CAMERA:
        if key not in cam:
            raise ReportError(f"camera has no {key!r}")
    # A mode list is how Stage A proves the thing on the ribbon is a sensor and
    # not a device node. An empty list is a legitimate answer -- it means the
    # enumeration found nothing -- but a missing one means nobody looked.
    if not isinstance(cam["modes"], list):
        raise ReportError("camera.modes is not a list")

    runs = report["runs"]
    if not isinstance(runs, list) or not runs:
        raise ReportError("'runs' must be a non-empty list")
    seen: set[str] = set()
    for i, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ReportError(f"runs[{i}] is not an object")
        for key in REQUIRED_RUN:
            if key not in run:
                raise ReportError(f"runs[{i}] has no {key!r}")
        # Labels are how a measured record finds its run and how the plot names
        # its series. Two runs with one label is not a duplicate row, it is a
        # report in which a measurement cannot be attributed.
        if run["label"] in seen:
            raise ReportError(f"runs[{i}]: two runs labelled {run['label']!r}")
        seen.add(run["label"])
        if not isinstance(run["stages"], list) or not run["stages"]:
            raise ReportError(
                f"runs[{i}]: 'stages' is empty. A pipeline with no stages is "
                "not a cheap pipeline, it is a pipeline nobody described."
            )
        if not isinstance(run["bytes_per_frame"], int):
            raise ReportError(
                f"runs[{i}]: bytes_per_frame is "
                f"{type(run['bytes_per_frame']).__name__}, want int. Bytes "
                "moved is a count of bytes and there is no such thing as "
                "half of one."
            )

    verdicts = report["verdicts"]
    if not isinstance(verdicts, dict):
        raise ReportError("'verdicts' is not an object")
    missing = [v for v in VERDICTS if v not in verdicts]
    if missing:
        raise ReportError(f"verdicts missing: {missing}")
    for name, v in verdicts.items():
        if not isinstance(v, dict) or v.get("result") not in VALID_RESULT:
            raise ReportError(f"verdict {name!r} has no valid result")

    for name, f in (report["findings"] or {}).items():
        if isinstance(f, dict) and "status" in f:
            if f["status"] not in VALID_STATUS:
                raise ReportError(f"finding {name!r} has status {f['status']!r}")


def exit_code(report: dict[str, Any]) -> int:
    """Which of the four the verdicts add up to.

    Fail dominates unknown. A report with one failure and four unknowns is a
    report with a failure in it, and the exit code that hides the failure behind
    the unknowns is the one that lets somebody ship.
    """
    results = [v.get("result") for v in report.get("verdicts", {}).values()]
    if "fail" in results:
        return EXIT_FAILED
    if "unknown" in results:
        return EXIT_UNKNOWN
    return EXIT_OK