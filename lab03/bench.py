from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


class Workload(Protocol):
    """One unit of the work being measured."""

    def run(self) -> Any:
        """Do the work once. May return before the work has finished."""

    def synchronize(self) -> None:
        """Block until everything `run` has started has finished."""


@dataclass(frozen=True)
class CommandResult:
    """What a command said. `ok` is false when it could not be run at all."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    ok: bool = True
    error: str = ""

    @property
    def source(self) -> str:
        return " ".join(self.argv)


def real_runner(argv: list[str], timeout: float = 5.0) -> CommandResult:
    """Run a command for real, converting every failure into a result.

    Nothing here raises. A probe's job is to come back with a finding, and
    "the command is not installed" is a finding — a considerably more useful one
    than a traceback, because it names what to go and fix.
    """
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError) as e:
        return CommandResult(tuple(argv), -1, "", ok=False, error=str(e))
    return CommandResult(tuple(argv), proc.returncode, proc.stdout, ok=True)


@dataclass(frozen=True)
class Bench:
    """A measurement rig, real or fabricated, as far as any probe can tell."""

    workload: Workload
    clock: Callable[[], int] = lambda: 0
    telemetry: Path = Path("/")
    runner: Callable[[list[str]], CommandResult] = real_runner
    device: str = "cpu"

    # Everything the record has to state about *what* was measured. Carried on
    # the rig rather than passed around separately because the pairing is the
    # point of the lab: a number and its conditions travel together or the
    # number is not a measurement.
    model: str = "unknown"
    precision: str = "fp32"
    input_shape: tuple[int, ...] = (1, 3, 224, 224)
    batch_size: int = 1

    # Two of the schema's fields that are properties of the rig rather than of
    # the run. `model_size_mb` is known before anything is measured;
    # `gpu_memory_mb` is a callable rather than a value because the number worth
    # recording is the peak, and the peak is not known until the run is over.
    model_size_mb: float | None = None
    gpu_memory_mb: Callable[[], float | None] = lambda: None

    @classmethod
    def real(
        cls,
        device: str = "cuda",
        model: str = "mobilenet_v3_small",
        batch_size: int = 1,
    ) -> Bench:
        """This machine, right now, with a real model on a real processor.

        torch and torchvision are imported here rather than at module level, and
        that is not tidiness. The test suite imports this module on machines
        with no torch on them at all; a top-level import would make the entire
        lab ungradeable anywhere but a Jetson, which is the failure Lab 02 spent
        a session on.
        """
        import time

        import torch  # noqa: PLC0415
        import torchvision  # noqa: PLC0415

        if device == "cuda" and not torch.cuda.is_available():
            raise SystemExit(
                "asked for --device cuda and torch.cuda.is_available() is False.\n"
                "Run Lab 02's verify_environment.py before assuming this is a "
                "Lab 03 problem. It is almost certainly the wheel."
            )

        net = getattr(torchvision.models, model)(weights=None).eval().to(device)
        shape = (batch_size, 3, 224, 224)

        # The input is created on the device *before* the loop, deliberately.
        # That is what makes the default timing boundary `kernel` rather than
        # `kernel_plus_copies`: no host-to-device copy is inside the stopwatch.
        # Move this line into the workload and you have measured a different
        # quantity, which is fine as long as the record says so.
        x = torch.randn(*shape, device=device)

        size_mb = (
            sum(p.numel() * p.element_size() for p in net.parameters()) / 1_000_000.0
        )

        def peak_gpu_mb() -> float | None:
            if device != "cuda":
                return None
            return torch.cuda.max_memory_allocated() / 1_000_000.0

        return cls(
            workload=TorchWorkload(net, x, device),
            clock=time.perf_counter_ns,
            telemetry=Path("/"),
            runner=real_runner,
            device=device,
            model=model,
            precision="fp32",
            input_shape=shape,
            batch_size=batch_size,
            model_size_mb=round(size_mb, 3),
            gpu_memory_mb=peak_gpu_mb,
        )


class TorchWorkload:
    """One forward pass, and the barrier that makes timing it meaningful."""

    def __init__(self, net: Any, x: Any, device: str) -> None:
        self._net, self._x, self._device = net, x, device

    def run(self) -> Any:
        import torch  # noqa: PLC0415

        with torch.inference_mode():
            return self._net(self._x)

    def synchronize(self) -> None:
        if self._device != "cuda":
            # On CPU the call is already synchronous, so there is nothing to
            # wait for. Returning quietly rather than raising matters: the same
            # harness has to time both processors, and a barrier that only
            # exists on one of them would make the two runs incomparable.
            return
        import torch  # noqa: PLC0415

        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Helpers. Given to students; the exercise is in `measure.py`.
# ---------------------------------------------------------------------------


def read_text(root: Path, rel: str) -> str | None:
    """Read `root/rel`, returning None if it is missing or unreadable."""
    p = Path(root) / rel.lstrip("/")
    try:
        return p.read_text(errors="replace").strip("\x00").strip()
    except (OSError, TypeError, UnicodeDecodeError):
        return None


def read_first(root: Path, candidates: tuple[str, ...]) -> tuple[str, str] | None:
    """Try several paths, return `(path, contents)` for the first that reads.

    Sysfs node names move between L4T releases and between carrier boards, and
    a probe pinned to one path reports a healthy board as unmeasurable the first
    time NVIDIA renames a directory. Returning the path that worked, rather than
    just the value, is what makes the finding's `source` honest — it names where
    the number actually came from on *this* board and not where the author
    hoped it would be.
    """
    for rel in candidates:
        text = read_text(root, rel)
        if text:
            return rel, text
    return None


def unknown(source: str, why: str) -> dict[str, Any]:
    """What a probe returns when it cannot determine something.

    Same contract as Labs 01 and 02, deliberately and for the third time. Not
    None threaded through the record, not a plausible default, not a zero: an
    explicit record that the probe ran, failed, and knows why. It is marked
    positively and it is graded positively.

    A zero is the dangerous one here in a way it was not in the earlier labs.
    `"power_mw": 0` is a number, it validates against the schema, it plots, and
    it will be silently averaged into somebody's energy-per-inference figure in
    week fifteen.
    """
    return {"value": None, "source": source, "status": "unknown", "detail": why}


def measured(value: Any, source: str, **extra: Any) -> dict[str, Any]:
    """What a probe returns when it does know something."""
    return {"value": value, "source": source, "status": "ok", **extra}
