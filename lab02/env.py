from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


class ModuleNotAvailable(Exception):
    """Raised by an importer when a module cannot be imported.

    A distinct type rather than reusing ImportError, so a probe cannot catch
    this by accident while meaning to catch something raised *inside* the
    module it asked for. Those are different findings: "TensorRT is not
    installed" and "TensorRT is installed and blew up on import" send a student
    to entirely different places.
    """


def real_importer(name: str) -> Any:
    """Import for real, converting failure into `ModuleNotAvailable`."""
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise ModuleNotAvailable(str(e)) from e


@dataclass(frozen=True)
class PythonRuntime:
    """What the interpreter says about itself.

    `prefix` and `base_prefix` are the pair that answers "am I in a virtual
    environment": they differ inside one and are equal outside. The obvious
    alternative — reading the VIRTUAL_ENV environment variable — is wrong in
    exactly the cases that matter, because it is set by the activate script and
    not by the interpreter. A venv's python invoked by absolute path, by a
    Makefile, by cron, or under sudo is a real venv with VIRTUAL_ENV unset.
    """

    version: tuple[int, int, int] = (0, 0, 0)
    executable: str = ""
    prefix: str = ""
    base_prefix: str = ""

    @classmethod
    def current(cls) -> PythonRuntime:
        return cls(
            version=tuple(sys.version_info[:3]),  # type: ignore[arg-type]
            executable=sys.executable,
            prefix=sys.prefix,
            base_prefix=sys.base_prefix,
        )


@dataclass(frozen=True)
class Env:
    """A machine, real or fabricated, as far as any probe can tell."""

    root: Path = Path("/")
    importer: Callable[[str], Any] = real_importer
    python: PythonRuntime = field(default_factory=PythonRuntime)

    @classmethod
    def real(cls) -> Env:
        """This machine, right now."""
        return cls(root=Path("/"), importer=real_importer, python=PythonRuntime.current())


# ---------------------------------------------------------------------------
# Helpers. Given to students; the exercise is the probes.
# ---------------------------------------------------------------------------
def read_text(root: Path, rel: str) -> str | None:
    """Read `root/rel`, returning None if it is missing or unreadable."""
    p = Path(root) / rel.lstrip("/")
    try:
        return p.read_text(errors="replace").strip("\x00").strip()
    except (OSError, UnicodeDecodeError):
        return None


def unknown(source: str, why: str) -> dict[str, Any]:
    """What a probe returns when it cannot determine something.

    Same contract as Lab 01, deliberately. Not None threaded through the report,
    not a plausible default: an explicit record that the probe ran, failed, and
    knows why. It is marked positively.
    """
    return {"value": None, "source": source, "status": "unknown", "detail": why}


def getattr_path(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        try:
            cur = getattr(cur, part)
        except AttributeError:
            return default
    return cur

def major_minor(version: str) -> str | None:
    if not version:
        return None
    parts = str(version).split(".")
    if len(parts) < 2:
        return None
    try:
        return f"{int(parts[0])}.{int(parts[1])}"
    except ValueError:
        return None