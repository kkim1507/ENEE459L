"""Probes — read what the machine says about itself.

INSTRUCTOR SOLUTION. Do not distribute. The student copy of this file has the
body of every function below replaced by `raise NotImplementedError`.

Every probe takes a `root` argument and reads nothing outside it. That is not
decoration: it is what makes this lab gradeable without twenty boards on a
desk, and it is the reason the test suite can present a fake SD-booted machine
and check that the student's code notices. Code that hardcodes "/" cannot be
tested, and a measurement you cannot test is a measurement you cannot trust —
which is the whole argument of Lecture 01, applied to the student's own code.

Each probe returns a dict with, at minimum, a `value` and a `source` key. The
`source` is the path or command the value came from. A number without its
provenance is not evidence, so the report format refuses to carry one.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any
import pdb
import json

# ---------------------------------------------------------------------------
# Small helpers. These are given to students; the exercise is the probes.
# ---------------------------------------------------------------------------


def read_text(root: Path, rel: str) -> str | None:
    """Read `root/rel`, returning None if it is missing or unreadable.

    Missing is a normal outcome here, not an error: a devkit with no NVMe
    genuinely has no /sys/block/nvme0n1, and the report needs to say so rather
    than crash.
    """
    p = Path(root) / rel.lstrip("/")
    try:
        return p.read_text(errors="replace").strip("\x00").strip()
    except (OSError, UnicodeDecodeError):
        return None


def run(cmd: list[str]) -> str | None:
    """Run a command, returning stdout, or None if it is absent or fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def unknown(source: str, why: str) -> dict[str, Any]:
    """The value this lab returns when it cannot determine something.

    Note what this is not: it is not None threaded through the report, and it
    is not a plausible default. It is an explicit record that the probe ran and
    failed, carrying the reason. Assignment 1's rubric gives credit for these.
    """
    return {"value": None, "source": source, "status": "unknown", "detail": why}

# LnkSta/LnkCap lines look like:
#   LnkSta: Speed 8GT/s, Width x4, TrErr- Train- SlotClk+ DLActive- ...
#   LnkCap: Port #0, Speed 16GT/s, Width x4, ASPM L1, Exit Latency L1 <64us
_SPEED_RE = re.compile(r"Speed\s+([\d.]+)GT/s")
_WIDTH_RE = re.compile(r"Width\s+x(\d+)")

# PCIe generation by per-lane transfer rate. Gen3 is 8 GT/s; the Orin Nano
# devkit's M.2 Key-M slot is wired Gen3 x4, so a Gen4 drive reporting 16 GT/s
# capability and 8 GT/s status is behaving correctly, not underperforming.
#
# Keyed by float, not by the string lspci printed. Keying by string means
# deciding whether "8", "8.0" and "08" are the same rate, and the obvious
# normalisation — stripping trailing zeros and dots — silently turns 20 into 2.
_GEN_BY_GTS = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}


def _parse_link_line(line: str) -> dict[str, Any]:
    speed = _SPEED_RE.search(line)
    width = _WIDTH_RE.search(line)
    gts = float(speed.group(1)) if speed else None
    return {
        "raw": line.strip(),
        "gts": gts,
        "width": int(width.group(1)) if width else None,
        "gen": _GEN_BY_GTS.get(gts) if gts is not None else None,
    }

def generate_interpretation_string(negotiated, capability):
    if capability["gts"] > negotiated["gts"]:
        interpretation = (
            f"drive capable of Gen{capability['gen']}, link running at "
            f"Gen{negotiated['gen']} — expected on this carrier board, "
            "whose M.2 Key-M slot is wired Gen3 x4"
        )
    else:
        interpretation = (
            f"link running at its full capability, Gen{negotiated['gen']} "
            f"x{negotiated['width']}"
        )
    return interpretation
    


# ---------------------------------------------------------------------------
# The probes.
# ---------------------------------------------------------------------------

## An example code.
def probe_module_model(root: Path = Path("/")) -> dict[str, Any]:
    """Which board is this?

    The device tree model string is the most trustworthy identity on a Jetson —
    it comes from the hardware description the bootloader handed the kernel,
    not from anything installed afterwards.
    """

    # step 1: Read the raw null-terminated text from /proc/device-tree/model
    src = "/proc/device-tree/model"
    raw = read_text(root, src)

    # if unable to read, return an empty dictionary by calling unknown().
    if not raw:
        return unknown(src, "device tree model node absent — not a Jetson, or /proc not mounted")
    
    # step 2: strip null bytes and whitespace from raw string
    raw = raw.rstrip("\x00").strip()

    return {
        "value": raw, 
        "source": src, 
        "status": "ok"
    }

# todo by students
def probe_memory_total_kb(root: Path = Path("/")) -> dict[str, Any]:

    src = "/proc/meminfo"
    raw = read_text(root, src)
    pattern = r"^MemTotal:\s+(\d+)\s+kB"
    m = re.search(pattern, raw) if raw else None
    if not m:
        return unknown(src, "MemTotal entry not found in /proc/meminfo") 

    return {
        "value": int(m.group(1)), 
        "source": src, 
        "status": "ok"
    }


def probe_root_source(root: Path = Path("/")) -> dict[str, Any]:
    """
    What device is the root filesystem actually mounted from?

    This is the probe the lab is built around. A unit that boots from the SD
    card works, boots, and passes every casual inspection — and then runs the
    semester's benchmarks against a card an order of magnitude slower than the
    NVMe sitting unused in the slot. The failure is silent, which is exactly
    why it has to be a command rather than an assumption.

    /proc/mounts is preferred over `findmnt` because it needs no external
    binary and no elevation, and because it is what findmnt reads anyway.
    """
    
    src = "/proc/mounts"
    raw = read_text(root, src)
    if not raw:
        return unknown(src, "no root mount entry found in mount table")
    
    value = None
    kind = None

    for line in raw.splitlines():
        words = line.split()
        if len(words) < 2:
            continue
        device, mount_point = words[0], words[1]
        if mount_point == "/":
            value = device
            if device.startswith("/dev/nvme"):
                kind = "nvme"
            elif device.startswith("/dev/mmcblk") or device.startswith("/dev/sd"):
                kind = "ssd"
            else:
                kind = "unknown"
            break

        if value is None:
            return unknown(src, "root mount entry not found in /proc/mounts")

    return {
        "value": value, 
        "kind": kind, 
        "source": src, 
        "status": "ok"
    }


def probe_nvme_present(root: Path = Path("/")) -> dict[str, Any]:
    """Is there an NVMe device visible as a block device at all?

    Deliberately separate from probe_root_source. A machine can have an NVMe
    fitted and still boot from the SD card, and telling those two states apart
    is what lets the troubleshooting tree in the lab guide send a student to
    the right branch.
    """

    src = "/sys/block/nvme0n1"
    device_path = Path(root) / "sys/block/nvme0n1"
    value = device_path.exists()
    model = None

    if value:
        model_path = device_path / "device/model"
    try:
        model = model_path.read_text(errors="replace").strip("\x00").strip()
    except (OSError, UnicodeDecodeError):
        return unknown(src, "NVMe model could not be read")

    return {
        "value": value,
        "model": model,
        "source": src,
        "status": "ok",
    }




def probe_pcie_link(root: Path = Path("/"), lspci_output: str | None = None) -> dict[str, Any]:
    """What did the PCIe link negotiate, and what was it capable of?

    Two numbers, not one. The gap between them is the lab's worked example of
    spec sheet against measured reality: a Gen4 drive in a Gen3 slot advertises
    16 GT/s and settles at 8 GT/s, and a student who reports only the second
    number has recorded a fact without recording what it means.

    `lspci_output` exists so the tests can drive this without root or hardware.
    In normal use it is None and the probe shells out.
    """

    src = "lspci -vv"

    if lspci_output is None:
        lspci_output = run(["lspci", "-vv"])
        if lspci_output is None:
            return unknown(src, "lspci command missing or failed")

    cap_line = None
    sta_line = None
    
    for line in lspci_output.splitlines():
        if "LnkCap:" in line:
            cap_line = line
        if "LnkSta:" in line:
            sta_line = line

    if cap_line is None or sta_line is None:
        return unknown(src, "PCIe link capability/status lines missing")

    capability = _parse_link_line(cap_line)
    negotiated = _parse_link_line(sta_line)

    interpretation = generate_interpretation_string(negotiated, capability)

    return {
        "value": negotiated["gts"],
        "negotiated": negotiated,
        "capability": capability,
        "source": src,
        "status": "ok",
        "interpretation": interpretation
    }


def probe_thermal_zones(root: Path = Path("/")) -> dict[str, Any]:
    """Every thermal zone the kernel exposes, in degrees C.

    Sysfs reports millidegrees. The division by 1000 is the entire trap: a
    report claiming the board idles at 43,000 degrees has been submitted more
    than once, and it is a good, cheap lesson in reading units before reading
    numbers.
    """

    src = "/sys/class/thermal/thermal_zone*/temp"
    zones = []

    for zone_path in Path(root).glob("sys/class/thermal/thermal_zone*/"):
        type_path = zone_path / "type"
        temp_path = zone_path / "temp"

        try:
            type_raw = type_path.read_bytes()
            temp_raw = temp_path.read_bytes()

            if type_raw is None or temp_raw is None:
                continue

            zone_type = type_raw.decode("utf-8", errors="replace").strip("\x00").strip()
            temp_val = temp_raw.decode("utf-8", errors="replace").strip("\x00").strip()
            temp_celsius = (int)(temp_val) // 1000
            zone_name = zone_path.name  


            zones.append({
                "zone": zone_name,
                "type": zone_type,
                "temp_c": temp_celsius,
            })

        except (OSError, UnicodeDecodeError, ValueError, AttributeError):
            continue

    if not zones:
        return unknown(src, "no thermal zones found")

    max_temp = max(zone["temp_c"] for zone in zones)

    return {
        "value": max_temp,
        "zones": zones,
        "source": src,
        "status": "ok",
    }


def probe_power_mode(root: Path = Path("/"), nvpmodel_output: str | None = None) -> dict[str, Any]:
    """Which nvpmodel power mode is active?

    Recorded on every artifact this course produces. Lecture 01 slide 24 is
    the argument for why: two students reporting different throughput for the
    same model are usually reporting different power modes, and without this
    field there is no way to find that out after the fact.
    """

    src = "nvpmodel -q"

    if nvpmodel_output is None:
        nvpmodel_output = run(["nvpmodel", "-q"])

    if not nvpmodel_output:
        return unknown(src, "nvpmodel -q failed or was not found")

    mode_match = re.search(r"NV Power Mode:\s*(.+)", nvpmodel_output)

    if mode_match is None:
        return unknown(src, "NV Power Mode line not found")

    mode_name = mode_match.group(1).strip()
    mode_id_match = re.search(r"^\s*(\d+)\s*$", nvpmodel_output, re.MULTILINE)

    if mode_id_match is None:
        return unknown(src, "power mode ID not found")

    mode_id = int(mode_id_match.group(1))

    return {
        "value": mode_name,
        "mode_id": mode_id,
        "source": src,
        "status": "ok",
    }

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
#    out = probe_power_mode()
#    print(out)

# for generating system_report.json
if __name__ == "__main__":
    report = {
        "module_model": probe_module_model(),
        "memory_total_kb": probe_memory_total_kb(),
        "root_source": probe_root_source(),
        "nvme_present": probe_nvme_present(),
        "pcie_link": probe_pcie_link(),
        "thermal_zones": probe_thermal_zones(),
        "power_mode": probe_power_mode(),
    }
    
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

