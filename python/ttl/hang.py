# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hang detection: arm tt-metal's timeout, and record what a collector will need.

tt-metal already has the hard part: its dispatch waits are progress-gated against
a device-side counter of dispatch commands completed, so a queue that is still
retiring work does not trip the timeout however long the host waits. What it does
not have by default is a timeout at all (0.0 means spin forever) or anything to
run when one trips.

The counter only moves when a command *completes*, which sets where this can cry
wolf: one dispatch command that legitimately runs longer than the window looks
exactly like a hang. Host-side work such as JIT compilation is outside the guarded
waits entirely, so it neither counts as progress nor consumes the window.

tt-lang arms it and points it at ``ttl/hang_collect.py``, which tt-metal runs
synchronously inside the hung process, before it throws. That collector reads PCs
and symbolizes them without halting anything, writes an incident directory, and
says where it is. Then tt-metal's throw propagates as it always would.

Nothing here touches the device or the process. Inspecting the incident, killing
the process and resetting the device are all the caller's calls to make, on the
evidence the collector left behind.
"""

import json
import os
import sys
from pathlib import Path

from .hang_collect import (
    DIR_ENV,
    INCIDENT_DIR,
    LAUNCH_ENV,
    MODE_ENV,
    MODE_OFF,
    MODE_ON,
    MODES,
    PROGRAMS_FILE,
    RETIRED_MODES,
    incident_dir,
    utc_stamp,
)

TIMEOUT_ENV = "TTLANG_HANG_TIMEOUT_SECONDS"
FORCE_REINIT_ENV = "TTLANG_FORCE_REINIT"

# Short, because this is "no dispatch command completed at all for five seconds",
# not "one operation took five seconds", and our programs run in microseconds to
# milliseconds. Read the overload warning on TT_METAL_OPERATION_TIMEOUT_SECONDS in
# configure_metal_env before lowering it further.
DEFAULT_TIMEOUT_SECONDS = 5.0

# Only the last few launches are worth symbolizing against, and the ring is a
# heuristic anyway: dispatch is asynchronous, so the device may be behind.
LAUNCH_RING = 8

_launch_ring: list = []
_registry_started = False


def mode() -> str:
    """The configured hang mode, validated."""
    value = os.environ.get(MODE_ENV, MODE_ON).strip().lower()
    if value in RETIRED_MODES:
        raise NotImplementedError(
            f"{MODE_ENV}={value} is no longer a mode. Collection is halt-free and "
            f"acts on nothing: it reports the hang and leaves the process and the "
            f"device to you. Use '{MODE_ON}' (the default) or '{MODE_OFF}'."
        )
    if value not in MODES:
        raise ValueError(
            f"{MODE_ENV}={value!r} is not one of {', '.join(MODES)}. "
            f"'{MODE_ON}' reports a dispatch timeout and collects stacks, "
            f"'{MODE_OFF}' restores tt-metal's default of waiting forever."
        )
    return value


def timeout_seconds() -> float:
    """Seconds without dispatch progress that count as a hang."""
    raw = os.environ.get(TIMEOUT_ENV)
    if raw is None:
        return DEFAULT_TIMEOUT_SECONDS
    value = float(raw)
    if value <= 0.0:
        raise ValueError(
            f"{TIMEOUT_ENV}={raw!r} must be positive. "
            f"Use {MODE_ENV}={MODE_OFF} to disable hang detection."
        )
    return value


def configure_metal_env() -> None:
    """Arm tt-metal's timeout and collector before the first device open.

    These are env-only: tt-metal reads them once when RunTimeOptions is
    constructed, at the first device open, and exposes no setter. setdefault
    throughout, so an explicitly set tt-metal variable always wins.

    TT_METAL_OPERATION_TIMEOUT_SECONDS is overloaded, which is why the window is
    not smaller. Besides the progress-gated dispatch waits it also bounds, as
    plain wall clock, ``wait_until_cores_done`` at device init and teardown (which
    is otherwise unbounded) and the fabric topology mapping rendezvous (whose own
    default is 120s). The rendezvous is a cross-host all-gather, so it is instant
    at world_size 1; the exposure worth watching is a device open that starts
    reporting cores not done.
    """
    if os.environ.get(FORCE_REINIT_ENV, "1") != "0":
        # tt-metal enables this on mere presence, so setting it to "0" would
        # still enable it; the opt-out has to be our own variable.
        os.environ.setdefault("TT_METAL_FORCE_REINIT", "1")

    if mode() == MODE_OFF:
        return

    os.environ.setdefault(
        "TT_METAL_OPERATION_TIMEOUT_SECONDS", str(timeout_seconds())
    )
    collector = Path(__file__).resolve().parent / "hang_collect.py"
    os.environ.setdefault(
        "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE",
        f"{sys.executable} {collector}",
    )
    os.environ.setdefault(DIR_ENV, INCIDENT_DIR)


def note_program(program_hash, kernel_paths, core_ranges) -> None:
    """Record a compiled program so the collector can find its ELFs and cores.

    Written at compile time, which happens once per program, rather than per
    launch. The collector needs the generated source names to locate kernel ELFs
    in the tt-metal cache, and the grid to know which cores to sample.
    """
    global _registry_started
    if mode() == MODE_OFF or not kernel_paths:
        return

    from .kernel_runner import _serialize_core_ranges

    entry = {
        "key": program_key(kernel_paths),
        "program_hash": program_hash,
        "stamp": utc_stamp(),
        "pid": os.getpid(),
        "kernels": [
            {"path": path, "thread_type": thread_type}
            for path, thread_type in kernel_paths
        ],
        "cores": _serialize_core_ranges(core_ranges),
    }
    path = incident_dir() / PROGRAMS_FILE
    with open(path, "a" if _registry_started else "w") as fd:
        fd.write(json.dumps(entry) + "\n")
    _registry_started = True


def program_key(kernel_paths) -> str:
    """Stable id for a compiled program.

    The generated source names are content addressed, so the first kernel's stem
    identifies the program without needing a program hash, which is optional.
    """
    return Path(kernel_paths[0][0]).stem


def note_launch(key: str) -> None:
    """Keep the last few launched program keys where the collector can read them.

    In the environment rather than a file: the collector is a subprocess of the
    hung process, so it inherits this for free, and a decode loop cannot afford a
    file write per launch.
    """
    if mode() == MODE_OFF:
        return
    if _launch_ring and _launch_ring[-1] == key:
        return
    _launch_ring.append(key)
    del _launch_ring[:-LAUNCH_RING]
    os.environ[LAUNCH_ENV] = ",".join(_launch_ring)
