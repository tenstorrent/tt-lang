# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hang detection, reporting and fast device recovery.

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
synchronously inside the hung process before it throws. This module handles what
happens after that throw: report, try to hand back a clean device, and exit with
a code that says which of those two happened.

Exiting is deliberate. A dispatch timeout leaves the process holding tensors on a
device whose kernels were killed, so there is nothing useful to continue with;
the value on offer is that the *next* process starts on a clean device instead of
paying a full reset.
"""

import json
import os
import sys
import threading
from pathlib import Path

from .hang_collect import (
    DEVICES_ENV,
    DIR_ENV,
    DIRTY_SENTINEL,
    INCIDENT_DIR,
    LAUNCH_ENV,
    MODE_DEEP,
    MODE_ENV,
    MODE_FAST,
    MODE_OFF,
    MODES,
    PROGRAMS_FILE,
    incident_dir,
    mark_device_dirty,
    utc_stamp,
)

TIMEOUT_ENV = "TTLANG_HANG_TIMEOUT_SECONDS"
FORCE_REINIT_ENV = "TTLANG_FORCE_REINIT"

# Short, because this is "no dispatch command completed at all for five seconds",
# not "one operation took five seconds", and our programs run in microseconds to
# milliseconds. Read the overload warning on TT_METAL_OPERATION_TIMEOUT_SECONDS in
# configure_metal_env before lowering it further.
DEFAULT_TIMEOUT_SECONDS = 5.0

EXIT_RECOVERED = 2
EXIT_RESET_REQUIRED = 3

# Recovery itself can wedge (a teardown that waits on the same dead core). A
# deadline keeps the worst case bounded, which is the whole point of the feature.
RECOVERY_DEADLINE_SECONDS = 180.0

# Only the last few launches are worth symbolizing against, and the ring is a
# heuristic anyway: dispatch is asynchronous, so the device may be behind.
LAUNCH_RING = 8

RECOVERY_FILE = "recovery.txt"

_launch_ring: list = []
_registry_started = False
_last_device = None


def mode() -> str:
    """The configured hang mode, validated."""
    value = os.environ.get(MODE_ENV, MODE_FAST).strip().lower()
    if value not in MODES:
        raise ValueError(
            f"{MODE_ENV}={value!r} is not one of {', '.join(MODES)}. "
            f"'{MODE_OFF}' restores tt-metal's default of waiting forever, "
            f"'{MODE_FAST}' collects without halting and recovers the device, "
            f"'{MODE_DEEP}' also unwinds real stack frames and forfeits the device."
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
    """Arm tt-metal's timeout and recovery before the first device open.

    These three are env-only: tt-metal reads them once when RunTimeOptions is
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


def note_device(device) -> None:
    """Remember the device a kernel was launched on, so recovery can reopen it."""
    global _last_device
    _last_device = device


def is_dispatch_timeout(error: BaseException) -> bool:
    """True for the two tt-metal dispatch-timeout throws.

    Matched on the message because tt-metal raises a plain RuntimeError for
    everything; both sites start with "TIMEOUT:".
    """
    return "TIMEOUT:" in str(error) and "potential hang detected" in str(error)


def handle_hang(error: BaseException, device=None):
    """Report a dispatch timeout, try to hand back a clean device, and exit.

    Never returns, and never raises: replacing a hang report with a traceback
    about the reporter would leave the user with strictly less than they had.
    """
    try:
        _handle_hang(error, device)
    except BaseException as reporter_error:
        sys.stderr.write(
            f"tt-lang: hang handling itself failed with {reporter_error!r}. "
            f"FULL DEVICE RESET REQUIRED.\n"
        )
        sys.stderr.flush()
        os._exit(EXIT_RESET_REQUIRED)


def _handle_hang(error: BaseException, device=None):
    active_mode = mode()
    directory = incident_dir()
    lines = [
        f"tt-lang hang recovery, {utc_stamp()}",
        f"  mode      {active_mode}",
        f"  pid       {os.getpid()}",
        f"  error     {error}",
        "",
    ]

    _arm_deadline()

    if active_mode == MODE_DEEP:
        mark_device_dirty("deep hang collection halted cores")
        lines.append("Deep mode halted cores to unwind stacks, so no recovery was")
        lines.append("attempted. FULL DEVICE RESET REQUIRED before the next run.")
        _finish(directory, lines, EXIT_RESET_REQUIRED)

    target = device if device is not None else _last_device
    if target is None:
        mark_device_dirty("no device handle available for recovery")
        lines.append("No device handle was available, so recovery was not attempted.")
        lines.append("FULL DEVICE RESET REQUIRED before the next run.")
        _finish(directory, lines, EXIT_RESET_REQUIRED)

    try:
        _recover(target, lines)
    except Exception as recovery_error:
        lines.append(
            f"  recovery failed: "
            f"{type(recovery_error).__name__}: {recovery_error}"
        )
        mark_device_dirty(f"recovery failed: {recovery_error}")
        lines.append("FULL DEVICE RESET REQUIRED before the next run.")
        _finish(directory, lines, EXIT_RESET_REQUIRED)

    _clear_dirty()
    lines.append("Device closed, reopened and smoke tested. Safe to run again.")
    _finish(directory, lines, EXIT_RECOVERED)


def _recover(device, lines: list) -> None:
    """Close, reopen and prove the device can run a program again.

    Reopened with default parameters rather than the caller's: the reopened
    device only has to pass the smoke test before we exit, and a parameter
    mismatch just costs one extra teardown inside tt-metal.
    """
    import ttnn

    is_mesh = isinstance(device, ttnn.MeshDevice)
    lines.append(f"  closing {'mesh ' if is_mesh else ''}device")
    if is_mesh:
        shape = device.shape
        ttnn.close_mesh_device(device)
        fresh = ttnn.open_mesh_device(mesh_shape=shape)
        close = ttnn.close_mesh_device
    else:
        device_id = device.id()
        ttnn.close_device(device)
        fresh = ttnn.open_device(device_id=device_id)
        close = ttnn.close_device
    lines.append("  reopened")

    try:
        _smoke_test(fresh, is_mesh)
        lines.append("  smoke test passed")
    finally:
        close(fresh)
    lines.append("  closed")


def _smoke_test(device, is_mesh: bool) -> None:
    """Run one real program end to end.

    Firmware coming back proves the RISCs were reset; it does not prove dispatch,
    the command queue or fabric still work. Only launching something does. This
    is a ttnn op rather than a tt-lang kernel so that recovery never depends on
    the compiler, which is a lot of machinery to lean on while cleaning up.
    """
    import torch
    import ttnn

    expected = 4.0
    host = torch.full((32, 32), expected / 2, dtype=torch.bfloat16)
    kwargs = {"dtype": ttnn.bfloat16, "layout": ttnn.TILE_LAYOUT, "device": device}
    if is_mesh:
        kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
    operand = ttnn.from_torch(host, **kwargs)
    result = ttnn.add(operand, operand)
    if is_mesh:
        out = ttnn.to_torch(
            result, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0)
        )
    else:
        out = ttnn.to_torch(result)
    if not torch.allclose(out.float(), torch.full(out.shape, expected), atol=0.1):
        saw = out.float().flatten()[:4].tolist()
        raise RuntimeError(f"smoke test produced {saw}, expected {expected}")


def _arm_deadline() -> None:
    """Force an exit if recovery itself wedges."""

    def expire():
        sys.stderr.write(
            f"tt-lang: hang recovery exceeded {RECOVERY_DEADLINE_SECONDS}s, "
            f"giving up. FULL DEVICE RESET REQUIRED.\n"
        )
        sys.stderr.flush()
        mark_device_dirty("recovery exceeded its deadline")
        os._exit(EXIT_RESET_REQUIRED)

    timer = threading.Timer(RECOVERY_DEADLINE_SECONDS, expire)
    timer.daemon = True
    timer.start()


def _clear_dirty() -> None:
    try:
        os.unlink(DIRTY_SENTINEL)
    except FileNotFoundError:
        pass


def _finish(directory: Path, lines: list, code: int):
    """Write the recovery record, say where everything is, and exit."""
    verdict = "device recovered" if code == EXIT_RECOVERED else "reset required"
    lines.append("")
    lines.append(f"incident directory {directory}")
    lines.append(f"exit code {code} ({verdict})")
    text = "\n".join(lines) + "\n"
    try:
        (directory / RECOVERY_FILE).write_text(text)
    except OSError:
        pass
    # The tt-metal variable, not ours: it is the value actually in force, and it
    # is readable even if someone edited the tt-lang variable mid-run.
    window = os.environ.get("TT_METAL_OPERATION_TIMEOUT_SECONDS", "unset")
    sys.stderr.write("\n" + text)
    sys.stderr.write(
        f"tt-lang: dispatch timeout after {window}s without progress. "
        f"Set {MODE_ENV}={MODE_OFF} to wait forever instead, "
        f"{TIMEOUT_ENV} to change the window, "
        f"{MODE_ENV}={MODE_DEEP} for real stack frames at the cost of the device, "
        f"or {DEVICES_ENV} to sample more chips.\n"
    )
    sys.stderr.flush()
    sys.stdout.flush()
    os._exit(code)
