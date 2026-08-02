# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Out-of-process hang collector, and the incident-directory contract.

tt-metal runs this from its dispatch-timeout hook, through ``std::system``, from
inside the hung process and before it throws. Collecting first is the required
order: launching anything overwrites the evidence.

It is a standalone script on purpose. It must not import the ``ttl`` package,
because loading the MLIR extension and ttnn a second time inside a process that
is already stuck is both slow and a chance to fail while reporting a failure.
For the same reason it must never raise: anything it cannot collect is written
into the report as a named failure.

Every read is halt-free. PCs come off the debug bus and frames come from DWARF,
so the device is left exactly as the hang found it and nothing here decides its
fate. Halting a core to unwind real frames is terminal on Blackhole, which is why
this does not do it.

Then it parks: see park() for why never returning is what keeps the device open
and the teardown waits from firing. Killing the process and resetting the device
are the caller's calls to make.
"""

import json
import os
import select
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# The incident directory is a fixed location, not a per-run one, so that a
# wrapper script or a follow-up shell always knows where to look. Every file in
# it carries a UTC stamp so a leftover from an earlier run reads as stale.
INCIDENT_DIR = "/tmp/ttlang_hang"
CB_TABLE_PATH = "/tmp/ttlang_cb_table.txt"

DIR_ENV = "TTLANG_HANG_DIR"
MODE_ENV = "TTLANG_ON_HANG"
LAUNCH_ENV = "TTLANG_HANG_LAUNCHES"
DEVICES_ENV = "TTLANG_HANG_DEVICES"
MAX_CORES_ENV = "TTLANG_HANG_MAX_CORES"

MODE_OFF = "off"
MODE_ON = "on"
MODES = (MODE_OFF, MODE_ON)

# Selectable earlier, kept only so choosing one says what happened to it rather
# than reading as a typo. All three acted on the hang: fast and deep stopped the
# process, recover closed and reopened the device, deep halted cores to unwind.
RETIRED_MODES = ("fast", "recover", "deep")

PROGRAMS_FILE = "programs.jsonl"
REPORT_FILE = "report.txt"
STACKS_FILE = "stacks.txt"
MANIFEST_FILE = "manifest.json"

RISCS = ("brisc", "ncrisc", "trisc0", "trisc1", "trisc2")

# Bounds. Stack collection is the slow part (DWARF per ELF, two NOC reads per
# PC), and an unbounded sweep of a 32-chip galaxy would take minutes at exactly
# the moment the user is waiting. Truncation is always reported, never silent.
DEFAULT_MAX_CORES = 64
MAX_ELFS_PER_RISC = 8
PC_RESAMPLE_SECONDS = 0.05
LAUNCH_REPORT_LIMIT = 8


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def mode() -> str:
    """The configured mode, never raising: ttl.hang.mode is what validates."""
    value = os.environ.get(MODE_ENV, MODE_ON).strip().lower()
    if value not in MODES:
        return MODE_ON
    return value


def incident_dir() -> Path:
    path = Path(os.environ.get(DIR_ENV, INCIDENT_DIR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def max_cores() -> int:
    """Bound a core sweep without letting diagnostics configuration raise."""
    try:
        return max(1, int(os.environ.get(MAX_CORES_ENV, DEFAULT_MAX_CORES)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_CORES


class Report:
    """Accumulates lines for report.txt and named failures for the manifest."""

    def __init__(self):
        self.lines = []
        self.failures = {}

    def say(self, line: str = "") -> None:
        self.lines.append(line)

    def fail(self, what: str, error: BaseException) -> None:
        self.say(f"  FAILED {what}: {type(error).__name__}: {error}")
        self.failures[what] = traceback.format_exc()

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def load_programs(directory: Path) -> list:
    """Read the compile-time program registry, newest last."""
    path = directory / PROGRAMS_FILE
    if not path.exists():
        return []
    programs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            programs.append(json.loads(line))
    return programs


def select_programs(programs: list) -> list:
    """The programs worth symbolizing against, most recently launched first.

    The launch ring is a heuristic and says so: dispatch is asynchronous, so the
    program the device is stuck in may be several behind the one the host
    launched last. Everything compiled in this process stays available as a
    fallback.
    """
    launched = [key for key in os.environ.get(LAUNCH_ENV, "").split(",") if key]
    by_key = {}
    for program in programs:
        by_key[program["key"]] = program
    selected = [by_key[key] for key in reversed(launched) if key in by_key]
    if selected:
        return selected
    return list(reversed(programs))


def kernel_elfs(programs: list, cache_root) -> dict:
    """Map risc name to candidate ELFs built from tt-lang's generated sources.

    tt-metal lays the cache out as
    ``<cache>/<build key>/kernels/<kernel name>/<kernel hash>/<risc>/<risc>.elf``
    where the kernel name is the generated source stem, so the source paths
    tt-lang already recorded are enough to find the ELFs. That avoids decoding
    launch messages to recover kernel ids, which is the part of this that goes
    stale between tt-metal versions.
    """
    by_risc = {}
    root = Path(cache_root)
    for program in programs:
        for kernel in program.get("kernels", []):
            stem = Path(kernel["path"]).stem
            matches = sorted(
                root.glob(f"*/kernels/{stem}/*/*/*.elf"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for elf in matches:
                if elf.name.endswith(".xip.elf") or "_weakened" in elf.name:
                    continue
                risc = elf.parent.name
                paths = by_risc.setdefault(risc, [])
                if str(elf) not in paths:
                    paths.append(str(elf))
    return by_risc


def firmware_elfs(cache_root) -> dict:
    """Map risc name to firmware ELFs, for PCs that stopped outside the kernel."""
    by_risc = {}
    for elf in Path(cache_root).glob("*/firmware/*/*.elf"):
        if "_weakened" in elf.name:
            continue
        by_risc.setdefault(elf.parent.name, []).append(str(elf))
    for risc in by_risc:
        by_risc[risc].sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return by_risc


def select_cores(programs: list) -> list:
    """Logical cores to sample: the union of the selected programs' grids."""
    cores = []
    for program in programs:
        for core_range in program.get("cores") or []:
            (start_x, start_y), (end_x, end_y) = core_range
            for y in range(start_y, end_y + 1):
                for x in range(start_x, end_x + 1):
                    if (x, y) not in cores:
                        cores.append((x, y))
    return cores


def cache_roots() -> list:
    """Candidate roots holding the tt-metal kernel cache, best guess first.

    TT_METAL_CACHE is the *parent* of the cache: tt-metal appends the
    "tt-metal-cache" component itself (``rtoptions.cpp:409``,
    ``normalize_path(value, "tt-metal-cache")``), so the variable's value alone is
    one directory short of the ELFs.
    """
    candidates = []
    configured = os.environ.get("TT_METAL_CACHE")
    if configured:
        candidates.append(Path(configured) / "tt-metal-cache")
        candidates.append(Path(configured))
    home = os.environ.get("HOME")
    if home:
        candidates.append(Path(home) / ".cache" / "tt-metal-cache")
    return candidates


def resolve_cache_root(report: Report) -> Path:
    """The first candidate root that actually holds a built cache.

    Tested by looking for a firmware or kernel directory rather than by trusting
    the variable. Some cache configurations contain compiled kernels but no
    cached firmware, and rejecting those roots silently loses all symbols.
    """
    tried = []
    for candidate in cache_roots():
        tried.append(str(candidate))
        if any(candidate.glob("*/firmware")) or any(
            candidate.glob("*/kernels")
        ):
            report.say(f"kernel cache: {candidate}")
            return candidate
    report.say(f"no built kernel cache found; PCs only. Tried: {', '.join(tried)}")
    return None


def select_devices() -> list:
    """Device ids to sample.

    Defaults to device 0 rather than the whole mesh: on a 32-chip galaxy every
    chip runs the same program, and sampling all of them turns a few seconds of
    collection into minutes. Widen with TTLANG_HANG_DEVICES=0,1,2.
    """
    raw = os.environ.get(DEVICES_ENV, "0")
    return [int(part) for part in raw.split(",") if part.strip()]


def frame_text(entry) -> str:
    """One symbolized frame."""
    name = entry.function_name or "??"
    info = entry.file_info
    if info is None:
        return f"{name}"
    return f"{name} at {info.file}:{info.line}"


def sample_pcs(context, device_id: int, cores: list, report: Report) -> dict:
    """Read every RISC PC on the given cores without halting anything.

    ``get_pc`` falls back to halting when a core has no debug-bus PC signal, so
    the signal is checked first and a core without one is reported rather than
    silently halted: a halt on Blackhole costs the device.
    """
    from ttexalens._lib_helpers import convert_coordinate

    pcs = {}
    for x, y in cores:
        try:
            coordinate = convert_coordinate(f"{x},{y}", device_id, context)
        except Exception as error:
            report.fail(f"coordinate {x},{y} on device {device_id}", error)
            continue
        for risc_name in RISCS:
            key = (x, y, risc_name)
            try:
                risc_debug = coordinate.noc_block.get_risc_debug(risc_name)
                if risc_debug.is_in_reset():
                    pcs[key] = ("in reset", None)
                    continue
                if getattr(risc_debug, "debug_bus_pc_signal", None) is None:
                    pcs[key] = ("no halt-free PC signal", None)
                    continue
                pcs[key] = (None, risc_debug.get_pc())
            except Exception as error:
                pcs[key] = (f"{type(error).__name__}: {error}", None)
    return pcs


def collect_stacks(
    context, device_id: int, cores: list, elfs: dict, report: Report
) -> list:
    """Per-RISC PC, motion verdict and symbolized top frame for one device."""
    lines = []
    first = sample_pcs(context, device_id, cores, report)
    time.sleep(PC_RESAMPLE_SECONDS)
    second = sample_pcs(context, device_id, cores, report)

    for (x, y, risc_name), (error, pc) in first.items():
        label = f"device {device_id} core {x},{y} {risc_name}"
        if pc is None:
            lines.append(f"{label}: {error}")
            continue
        later = second.get((x, y, risc_name), (None, None))[1]
        motion = "STATIONARY" if later == pc else f"ADVANCING (then 0x{later:08x})"
        lines.append(f"{label}: pc=0x{pc:08x} {motion}")
        lines.extend(symbolize(context, x, y, risc_name, pc, elfs, report))
    return lines


def symbolize(context, x, y, risc_name, pc, elfs, report: Report) -> list:
    """Frames for one PC, from DWARF alone.

    ``top_callstack`` resolves the PC and its inlined frames without touching the
    core. Walking further up the stack needs the core halted to read registers,
    which on Blackhole is terminal, so the top frames are all this collects.
    """
    candidates = elfs.get(risc_name, [])
    if not candidates:
        return ["    (no ELF found for this risc; PC is unsymbolized)"]

    lines = []
    try:
        from ttexalens.tt_exalens_lib import top_callstack

        frames = top_callstack(pc, candidates, context=context, extract_variables=False)
        lines.extend(f"    {frame_text(frame)}" for frame in frames)
        if not frames:
            lines.append("    (PC did not resolve in any candidate ELF)")
    except Exception as error:
        report.fail(f"top_callstack for {risc_name} at {x},{y}", error)
    return lines


def copy_kernel_sources(directory: Path, programs: list, report: Report) -> list:
    """Copy the generated sources in: they live in /tmp and can be cleaned."""
    target = directory / "kernels"
    target.mkdir(exist_ok=True)
    copied = []
    for program in programs:
        for kernel in program.get("kernels", []):
            source = Path(kernel["path"])
            try:
                (target / source.name).write_bytes(source.read_bytes())
                copied.append(source.name)
            except OSError as error:
                report.fail(f"copy {source}", error)
    return copied


def file_stamp(path) -> dict:
    """Presence, size and write time, so a leftover file reads as stale."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "present": False}
    stat = path.stat()
    written = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
    return {
        "path": str(path),
        "present": True,
        "bytes": stat.st_size,
        "modified": written.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def describe_file(path) -> str:
    stamp = file_stamp(path)
    if not stamp["present"]:
        return f"{stamp['path']} MISSING"
    return f"{stamp['path']} {stamp['bytes']} bytes, written {stamp['modified']}"


def guard(report: Report, what: str, action, fallback=None):
    """Run one collection phase. A failure is recorded, never propagated.

    Phases are isolated from each other on purpose: a device that cannot be
    attached must not cost us the program registry or the kernel sources, which
    are the parts that are still readable when the hardware is not.
    """
    try:
        return action()
    except Exception as error:
        report.fail(what, error)
        return fallback


def resolve_elfs(selected: list, cache_root) -> dict:
    """Kernel ELFs for the selected programs, plus firmware, capped per risc."""
    elfs = kernel_elfs(selected, cache_root)
    for risc, paths in firmware_elfs(cache_root).items():
        elfs.setdefault(risc, []).extend(paths)
    return {risc: paths[:MAX_ELFS_PER_RISC] for risc, paths in elfs.items()}


def sample_device(devices: list, cores: list, elfs: dict, report: Report) -> list:
    """Attach to the device and collect stacks for every selected core."""
    from ttexalens.tt_exalens_init import init_ttexalens

    context = init_ttexalens()
    stacks = []
    for device_id in devices:
        stacks.extend(collect_stacks(context, device_id, cores, elfs, report))
    return stacks


def main() -> int:
    started = utc_stamp()
    directory = incident_dir()
    report = Report()
    active_mode = mode()

    report.say(f"tt-lang hang incident, collected {started}")
    report.say(f"  mode          {active_mode}")
    # Parent, not "hung process": std::system may leave a /bin/sh in between, so
    # the hung python is one or two levels up.
    report.say(f"  collector pid {os.getpid()}, parent pid {os.getppid()}")
    report.say(f"  incident dir  {directory}")
    report.say()

    manifest = {
        "collected": started,
        "mode": active_mode,
        "collector_pid": os.getpid(),
        "parent_pid": os.getppid(),
        "launched": os.environ.get(LAUNCH_ENV, ""),
    }

    programs = guard(
        report, "read program registry", lambda: load_programs(directory), []
    )
    selected = select_programs(programs)
    manifest["programs_compiled"] = len(programs)
    manifest["programs_selected"] = [p["key"] for p in selected]
    report.say(
        f"programs: {len(programs)} compiled this process, "
        f"{len(selected)} selected for symbolization"
    )
    for program in selected[:LAUNCH_REPORT_LIMIT]:
        report.say(f"  {program['key']} (compiled {program['stamp']})")
    report.say()

    manifest["kernels_copied"] = guard(
        report,
        "copy kernel sources",
        lambda: copy_kernel_sources(directory, selected, report),
        [],
    )

    cache_root = guard(
        report, "resolve kernel cache", lambda: resolve_cache_root(report)
    )
    manifest["cache_root"] = str(cache_root) if cache_root else None
    elfs = {}
    if cache_root is not None:
        elfs = guard(
            report, "find ELFs", lambda: resolve_elfs(selected, cache_root), {}
        )
        manifest["elfs"] = elfs
        found = ", ".join(
            f"{risc}:{len(paths)}" for risc, paths in sorted(elfs.items())
        )
        report.say(f"ELFs found: {found or 'none'}")

    cores = select_cores(selected)
    core_limit = max_cores()
    if len(cores) > core_limit:
        report.say(
            f"grid has {len(cores)} cores; sampling the first {core_limit}. "
            f"Raise {MAX_CORES_ENV} to widen."
        )
        cores = cores[:core_limit]
    manifest["cores"] = [list(core) for core in cores]

    devices = select_devices()
    manifest["devices"] = devices
    stacks = []
    if not cores:
        report.say("no core ranges recorded, so no core was sampled")
    else:
        report.say(
            f"sampling {len(cores)} cores x {len(RISCS)} riscs on devices {devices}. "
            f"Other devices were not sampled; widen with {DEVICES_ENV}=0,1,2."
        )
        stacks = guard(
            report,
            "sample device",
            lambda: sample_device(devices, cores, elfs, report),
            [],
        )

    manifest["cb_table"] = file_stamp(CB_TABLE_PATH)
    manifest["failures"] = report.failures
    manifest["finished"] = utc_stamp()

    # Stacks and manifest first, then the report, so the report's artifact list
    # describes files that exist rather than files it is about to create.
    try:
        (directory / STACKS_FILE).write_text("\n".join(stacks) + "\n")
        (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
    except OSError as error:
        print(f"tt-lang: could not write hang incident to {directory}: {error}")
        return 1

    report.say()
    report.say(f"{len(stacks)} stack line(s) collected")
    report.say("artifacts:")
    for path in (
        directory / STACKS_FILE,
        directory / MANIFEST_FILE,
        directory / PROGRAMS_FILE,
        Path(CB_TABLE_PATH),
    ):
        report.say(f"  {describe_file(path)}")

    try:
        (directory / REPORT_FILE).write_text(report.text())
    except OSError as error:
        print(f"tt-lang: could not write {REPORT_FILE}: {error}")
        return 1

    print(f"tt-lang: HANG DETECTED, no dispatch progress. Incident: {directory}")
    print(f"tt-lang:   stacks    {directory / STACKS_FILE}")
    print(f"tt-lang:   report    {directory / REPORT_FILE}")
    print(f"tt-lang:   CB table  {CB_TABLE_PATH}")
    if report.failures:
        print(
            f"tt-lang: {len(report.failures)} collection step(s) failed; "
            f"see {REPORT_FILE}"
        )
    park()
    return 0


def park() -> None:
    """Block here forever, holding the device open, until the user kills us.

    tt-metal runs this collector through a blocking ``std::system`` and throws
    only once it returns (``metal_context.cpp:793``). Never returning means the
    throw never happens, so no caller ``finally`` and no pytest fixture closes the
    device: no teardown wait per chip, and no 32-chip cascade of them. The device
    stays exactly as the hang left it, which is what makes a live tt-exalens read
    from another shell worth doing.

    Ctrl-C exits instead, and then the throw and the teardown waits follow.
    Killing the hung process also exits automatically.  ``std::system`` inserts
    a shell between that process and this collector, so watching only
    ``getppid()`` would leak both the shell and collector after the hung process
    is killed.
    """
    print("tt-lang: PARKED, holding the device open so you can inspect it live.")
    print("tt-lang:   attach from another shell (halt-free, safe while parked);")
    print("tt-lang:   see 'Inspecting a stuck CB or semaphore' in CLAUDE.md.")
    print("tt-lang: when done, kill the hung process, then reset the device:")
    print("tt-lang:   tt-smi -glx_reset_auto on a galaxy, tt-smi -r otherwise.")
    sys.stdout.flush()
    hung_pid = parent_pid(os.getppid())
    try:
        if hung_pid is None:
            while True:
                time.sleep(3600)
        with os.fdopen(os.pidfd_open(hung_pid)) as pidfd:
            poller = select.poll()
            poller.register(pidfd, select.POLLIN)
            while not poller.poll(3_600_000):
                pass
        print("tt-lang: hung process exited; releasing collector resources.")
        sys.stdout.flush()
    except KeyboardInterrupt:
        print("tt-lang: unparked. tt-metal will now throw and close the device.")
        sys.stdout.flush()


def parent_pid(pid: int):
    """Return a Linux process's parent, or ``None`` if it already exited."""
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    return None


if __name__ == "__main__":
    sys.exit(main())
