# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Execute each tutorial example script as an isolated subprocess.

One parametrized test per script under examples/{elementwise-tutorial,
matmul-tutorial,tutorial}/. Each script is run in a fresh process rather than
imported: the scripts open and close a device at module top level, so a crash or
a device wedge in one must not poison the long-lived xdist worker or the next
tutorial. A subprocess inherits the worker's TT_VISIBLE_DEVICES and
TT_METAL_CACHE (set by pin_xdist_worker_to_device), so open_device(0) binds to
that worker's pinned chip during the per-chip parallel phase.

Scheduling reuses .github/scripts/run-hardware-pytests.sh. Single-device
tutorials run sharded across chips; tutorials that open a device mesh carry the
multi_device marker and run serially. Classification comes from the same
"# TTLANG_TUTORIAL_CI:" file tags that .github/scripts/run-tutorials.sh reads,
keeping the two runners consistent.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
TUTORIAL_SUBDIRS = ("elementwise-tutorial", "matmul-tutorial", "tutorial")

CI_TAG_PREFIX = "# TTLANG_TUTORIAL_CI:"
# Opens a device mesh over every visible chip; belongs in the serial phase.
MESH_TAG = "multi-device"
# A mesh tutorial that is meaningless on a single chip; also skipped there.
REQUIRES_MESH_TAG = "requires-multi-device"

# Per-script subprocess budget. 300s covers a cold-cache 8192x8192 matmul on a
# single, partially harvested chip (see run-tutorials.sh); multi-device hosts
# finish each in 20-30s. TUTORIAL_TIMEOUT_SECONDS overrides it.
SUBPROCESS_TIMEOUT_SECONDS = int(os.environ.get("TUTORIAL_TIMEOUT_SECONDS", "300"))
# SIGTERM-to-SIGKILL grace on timeout, so tt-metal can close the device.
SIGTERM_GRACE_SECONDS = 10
# Above the worst-case child lifetime (budget + grace) so the subprocess handles
# the timeout; pytest-timeout's thread method aborts the whole worker.
PYTEST_TIMEOUT_BACKSTOP_SECONDS = (
    SUBPROCESS_TIMEOUT_SECONDS + SIGTERM_GRACE_SECONDS + 60
)

# step_7's all_reduce over a full Galaxy mesh hangs on a known upstream fabric
# bug (tt-lang#585, tt-metal#43749 / #41794); the killed process leaves the
# board's ethernet dispatch firmware wedged. It sorts last within its directory,
# so nothing runs after it in the serial phase.
ALL_REDUCE_TUTORIAL = "step_7_multidevice_shard_k_all_reduce.py"


def _host_chip_count() -> int:
    """Physical Tenstorrent chips on the host.

    Counts digit-named nodes under /dev/tenstorrent, matching count_tt_chips in
    .github/scripts/hardware-test-common.sh and test/lit.cfg.py. This reflects the
    physical host rather than TT_VISIBLE_DEVICES, so the requires-mesh gate is not
    fooled by a pinned worker's single visible chip. TTLANG_TUTORIAL_NUM_DEVICES
    overrides it for host-independent testing.
    """
    override = os.environ.get("TTLANG_TUTORIAL_NUM_DEVICES")
    if override is not None:
        return int(override)
    dev_root = Path("/dev/tenstorrent")
    if not dev_root.is_dir():
        return 0
    return sum(1 for node in dev_root.iterdir() if node.name.isdigit())


def _on_galaxy() -> bool:
    """True on a Galaxy runner, detected from the CI-provided RUNS_ON label."""
    return "galaxy" in os.environ.get("RUNS_ON", "").lower()


def _visible_chip_count() -> int:
    """Number of chips visible to the tutorial subprocess."""
    visible_devices = os.environ.get("TT_VISIBLE_DEVICES")
    if visible_devices is None:
        return _host_chip_count()
    if not visible_devices.strip():
        return 0
    return sum(
        1 for visible_device in visible_devices.split(",") if visible_device.strip()
    )


def _uses_full_galaxy_mesh() -> bool:
    """True when the full Galaxy host is visible to a mesh tutorial."""
    host_chip_count = _host_chip_count()
    return (
        _on_galaxy()
        and host_chip_count >= 32
        and _visible_chip_count() >= host_chip_count
    )


def _ci_tags(script: Path) -> set:
    """TTLANG_TUTORIAL_CI tag values declared in the script's first 80 lines."""
    tags = set()
    with script.open(encoding="utf-8") as handle:
        for _, line in zip(range(80), handle):
            stripped = line.strip()
            if stripped.startswith(CI_TAG_PREFIX):
                tags.add(stripped[len(CI_TAG_PREFIX) :].strip())
    return tags


def _discover_tutorials() -> list:
    scripts = []
    for subdir in TUTORIAL_SUBDIRS:
        directory = EXAMPLES_DIR / subdir
        if directory.is_dir():
            scripts.extend(
                sorted(p for p in directory.glob("*.py") if p.name != "__init__.py")
            )
    return scripts


def _tutorial_param(script: Path) -> "pytest.ParameterSet":
    tags = _ci_tags(script)
    opens_mesh = MESH_TAG in tags or REQUIRES_MESH_TAG in tags

    marks = [pytest.mark.timeout(PYTEST_TIMEOUT_BACKSTOP_SECONDS)]
    if opens_mesh:
        marks.append(pytest.mark.multi_device)
    if REQUIRES_MESH_TAG in tags:
        marks.append(
            pytest.mark.skipif(_host_chip_count() < 2, reason="requires >= 2 devices")
        )
    if script.name == ALL_REDUCE_TUTORIAL and _uses_full_galaxy_mesh():
        marks.append(
            pytest.mark.xfail(
                reason=(
                    "Full-Galaxy all_reduce/reduce_scatter fabric hang: "
                    "tt-lang#585, tt-metal#43749 / #41794; the killed run "
                    "wedges ethernet dispatch firmware"
                ),
                strict=True,
            )
        )

    return pytest.param(
        script, id=script.relative_to(EXAMPLES_DIR).as_posix(), marks=marks
    )


TUTORIALS = [_tutorial_param(script) for script in _discover_tutorials()]

if not TUTORIALS:
    raise RuntimeError(f"no tutorial scripts discovered under {EXAMPLES_DIR}")


@pytest.mark.parametrize("script", TUTORIALS)
def test_tutorial(script: Path):
    """Run a tutorial to completion and require a zero exit status."""
    process = subprocess.Popen([sys.executable, str(script)], cwd=REPO_ROOT)
    try:
        returncode = process.wait(timeout=SUBPROCESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        # SIGTERM first so tt-metal can close the device; SIGKILL only if it
        # outlasts the grace window.
        process.terminate()
        try:
            process.wait(timeout=SIGTERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    assert returncode == 0, f"{script.relative_to(REPO_ROOT)} exited with {returncode}"
