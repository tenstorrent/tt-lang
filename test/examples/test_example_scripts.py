# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Execute each example script that hardware CI runs as an isolated subprocess.

Discovery and tags mirror .github/scripts/compile-and-run-examples.sh: the
top-level examples/*.py and every *.py under examples/tutorial and
examples/errors that contain ``@ttl.operation``, with tags read from a script's
first 80 lines. skip-compiler skips a script; xfail-compiler expects a nonzero
exit, and an unexpected pass fails. A ``requires-multi-device`` tutorial tag
marks the script multi_device and skips it, since this suite runs only on the
single-device emulator profile; hardware CI keeps using the shell script.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
SCAN_LINES = 80
SKIP_TAG = "# TTLANG_HARDWARE_CI: skip-compiler"
XFAIL_TAG = "# TTLANG_HARDWARE_CI: xfail-compiler"
REQUIRES_MESH_TAG = "# TTLANG_TUTORIAL_CI: requires-multi-device"
TIMEOUT_SECONDS = int(os.environ.get("TTLANG_EXAMPLE_TIMEOUT_SECONDS", "300"))
# SIGTERM-to-SIGKILL grace on timeout, so tt-metal can close the device.
SIGTERM_GRACE_SECONDS = 10


def discover_examples():
    """Repository-relative script paths, as the shell runner discovers them."""
    candidates = list(EXAMPLES_DIR.glob("*.py"))
    for subdir in ("tutorial", "errors"):
        candidates += (EXAMPLES_DIR / subdir).rglob("*.py")
    return sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in candidates
        if path.is_file() and b"@ttl.operation" in path.read_bytes()
    )


def tags(script):
    """The runner tags declared in the script's first lines."""
    with (REPO_ROOT / script).open(encoding="utf-8") as handle:
        head = [line for _, line in zip(range(SCAN_LINES), handle)]
    return {
        tag
        for tag in (SKIP_TAG, XFAIL_TAG, REQUIRES_MESH_TAG)
        if any(tag in line for line in head)
    }


def example_param(script):
    found = tags(script)
    # Above the child's worst-case lifetime so the subprocess handles timeouts;
    # pytest-timeout's thread method would abort the whole session.
    marks = [pytest.mark.timeout(TIMEOUT_SECONDS + SIGTERM_GRACE_SECONDS + 60)]
    if REQUIRES_MESH_TAG in found:
        marks += [pytest.mark.multi_device, pytest.mark.skip(reason="needs a mesh")]
    elif SKIP_TAG in found:
        marks.append(pytest.mark.skip(reason=SKIP_TAG.lstrip("# ")))
    elif XFAIL_TAG in found:
        marks.append(pytest.mark.xfail(reason=XFAIL_TAG.lstrip("# "), strict=True))
    return pytest.param(script, id=script.removeprefix("examples/"), marks=marks)


EXAMPLES = [example_param(script) for script in discover_examples()]

if not EXAMPLES:
    raise RuntimeError(f"no example scripts discovered under {EXAMPLES_DIR}")


@pytest.mark.parametrize("script", EXAMPLES)
def test_example(script):
    """Run an example to completion and require a zero exit status."""
    process = subprocess.Popen([sys.executable, script], cwd=REPO_ROOT)
    try:
        returncode = process.wait(timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=SIGTERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    assert returncode == 0, f"{script} exited with {returncode}"
