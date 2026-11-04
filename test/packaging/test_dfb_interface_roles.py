# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Run actual DFB interface and geometry updates without synchronization."""

from pathlib import Path
import re
import shutil
import subprocess

import pytest

from conftest import REPO_ROOT


ROLE_CASES = [
    pytest.param(["COMPILE_FOR_DM=0"], (True, True, False, False), id="dm0"),
    pytest.param(["COMPILE_FOR_DM=1"], (True, True, False, True), id="dm1"),
    pytest.param(["TRISC_UNPACK=1"], (True, False, False, False), id="unpack"),
    pytest.param(["TRISC_PACK=1"], (False, True, True, False), id="pack"),
    pytest.param(
        ["TRISC_UNPACK=1", "TRISC_MATH=1", "TRISC_PACK=1"],
        (True, True, True, False),
        id="fused-compute",
    ),
    pytest.param(["TRISC_MATH=1"], (False, False, False, False), id="math-only"),
]


@pytest.mark.parametrize("role_defines, expected_flags", ROLE_CASES)
@pytest.mark.parametrize("emule_pool", [False, True], ids=["hardware", "emule"])
def test_dfb_interface_role_updates(
    tmp_path: Path, role_defines, expected_flags, emule_pool: bool
):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    source = Path(__file__).parent / "Inputs/dfb_interface_roles.cpp"
    flags = [
        "-std=c++17",
        "-O2",
        "-I",
        str(REPO_ROOT / "include"),
        *[f"-D{definition}" for definition in role_defines],
        f"-DTEST_ACTIVE={int(any(expected_flags))}",
        f"-DTEST_SHARED_GEOMETRY={int(emule_pool and expected_flags[3])}",
        *[
            f"-DTEST_{name}={int(value)}"
            for name, value in zip(
                ("READ", "WRITE", "WRITE_TILE", "RESET_COUNTERS"), expected_flags
            )
        ],
    ]
    if emule_pool:
        # Only DM1 declares the hook, so other roles must compile without it.
        flags.append("-DTT_EMULE_USE_L1_POOL=1")
    preprocessed = subprocess.run(
        [compiler, *flags, "-E", "-P", str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert preprocessed.returncode == 0, preprocessed.stderr
    helper_body = preprocessed.stdout.split(
        "void applyReconfiguration(uint32_t configurationAddress)", 1
    )[1].split("\n}", 1)[0]
    selected_flags = dict(
        re.findall(r"constexpr bool (\w+) = (true|false);", helper_body)
    )
    names = (
        "updateReadPointer",
        "updateWritePointer",
        "updateWriteTilePointer",
        "resetStreamCounters",
    )
    if any(expected_flags):
        assert selected_flags == {
            name: str(value).lower() for name, value in zip(names, expected_flags)
        }
    else:
        assert selected_flags == {}

    # Use the header's selected template arguments while testing its updates.
    flags.extend(
        f"-DTEST_RECONFIG_{name}={selected_flags.get(parameter, 'false')}"
        for name, parameter in zip(
            ("READ", "WRITE", "WRITE_TILE", "RESET_COUNTERS"), names
        )
    )
    executable = tmp_path / "dfb_interface_roles"
    compiled = subprocess.run(
        [compiler, *flags, str(source), "-o", str(executable)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
