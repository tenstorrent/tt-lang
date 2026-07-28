# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python -m pytest %s -v

"""Off-device unit tests for hang detection configuration and collection.

Everything here is the half of the feature that runs without hardware: which
tt-metal variables get armed, what the collector records about each compiled
program, and how it finds kernel ELFs in the tt-metal cache. The device half
(reading PCs, recovering) needs a real hang and is exercised by hand.
"""

import os
from types import SimpleNamespace

import pytest

from ttl import hang, hang_collect

METAL_VARS = (
    "TT_METAL_FORCE_REINIT",
    "TT_METAL_OPERATION_TIMEOUT_SECONDS",
    "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE",
)


@pytest.fixture
def clean_env(monkeypatch):
    """A process env with every variable this feature touches unset."""
    for name in METAL_VARS + (
        hang.MODE_ENV,
        hang.TIMEOUT_ENV,
        hang.FORCE_REINIT_ENV,
        hang.LAUNCH_ENV,
        hang.DEVICES_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _core_range_set(ranges):
    """Stand-in for ttnn.CoreRangeSet, so these tests need no ttnn."""
    return SimpleNamespace(
        ranges=lambda: [
            SimpleNamespace(
                start=SimpleNamespace(x=sx, y=sy), end=SimpleNamespace(x=ex, y=ey)
            )
            for (sx, sy), (ex, ey) in ranges
        ]
    )


def test_default_mode_is_deep(clean_env):
    assert hang.mode() == hang.MODE_DEEP


def test_unknown_mode_is_rejected(clean_env):
    clean_env.setenv(hang.MODE_ENV, "sideways")
    with pytest.raises(ValueError, match="sideways"):
        hang.mode()


def test_timeout_default_and_override(clean_env):
    assert hang.timeout_seconds() == hang.DEFAULT_TIMEOUT_SECONDS
    clean_env.setenv(hang.TIMEOUT_ENV, "12.5")
    assert hang.timeout_seconds() == 12.5


def test_non_positive_timeout_is_rejected(clean_env):
    clean_env.setenv(hang.TIMEOUT_ENV, "0")
    with pytest.raises(ValueError, match=hang.MODE_ENV):
        hang.timeout_seconds()


def test_configure_arms_metal(clean_env):
    hang.configure_metal_env()
    assert os.environ["TT_METAL_FORCE_REINIT"] == "1"
    assert os.environ["TT_METAL_OPERATION_TIMEOUT_SECONDS"] == str(
        hang.DEFAULT_TIMEOUT_SECONDS
    )
    command = os.environ["TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"]
    assert command.endswith("hang_collect.py")


def test_explicit_metal_value_wins(clean_env):
    clean_env.setenv("TT_METAL_OPERATION_TIMEOUT_SECONDS", "5.0")
    hang.configure_metal_env()
    assert os.environ["TT_METAL_OPERATION_TIMEOUT_SECONDS"] == "5.0"


def test_off_leaves_the_timeout_alone_but_still_forces_reinit(clean_env):
    clean_env.setenv(hang.MODE_ENV, hang.MODE_OFF)
    hang.configure_metal_env()
    assert os.environ["TT_METAL_FORCE_REINIT"] == "1"
    assert "TT_METAL_OPERATION_TIMEOUT_SECONDS" not in os.environ


def test_force_reinit_opt_out(clean_env):
    clean_env.setenv(hang.FORCE_REINIT_ENV, "0")
    hang.configure_metal_env()
    assert "TT_METAL_FORCE_REINIT" not in os.environ


def test_program_registry_round_trip(clean_env, tmp_path, monkeypatch):
    clean_env.setenv(hang.DIR_ENV, str(tmp_path))
    monkeypatch.setattr(hang, "_registry_started", False)
    monkeypatch.setattr(hang, "_launch_ring", [])

    add = [("/tmp/u/ttlang_kernel_add__brisc_aaaa.cpp", "noc")]
    mul = [
        ("/tmp/u/ttlang_kernel_mul__trisc_bbbb.cpp", "compute"),
        ("/tmp/u/ttlang_kernel_mul__brisc_cccc.cpp", "noc"),
    ]
    grid = _core_range_set([((0, 0), (1, 1))])
    hang.note_program(11, add, grid)
    hang.note_program(22, mul, grid)

    programs = hang_collect.load_programs(tmp_path)
    assert [p["key"] for p in programs] == [
        "ttlang_kernel_add__brisc_aaaa",
        "ttlang_kernel_mul__trisc_bbbb",
    ]
    assert programs[0]["cores"] == [[[0, 0], [1, 1]]]

    # The launch ring is what the collector prefers, most recent first.
    hang.note_launch("ttlang_kernel_add__brisc_aaaa")
    hang.note_launch("ttlang_kernel_mul__trisc_bbbb")
    selected = hang_collect.select_programs(programs)
    assert [p["key"] for p in selected] == [
        "ttlang_kernel_mul__trisc_bbbb",
        "ttlang_kernel_add__brisc_aaaa",
    ]
    assert hang_collect.select_cores(selected) == [(0, 0), (1, 0), (0, 1), (1, 1)]


def test_launch_ring_is_bounded_and_skips_repeats(clean_env, monkeypatch):
    monkeypatch.setattr(hang, "_launch_ring", [])
    hang.note_launch("a")
    hang.note_launch("a")
    assert os.environ[hang.LAUNCH_ENV] == "a"
    for index in range(hang.LAUNCH_RING * 2):
        hang.note_launch(f"k{index}")
    assert len(os.environ[hang.LAUNCH_ENV].split(",")) == hang.LAUNCH_RING


def _build_fake_cache(tmp_path):
    """A tt-metal kernel cache laid out the way tt-metal lays one out."""
    for risc, stem in (
        ("brisc", "ttlang_kernel_add__brisc_aaaa"),
        ("trisc0", "ttlang_kernel_mul__trisc_bbbb"),
    ):
        directory = tmp_path / "buildkey" / "kernels" / stem / "9999" / risc
        directory.mkdir(parents=True)
        (directory / f"{risc}.elf").write_bytes(b"elf")
        (directory / f"{risc}.elf.xip.elf").write_bytes(b"skipped")
    firmware = tmp_path / "buildkey" / "firmware" / "ncrisc"
    firmware.mkdir(parents=True)
    (firmware / "ncrisc.elf").write_bytes(b"fw")
    (firmware / "ncrisc_weakened.elf").write_bytes(b"skipped")

    return [
        {"kernels": [{"path": "/tmp/u/ttlang_kernel_add__brisc_aaaa.cpp"}]},
        {"kernels": [{"path": "/tmp/u/ttlang_kernel_mul__trisc_bbbb.cpp"}]},
    ]


def test_elf_discovery_from_generated_source_names(tmp_path):
    """Kernel ELFs are found by generated source stem, ignoring xip and weakened."""
    programs = _build_fake_cache(tmp_path)
    found = hang_collect.kernel_elfs(programs, str(tmp_path))
    assert sorted(found) == ["brisc", "trisc0"]
    assert all(len(paths) == 1 for paths in found.values())
    assert found["brisc"][0].endswith("brisc/brisc.elf")

    assert list(hang_collect.firmware_elfs(str(tmp_path))) == ["ncrisc"]


def test_devices_default_to_the_first_only(clean_env):
    assert hang_collect.select_devices() == [0]
    clean_env.setenv(hang.DEVICES_ENV, "0,3")
    assert hang_collect.select_devices() == [0, 3]


def test_cache_root_is_the_parent_of_the_cache(clean_env, tmp_path):
    """TT_METAL_CACHE names the parent: tt-metal appends the tt-metal-cache component.

    Getting this wrong finds no ELF at all, so nothing symbolizes, which is the
    failure that looks like the feature being broken rather than misconfigured.
    """
    cache = tmp_path / "tt-metal-cache"
    _build_fake_cache(cache)
    clean_env.setenv("TT_METAL_CACHE", str(tmp_path))
    report = hang_collect.Report()
    assert hang_collect.resolve_cache_root(report) == cache

    # A root pointed straight at the cache also resolves.
    clean_env.setenv("TT_METAL_CACHE", str(cache))
    assert hang_collect.resolve_cache_root(hang_collect.Report()) == cache


def test_unbuilt_cache_root_is_reported_not_guessed(clean_env, tmp_path):
    clean_env.setenv("TT_METAL_CACHE", str(tmp_path))
    clean_env.setenv("HOME", str(tmp_path))
    report = hang_collect.Report()
    assert hang_collect.resolve_cache_root(report) is None
    assert "Tried:" in report.text()


@pytest.mark.parametrize("retired", hang.RETIRED_MODES)
def test_retired_modes_are_not_implemented(clean_env, retired):
    clean_env.setenv(hang.MODE_ENV, retired)
    with pytest.raises(NotImplementedError, match=retired):
        hang.mode()


def test_recover_mode_never_stops_the_process(clean_env):
    """recover hands the hang to Python, which needs the process alive."""
    report = hang_collect.Report()
    assert hang_collect.stop_target(report, hang.MODE_RECOVER) == 0


def test_stopping_can_be_turned_off(clean_env):
    clean_env.setenv(hang_collect.KILL_ENV, "0")
    report = hang_collect.Report()
    assert hang_collect.stop_target(report, hang.MODE_FAST) == 0
    assert hang_collect.KILL_ENV in report.text()


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="needs /proc")
def test_hung_pid_skips_the_shell():
    """std::system leaves a shell in between, which must not be the target."""
    pid = hang_collect.hung_pid()
    assert pid > 0
    comm = open(f"/proc/{pid}/comm").read().strip()
    assert comm not in ("sh", "bash", "dash", "zsh")
