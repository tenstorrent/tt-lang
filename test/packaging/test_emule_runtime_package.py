# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Check the emulator image's native runtime artifact layout."""

from pathlib import Path
import os
import subprocess

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / ".github/containers/package-emule-runtime.sh"
)


def write(root, name, content="artifact"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def runtime(tmp_path):
    source = tmp_path / "metal source"
    for name in (
        "LICENSE",
        "NOTICE",
        "runtime/hw/toolchain/blackhole/link.ld",
        "runtime/hw/lib/blackhole/startup.o",
        "runtime/sfpi/compiler/lib/libgcc.a",
        "tt_metal/hw/inc/kernel.h",
        "tt_metal/pre-compiled/firmware/brisc.elf",
        "tt_stl/tt_stl/span.hpp",
        "ttnn/cpp/ttnn/operations/kernel.cpp",
        "ttnn/ttnn/__init__.py",
        "build_emule/generated/tt_metal/impl/version.hpp",
        "build_emule/tt_metal/libtt_metal.so",
        "build_emule/tt_stl/libtt_stl.so",
        "build_emule/ttnn/_ttnn.so",
        "build_emule/ttnn/_ttnncpp.so",
        "build_emule/_deps/fmt-build/libfmt.so.11",
        "models/common/utility_functions.py",
        "models/common/tensor_utils.py",
        "tools/tracy/__init__.py",
    ):
        write(source, name)
    library_dir = source / "build_emule/lib"
    library_dir.mkdir()
    (library_dir / "libtt_metal.so").symlink_to("../tt_metal/libtt_metal.so")
    (source / "ttnn/ttnn/_ttnn.so").symlink_to(source / "build_emule/ttnn/_ttnn.so")
    return source, tmp_path / "packaged runtime"


def package(source, destination):
    return subprocess.run(
        ["bash", str(SCRIPT), str(source), str(destination)],
        capture_output=True,
        text=True,
    )


def test_keeps_jit_artifacts_and_library_layout(runtime):
    source, destination = runtime
    result = package(source, destination)
    assert result.returncode == 0, result.stderr
    for name in (
        "runtime/hw/toolchain/blackhole/link.ld",
        "runtime/hw/lib/blackhole/startup.o",
        "runtime/sfpi/compiler/lib/libgcc.a",
        "tt_metal/pre-compiled/firmware/brisc.elf",
        "ttnn/cpp/ttnn/operations/kernel.cpp",
        "build_emule/generated/tt_metal/impl/version.hpp",
        "build_emule/_deps/fmt-build/libfmt.so.11",
        "models/common/utility_functions.py",
        "models/common/tensor_utils.py",
        "tools/tracy/__init__.py",
        "LICENSE",
        "NOTICE",
    ):
        assert (destination / name).read_text() == "artifact"
    python_extension = destination / "ttnn/ttnn/_ttnn.so"
    assert python_extension.is_symlink()
    assert python_extension.resolve() == destination / "build_emule/ttnn/_ttnn.so"
    metal_library = destination / "build_emule/lib/libtt_metal.so"
    assert metal_library.is_symlink()
    assert (
        metal_library.resolve() == destination / "build_emule/tt_metal/libtt_metal.so"
    )


def test_excludes_build_intermediates_and_unrelated_sources(runtime):
    source, destination = runtime
    omitted = (
        ".git/config",
        ".cpmcache/boost/source.cpp",
        "tt_metal/third_party/umd/.git",
        "tt_metal/tests/unit.cpp",
        "tt_metal/docs/guide.md",
        "ttnn/ttnn/__pycache__/module.pyc",
        "build_emule/CMakeFiles/cache.h",
        "build_emule/ttnn/CMakeFiles/kernel.cpp.o",
        "build_emule/ttnn/libintermediate.a",
        "build_emule/tt-metal-cache/kernel.cpp",
        "build_emule/compile_commands.json",
        "models/demos/model.py",
        "tools/triage/__init__.py",
    )
    for name in omitted:
        write(source, name)
    result = package(source, destination)
    assert result.returncode == 0, result.stderr
    for name in omitted:
        assert not (destination / name).exists(), name


def test_rejects_missing_native_extension(runtime):
    source, destination = runtime
    (source / "build_emule/ttnn/_ttnn.so").unlink()
    result = package(source, destination)
    assert result.returncode != 0
    assert "Required emulator runtime artifact is missing" in result.stderr
    assert not destination.exists()


def test_preserves_dependency_notices_without_sources(runtime):
    source, destination = runtime
    licenses = (
        ".cpmcache/boost/version/LICENSE_1_0.txt",
        ".cpmcache/protobuf/version/COPYING",
        ".cpmcache/fmt/version/license.txt",
        "third_party/library/NOTICE",
        "third_party/library/Copyright.md",
    )
    for name in licenses:
        write(source, name, "license text")
    omitted = (
        ".cpmcache/boost/version/src/library.cpp",
        ".cpmcache/boost/version/.git/LICENSE",
        "third_party/library/src/library.cpp",
    )
    for name in omitted:
        write(source, name)
    license_link = source / ".cpmcache/fmt/version/NOTICE"
    license_link.symlink_to("license.txt")

    result = package(source, destination)
    assert result.returncode == 0, result.stderr
    notices = destination / "third-party-licenses"
    for name in licenses:
        assert (notices / name).read_text() == "license text"
    for name in omitted:
        assert not (notices / name).exists()
    packaged_link = notices / license_link.relative_to(source)
    assert not packaged_link.is_symlink()
    assert packaged_link.read_text() == "license text"


def test_preserves_existing_destination(runtime):
    source, destination = runtime
    sentinel = write(destination, "keep.txt", "keep")
    result = package(source, destination)
    assert result.returncode != 0
    assert "must not already exist" in result.stderr
    assert sentinel.read_text() == "keep"


def test_rejects_dependency_notice_outside_source(runtime):
    source, destination = runtime
    outside = write(source.parent, "outside-license", "outside")
    dependency = source / ".cpmcache/dependency"
    dependency.mkdir(parents=True)
    (dependency / "LICENSE").symlink_to(outside)
    result = package(source, destination)
    assert result.returncode != 0
    assert "Dependency notice escapes" in result.stderr
    assert not (destination / "third-party-licenses").exists()


def test_rejects_destination_inside_source(runtime):
    source, _ = runtime
    result = package(source, source / "packaged")
    assert result.returncode != 0
    assert "outside the source tree" in result.stderr
    assert not (source / "packaged").exists()


def test_rejects_missing_destination_parent_without_creating_it(
    runtime, tmp_path, monkeypatch
):
    source, destination = runtime
    mkdir_log = tmp_path / "mkdir.log"
    mkdir_spy = write(
        tmp_path,
        "bin/mkdir",
        '#!/bin/sh\nprintf "%s\\n" "$*" > "$EMULE_MKDIR_LOG"\nexit 1\n',
    )
    mkdir_spy.chmod(0o755)
    monkeypatch.setenv("PATH", str(mkdir_spy.parent), prepend=os.pathsep)
    monkeypatch.setenv("EMULE_MKDIR_LOG", str(mkdir_log))

    result = package(source, destination / "nested")
    assert result.returncode != 0
    assert not mkdir_log.exists(), "Invalid destination reached directory creation"
    assert not destination.exists()


@pytest.mark.parametrize("outside_exists", [True, False])
def test_rejects_noncontained_symlinks(runtime, outside_exists):
    source, destination = runtime
    outside = source.parent / "outside.h"
    if outside_exists:
        outside.write_text("outside")
    (source / "tt_metal/hw/inc/escape.h").symlink_to(outside)
    result = package(source, destination)
    assert result.returncode != 0
    assert "symlink" in result.stderr
