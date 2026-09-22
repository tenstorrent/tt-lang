# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from benchmarks.provenance import (
    dependency_revision,
    loaded_library_path,
    resolve_ttmetal_revision,
)


def test_loaded_library_path(tmp_path):
    maps_file = tmp_path / "maps"
    maps_file.write_text(
        "7f000000-7f001000 r-xp 00000000 00:00 0 /build/libtt_metal.so\n"
        "7f001000-7f002000 r--p 00001000 00:00 0 /build/libtt_metal.so\n"
    )

    assert loaded_library_path("libtt_metal.so", maps_file) == Path(
        "/build/libtt_metal.so"
    )


def test_loaded_library_path_rejects_ambiguous_libraries(tmp_path):
    maps_file = tmp_path / "maps"
    maps_file.write_text(
        "7f000000-7f001000 r-xp 00000000 00:00 0 /first/_ttnncpp.so\n"
        "7f001000-7f002000 r-xp 00000000 00:00 0 /second/_ttnncpp.so\n"
    )

    with pytest.raises(RuntimeError, match="expected one loaded _ttnncpp.so, found 2"):
        loaded_library_path("_ttnncpp.so", maps_file)


def test_dependency_revision_extracts_submodule_commit():
    tree = (
        "160000 commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\tthird-party/llvm-project\n"
        "160000 commit bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\tthird-party/tt-metal"
    )

    assert dependency_revision(tree, "third-party/tt-metal") == "b" * 40


def test_ttmetal_revision_falls_back_to_dependency_pin():
    revision, source = resolve_ttmetal_revision(None, None, "c" * 40)

    assert revision == "c" * 40
    assert source == "ttlang_dependency_pin"
