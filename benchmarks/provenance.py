# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Source and installed binary identities for reproducible benchmark reports."""

import hashlib
import os
import platform
import subprocess
from pathlib import Path

import ttl
import ttnn
from ttl._mlir_libs import _ttlang


def file_sha256(filename):
    with Path(filename).open("rb") as source_file:
        return hashlib.file_digest(source_file, "sha256").hexdigest()


def loaded_library_path(filename, maps_file=Path("/proc/self/maps")):
    matches = set()
    for line in maps_file.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        library_path = Path(fields[-1])
        if library_path.name == filename:
            matches.add(library_path.resolve())
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one loaded {filename}, found {len(matches)}: "
            f"{sorted(str(match) for match in matches)}"
        )
    return matches.pop()


def collect_provenance(sources, *, ttmetal_source_root=None):
    root = Path(__file__).resolve().parents[1]
    source_files = {Path(source).resolve() for source in sources}
    source_files.update((Path(__file__).resolve(), root / "benchmarks/common.py"))

    def git_output(repository, *arguments):
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={repository}", *arguments],
            cwd=repository,
            text=True,
        ).strip()

    metal_home = Path(os.environ["TT_METAL_HOME"])
    metal_runtime_root = Path(
        os.environ.get("TT_METAL_RUNTIME_ROOT", root / "third-party/tt-metal")
    ).resolve()
    metal_source_root = Path(ttmetal_source_root or metal_runtime_root).resolve()
    compiler_directory = Path(_ttlang.__file__).parent
    binaries = (
        Path(_ttlang.__file__),
        compiler_directory / "libTTLangPythonCAPI.so",
        Path(ttnn._ttnn.__file__),
        loaded_library_path("_ttnncpp.so"),
        loaded_library_path("libtt_metal.so"),
    )
    return {
        "ttlang_revision": git_output(root, "rev-parse", "HEAD"),
        "worktree_status": git_output(root, "status", "--short"),
        "source_sha256": {
            str(Path(source).resolve().relative_to(root)): file_sha256(source)
            for source in sorted(source_files)
        },
        "binary_sha256": {str(binary): file_sha256(binary) for binary in binaries},
        "ttlang_module": ttl.__file__,
        "ttnn_module": ttnn.__file__,
        "dependency_pins": git_output(
            root, "ls-tree", "HEAD", "third-party/tt-metal", "third-party/llvm-project"
        ),
        "ttmetal_revision": git_output(metal_source_root, "rev-parse", "HEAD"),
        "ttmetal_home": str(metal_home),
        "ttmetal_runtime_root": str(metal_runtime_root),
        "ttmetal_source_root": str(metal_source_root),
        "container_image": os.getenv("BENCHMARK_CONTAINER_IMAGE", "unrecorded"),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
    }
