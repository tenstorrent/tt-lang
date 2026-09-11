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


def collect_provenance(sources):
    root = Path(__file__).resolve().parents[1]
    source_files = {Path(source).resolve() for source in sources}
    source_files.update((Path(__file__).resolve(), root / "benchmarks/common.py"))

    def git_output(*arguments):
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={root}", *arguments], cwd=root, text=True
        ).strip()

    metal_home = Path(os.environ["TT_METAL_HOME"])
    compiler_directory = Path(_ttlang.__file__).parent
    binaries = (
        Path(_ttlang.__file__),
        compiler_directory / "libTTLangPythonCAPI.so",
        metal_home / "lib/_ttnncpp.so",
        metal_home / "lib/libtt_metal.so",
    )
    return {
        "ttlang_revision": git_output("rev-parse", "HEAD"),
        "worktree_status": git_output("status", "--short"),
        "source_sha256": {
            str(Path(source).resolve().relative_to(root)): file_sha256(source)
            for source in sorted(source_files)
        },
        "binary_sha256": {str(binary): file_sha256(binary) for binary in binaries},
        "ttlang_module": ttl.__file__,
        "ttnn_module": ttnn.__file__,
        "dependency_pins": git_output(
            "ls-tree", "HEAD", "third-party/tt-metal", "third-party/llvm-project"
        ),
        "ttmetal_home": str(metal_home),
        "container_image": os.getenv("BENCHMARK_CONTAINER_IMAGE", "unrecorded"),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
    }
