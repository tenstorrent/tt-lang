# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Keep shared compiler/test requirements usable by developer containers."""

from conftest import REPO_ROOT


def requirements(path):
    result = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            result.update(requirements(path.parent / line[3:].strip()))
        else:
            result.add(line)
    return result


def test_development_dependency_split_preserves_packages():
    test_packages = {
        "lit",
        "packaging",
        "pytest-order>=1.0.0",
        "pytest-rerunfailures>=12.0",
        "pytest-timeout>=2.0",
        "pytest-xdist>=3.0",
    }
    compiler_packages = requirements(REPO_ROOT / "requirements.txt")
    assert requirements(REPO_ROOT / "requirements-test.txt") == (
        compiler_packages | test_packages
    )
    assert requirements(REPO_ROOT / "dev-requirements.txt") == (
        compiler_packages
        | test_packages
        | requirements(REPO_ROOT / "docs/requirements.txt")
        | {"black", "pre-commit", "pyright"}
    )


def test_ird_copies_test_requirements_and_uplift_tracks_them():
    dockerfile = (REPO_ROOT / ".github/containers/Dockerfile").read_text()
    requirement_copy = next(
        line.split()
        for line in dockerfile.splitlines()
        if line.startswith("COPY ") and "dev-requirements.txt" in line
    )
    assert {
        "requirements.txt",
        "requirements-runtime.txt",
        "requirements-test.txt",
        "dev-requirements.txt",
    }.issubset(requirement_copy)
    assert "/tmp/requirements-test.txt" in dockerfile
    uplift = (REPO_ROOT / ".github/scripts/uplift-paths.sh").read_text()
    assert "    requirements-test.txt\n" in uplift
