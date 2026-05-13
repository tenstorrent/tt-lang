#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# tt-lang Python package setup. Project metadata lives in pyproject.toml; this
# file only provides the CMake-driven extension build.

import glob
import os
import pathlib
import platform
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist as _sdist


class NoSdist(_sdist):
    """Reject source distribution builds; tt-lang only ships pre-built wheels."""

    def run(self):
        raise SystemExit(
            "tt-lang does not publish source distributions. Build from a git "
            "checkout (https://github.com/tenstorrent/tt-lang) or install a "
            "pre-built wheel from PyPI."
        )


REPO_ROOT = pathlib.Path(__file__).resolve().parent


def _ttnn_requirement():
    """Build the platform-conditional ttnn requirement from the canonical
    tt-metal version.

    third-party/tt-metal-version holds a single tt-metal release tag (e.g.
    `v0.69.0`); the matching ttnn PyPI version is the tag minus the leading
    `v`. ttnn only publishes wheels for Linux x86_64 / aarch64, so the
    requirement is conditioned on those platforms via a PEP 508 marker;
    on macOS / Windows it is silently skipped (those platforms are
    sim-only via the separate `tt-lang-sim` package).
    """
    version_file = REPO_ROOT / "third-party" / "tt-metal-version"
    tag = version_file.read_text().strip()
    if not tag.startswith("v"):
        raise SystemExit(f"{version_file}: '{tag}' must start with 'v'")
    version = tag[1:]
    marker = (
        "sys_platform == 'linux' "
        "and (platform_machine == 'x86_64' or platform_machine == 'aarch64')"
    )
    return f"ttnn == {version} ; {marker}"


def _read_install_requires():
    """Read base runtime requirements from requirements-runtime.txt and
    append the dynamic ttnn requirement.
    """
    req_file = REPO_ROOT / "requirements-runtime.txt"
    requirements = []
    for line in req_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    requirements.append(_ttnn_requirement())
    return requirements


def get_version_from_git():
    """Get version from git tags, matching cmake/modules/GetVersionFromGit.cmake.

    Tag format: vMAJOR.MINOR.PATCH[+LOCAL]. Per PEP 440 the .devN segment must
    sit between the public release and the +local label, so the tag is split
    on '+' before the dev counter is inserted.

    Override mechanism: if TTLANG_PRETEND_VERSION is set in the environment, it
    is returned verbatim. Used by the publish-pypi workflow to stamp wheels
    built from a branch with a caller-supplied PEP 440 version (e.g. an rc/dev
    pre-release) when no matching git tag exists.
    """
    pretend = os.environ.get("TTLANG_PRETEND_VERSION", "").strip()
    if pretend:
        return pretend
    try:
        tag = (
            subprocess.check_output(
                ["git", "describe", "--tags", "--match", "v[0-9]*", "--abbrev=0"],
                stderr=subprocess.DEVNULL,
                text=True,
                cwd=str(REPO_ROOT),
            )
            .strip()
            .lstrip("v")
        )
        commits = subprocess.check_output(
            ["git", "rev-list", f"v{tag}..HEAD", "--count"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(REPO_ROOT),
        ).strip()
        base, sep, local = tag.partition("+")
        local_suffix = f"+{local}" if sep else ""
        if commits and commits != "0":
            return f"{base}.dev{commits}{local_suffix}"
        return f"{base}{local_suffix}"
    except Exception:
        return "0.2.0.dev0"


class TTLangExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            if "ttl" in ext.name:
                self.build_(ext)
            else:
                raise Exception("Unknown extension")

    def _strip_binaries(self, install_dir):
        """Strip debug symbols from .so/.dylib files to reduce wheel size."""
        if platform.system() == "Darwin":
            pattern = "**/*.dylib"
            strip_cmd = ["strip", "-x"]
        else:
            pattern = "**/*.so"
            strip_cmd = ["strip", "--strip-debug"]

        for lib_file in glob.glob(str(install_dir / pattern), recursive=True):
            self.spawn([*strip_cmd, lib_file])

    def _fix_rpath(self, install_dir):
        """Remove absolute build paths from RUNPATH, keeping only $ORIGIN."""
        if platform.system() == "Darwin":
            return  # macOS uses @loader_path, handled by CMake
        if not shutil.which("patchelf"):
            raise RuntimeError(
                "patchelf is required to sanitize RUNPATH on Linux but was not "
                "found on PATH. Install it (e.g. `pip install patchelf`) before "
                "building the wheel."
            )
        for so_file in glob.glob(str(install_dir / "**/*.so"), recursive=True):
            self.spawn(["patchelf", "--set-rpath", "$ORIGIN", so_file])

    def _sanitize_env_for_cmake(self):
        """Remove pip build-isolation env vars that break cmake's nested pip calls.

        When pip builds a wheel with PEP 517 isolation it sets PYTHONPATH
        to a temporary overlay directory.  This propagates into cmake's
        execute_process() calls and causes the toolchain-venv python to
        fail importing its own modules (including pip).  Clearing these
        vars is safe because cmake uses absolute paths to the toolchain
        python, which has its own site-packages.
        """
        for key in list(os.environ):
            if key.startswith("PIP_") or key in ("PYTHONNOUSERSITE", "PYTHONPATH"):
                del os.environ[key]

    def build_(self, ext):
        if self.inplace:
            # Editable install (`pip install -e .`): the cmake build is driven
            # by the developer's existing build/ tree, not by setup.py.
            return

        build_lib = pathlib.Path(self.build_lib)
        if not build_lib.exists():
            raise RuntimeError(
                f"build_lib {build_lib} does not exist; setuptools should have "
                f"created it before invoking build_ext."
            )

        extension_path = pathlib.Path(self.get_ext_fullpath(ext.name))
        print(f"Running cmake to install ttlang at {extension_path}")

        self._sanitize_env_for_cmake()

        source_dir = REPO_ROOT
        # Match the CMAKE_BINARY_DIR convention used by scripts/build-and-install.sh
        # so a developer with build-docker/ or build-debug/ can build the wheel
        # against their existing tree.
        build_dir_setting = os.environ.get("CMAKE_BINARY_DIR", "build")
        build_dir = pathlib.Path(build_dir_setting)
        if not build_dir.is_absolute():
            build_dir = source_dir / build_dir
        install_dir = build_lib

        # Configure only when no prior cmake configuration exists.  Local
        # developer builds already have a configured build/ directory; re-
        # running configure just to change the install prefix is unnecessary
        # and can fail when the cached toolchain venv lacks pip.
        cmake_cache = build_dir / "CMakeCache.txt"
        if not cmake_cache.exists():
            cmake_args = [
                "cmake",
                "-G",
                "Ninja",
                "-S",
                str(source_dir),
                "-B",
                str(build_dir),
                "-DCMAKE_BUILD_TYPE=Release",
            ]

            # Forward toolchain env vars as cmake -D flags.  cmake
            # option() does not read the environment, so the vars must be
            # forwarded explicitly.
            if os.environ.get("TTLANG_USE_TOOLCHAIN") == "ON":
                cmake_args.append("-DTTLANG_USE_TOOLCHAIN=ON")
                toolchain_dir = os.environ.get("TTLANG_TOOLCHAIN_DIR", "")
                if toolchain_dir:
                    cmake_args.append(f"-DTTLANG_TOOLCHAIN_DIR={toolchain_dir}")

            # Forward CC/CXX as cmake -D flags.  CMakeLists.txt defaults
            # to clang before project(), which runs before cmake reads the
            # CC/CXX env vars — so the env vars alone have no effect.
            cc = os.environ.get("CC")
            cxx = os.environ.get("CXX")
            if cc:
                cmake_args.append(f"-DCMAKE_C_COMPILER={cc}")
            if cxx:
                cmake_args.append(f"-DCMAKE_CXX_COMPILER={cxx}")

            self.spawn(cmake_args)

        self.spawn(
            ["cmake", "--build", str(build_dir), "--target", "TTLangPythonModules"]
        )

        # Use --prefix to override the install location at install time.
        # This avoids reconfiguring the build just to change
        # CMAKE_INSTALL_PREFIX.
        self.spawn(
            [
                "cmake",
                "--install",
                str(build_dir),
                "--component",
                "TTLangPythonWheel",
                "--prefix",
                str(install_dir),
            ]
        )

        # Post-install: strip binaries and fix RPATH for wheel distribution
        self._strip_binaries(install_dir)
        self._fix_rpath(install_dir)


ttlang_c = TTLangExtension("ttl")

sys.path.insert(0, str(REPO_ROOT / "packaging"))
from rewrite_readme import absolutize_readme_images, ref_for_version  # noqa: E402

_version = get_version_from_git()

readme_path = REPO_ROOT / "README.md"
with open(str(readme_path), "r", encoding="utf-8") as readme_file:
    readme = absolutize_readme_images(
        readme_file.read(), ref_for_version(_version), REPO_ROOT
    )

setup(
    version=_version,
    install_requires=_read_install_requires(),
    packages=[
        "ttl",
        "ttl._src",
        "ttl._setup",
        "ttl.pykernel",
        "ttl.pykernel._src",
        "ttl.sim",
        "ttl.tutorials",
        "ttl.tutorials.elementwise",
        "ttl.tutorials.matmul",
        "ttl.tutorials.broadcast",
        "ttl.utils",
        "sim_stats",
    ],
    package_dir={
        "ttl": "python/ttl",
        "ttl._src": "python/ttl/_src",
        "ttl._setup": "python/ttl/_setup",
        "ttl.pykernel": "python/pykernel",
        "ttl.pykernel._src": "python/pykernel/_src",
        "ttl.sim": "python/sim",
        "ttl.tutorials": "python/ttl/tutorials",
        "ttl.tutorials.elementwise": "examples/elementwise-tutorial",
        "ttl.tutorials.matmul": "examples/matmul-tutorial",
        "ttl.tutorials.broadcast": "examples/tutorial",
        "ttl.utils": "python/utils",
        "sim_stats": "python/sim_stats",
    },
    ext_modules=[ttlang_c],
    cmdclass={"build_ext": CMakeBuild, "sdist": NoSdist},
    zip_safe=False,
    long_description=readme,
    long_description_content_type="text/markdown",
)
