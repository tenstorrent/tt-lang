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

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


REPO_ROOT = pathlib.Path(__file__).resolve().parent


def get_version_from_git():
    """Get version from git tags, matching cmake/modules/GetVersionFromGit.cmake."""
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
        if commits and commits != "0":
            return f"{tag}.dev{commits}"
        return tag
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
            try:
                self.spawn([*strip_cmd, lib_file])
            except Exception as exc:
                print(f"Warning: failed to strip {lib_file}: {exc}")

    def _fix_rpath(self, install_dir):
        """Remove absolute build paths from RUNPATH, keeping only $ORIGIN."""
        if platform.system() == "Darwin":
            return  # macOS uses @loader_path, handled by CMake
        if not shutil.which("patchelf"):
            print("Warning: patchelf not found, skipping RPATH sanitization")
            return
        for so_file in glob.glob(str(install_dir / "**/*.so"), recursive=True):
            try:
                self.spawn(["patchelf", "--set-rpath", "$ORIGIN", so_file])
            except Exception as exc:
                print(f"Warning: failed to fix RPATH for {so_file}: {exc}")

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
        build_lib = self.build_lib
        if not os.path.exists(build_lib):
            return

        extension_path = pathlib.Path(self.get_ext_fullpath(ext.name))
        print(f"Running cmake to install ttlang at {extension_path}")

        self._sanitize_env_for_cmake()

        source_dir = REPO_ROOT
        build_dir = source_dir / "build"

        install_dir = pathlib.Path(self.build_lib)

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

        # The cmake install copies build/python_packages/ which includes a
        # ttl/sim symlink (from TTLangSimPackage).  setuptools' build_py
        # step already copied the real sim/ directory into install_dir,
        # so remove it before the cmake install to avoid a conflict.
        sim_dir = install_dir / "ttl" / "sim"
        if sim_dir.is_dir() and not sim_dir.is_symlink():
            shutil.rmtree(sim_dir)

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

readme_path = REPO_ROOT / "README.md"
with open(str(readme_path), "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    version=get_version_from_git(),
    packages=[
        "ttl",
        "ttl._src",
        "ttl.pykernel",
        "ttl.pykernel._src",
        "ttl.sim",
        "ttl.utils",
    ],
    package_dir={
        "ttl": "python/ttl",
        "ttl._src": "python/ttl/_src",
        "ttl.pykernel": "python/pykernel",
        "ttl.pykernel._src": "python/pykernel/_src",
        "ttl.sim": "python/sim",
        "ttl.utils": "python/utils",
    },
    ext_modules=[ttlang_c],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    long_description=readme,
    long_description_content_type="text/markdown",
)
