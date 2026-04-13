# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
# tt-lang Python package setup

import glob
import os
import pathlib
import platform
import shutil
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def get_version_from_git():
    """Get version from git tags, matching cmake/modules/GetVersionFromGit.cmake."""
    try:
        tag = (
            subprocess.check_output(
                ["git", "describe", "--tags", "--match", "v[0-9]*", "--abbrev=0"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
            .lstrip("v")
        )
        commits = subprocess.check_output(
            ["git", "rev-list", f"v{tag}..HEAD", "--count"],
            stderr=subprocess.DEVNULL,
            text=True,
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

    def rmdir(self, _dir: pathlib.Path):
        if _dir.exists():
            shutil.rmtree(_dir)

    def in_ci(self) -> bool:
        return os.environ.get("IN_CIBW_ENV") == "ON"

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

    def build_(self, ext):
        build_lib = self.build_lib
        if not os.path.exists(build_lib):
            return

        extension_path = pathlib.Path(self.get_ext_fullpath(ext.name))
        print(f"Running cmake to install ttlang at {extension_path}")

        cwd = pathlib.Path().absolute()
        build_dir = cwd.parent / "build"

        install_dir = pathlib.Path(self.build_lib)

        if self.in_ci():
            install_dir = cwd / "build" / install_dir.name

        cmake_args = [
            "-G",
            "Ninja",
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_PREFIX=" + str(install_dir),
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang++",
        ]

        if not self.in_ci():
            cmake_args.extend(["-S", str(cwd.parent)])

        if self.in_ci():
            subprocess.run(
                " ".join(
                    [
                        "cd",
                        str(cwd.parent),
                        "&&",
                        ".",
                        "env/activate",
                        "&&",
                        "cmake",
                        *cmake_args,
                    ]
                ),
                shell=True,
                check=True,
            )
        else:
            self.spawn(["cmake", *cmake_args])

        self.spawn(["cmake", "--build", str(build_dir), "--", "TTLangPythonModules"])

        self.spawn(
            ["cmake", "--install", str(build_dir), "--component", "TTLangPythonWheel"]
        )

        # Post-install: strip binaries and fix RPATH for wheel distribution
        self._strip_binaries(install_dir)
        self._fix_rpath(install_dir)

        # Ensure config.py exists (CMake generates it, but provide a fallback)
        config_path = install_dir / "ttl" / "config.py"
        if not config_path.exists():
            version = get_version_from_git()
            config_path.write_text(
                f"# Auto-generated fallback (CMake config.py was missing)\n"
                f"HAS_TT_DEVICE = False\n"
                f'VERSION = "{version}"\n'
            )


version = get_version_from_git()
ttlang_c = TTLangExtension("ttl")

readme_path = pathlib.Path(__file__).absolute().parent.parent / "README.md"
with open(str(readme_path), "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="tt-lang",
    version=version,
    packages=[
        "ttl",
        "ttl._src",
        "ttl.pykernel",
        "ttl.pykernel._src",
        "ttl.sim",
        "ttl.utils",
    ],
    package_dir={
        "ttl": "ttl",
        "ttl._src": "ttl/_src",
        "ttl.pykernel": "pykernel",
        "ttl.pykernel._src": "pykernel/_src",
        "ttl.sim": "sim",
        "ttl.utils": "utils",
    },
    ext_modules=[ttlang_c],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    long_description=readme,
    long_description_content_type="text/markdown",
)
