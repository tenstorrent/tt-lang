# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
# tt-lang Python package setup

import os
import pathlib
import shutil
import subprocess
from datetime import datetime

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

readme = None


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

    def build_(self, ext):
        build_lib = self.build_lib
        if not os.path.exists(build_lib):
            # Might be an editable install or something else
            return

        extension_path = pathlib.Path(self.get_ext_fullpath(ext.name))
        print(f"Running cmake to install ttlang at {extension_path}")

        cwd = pathlib.Path().absolute()
        build_dir = cwd.parent / "build"

        # Set it to install directly into the wheel
        install_dir = pathlib.Path(self.build_lib)

        # Fix install dir if using cibuildwheel
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

        # Run source env/activate if in ci, otherwise onus is on dev
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

        # Install the PythonWheel Component
        self.spawn(
            ["cmake", "--install", str(build_dir), "--component", "TTLangPythonWheel"]
        )


date = datetime.now().strftime("%y.%m.%d")
version = "0.1." + date + ".dev0"

# Only the ttlang package relies on the CMake build process
ttlang_c = TTLangExtension("ttl")

# Read README.md file from project root
readme_path = pathlib.Path(__file__).absolute().parent.parent / "README.md"
with open(str(readme_path), "r", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="ttl",
    version=version,
    install_requires=[
        "pydantic<3",
    ],
    # Include ttlang as top-level packages
    packages=["ttl", "ttl._src", "sim", "utils"],
    package_dir={
        "ttl": "ttl",
        "ttl._src": "ttl._src",
        "sim": "sim",
        "utils": "utils",
    },
    ext_modules=[ttlang_c],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    # Write to readme
    long_description=readme,
    long_description_content_type="text/markdown",
)
