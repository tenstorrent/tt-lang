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
import re
import shutil
import subprocess
import sys

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version
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
_TTNN_DEP_MODES = ("pypi", "external", "bundled")
_VERSION_OVERRIDE_ENV = "TTLANG_VERSION_OVERRIDE"
_ALLOW_FINAL_INTERNAL_VERSION_ENV = "TTLANG_ALLOW_FINAL_INTERNAL_VERSION"


def _ttnn_dep_mode():
    mode = os.environ.get("TTLANG_TTNN_DEP_MODE", "pypi").strip().lower()
    if mode not in _TTNN_DEP_MODES:
        allowed = ", ".join(_TTNN_DEP_MODES)
        raise SystemExit(f"TTLANG_TTNN_DEP_MODE must be one of: {allowed}")
    return mode


def _version_override():
    return os.environ.get(_VERSION_OVERRIDE_ENV, "").strip()


def _allow_final_internal_version():
    return os.environ.get(_ALLOW_FINAL_INTERNAL_VERSION_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _read_tt_metal_version_var(name):
    """Read a shell variable from third-party/tt-metal-version.

    The file is a sourceable shell snippet (`KEY="value"` assignments).
    Expected names include TTNN_PYPI, TTNN_PYPI_TT_METAL_TAG, and
    TT_METAL_TAG. See the file's header.
    """
    version_file = REPO_ROOT / "third-party" / "tt-metal-version"
    text = version_file.read_text()
    match = re.search(rf'^{re.escape(name)}="([^"]*)"$', text, re.MULTILINE)
    if not match:
        raise SystemExit(f"{version_file}: variable '{name}' not found")
    return match.group(1)


def _ttnn_requirement():
    """Build the platform-conditional ttnn requirement from
    third-party/tt-metal-version. ttnn only publishes wheels for Linux
    x86_64 / aarch64, so the requirement is conditioned on those platforms
    via a PEP 508 marker; on macOS / Windows it is silently skipped (those
    platforms are sim-only via the separate `tt-lang-sim` package).
    """
    version = _read_tt_metal_version_var("TTNN_PYPI")
    marker = (
        "sys_platform == 'linux' "
        "and (platform_machine == 'x86_64' or platform_machine == 'aarch64')"
    )
    return f"ttnn == {version} ; {marker}"


def _read_install_requires():
    """Read base runtime requirements from requirements-runtime.txt and
    append the dynamic ttnn requirement unless the wheel is explicitly built
    for an externally managed or bundled tt-metal/ttnn install.
    """
    req_file = REPO_ROOT / "requirements-runtime.txt"
    requirements = []
    for line in req_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    mode = _ttnn_dep_mode()
    if mode == "pypi":
        requirements.append(_ttnn_requirement())
    elif mode in ("external", "bundled"):
        requirements.extend(
            _missing_requirements(requirements, TTNN_RUNTIME_REQUIREMENTS)
        )
    return requirements


def _missing_requirements(existing_requirements, extra_requirements):
    existing_names = {
        _requirement_name(requirement) for requirement in existing_requirements
    }
    return [
        requirement
        for requirement in extra_requirements
        if _requirement_name(requirement) not in existing_names
    ]


def _requirement_name(requirement_text):
    try:
        return Requirement(requirement_text).name.lower()
    except InvalidRequirement as error:
        raise SystemExit(f"invalid requirement {requirement_text!r}") from error


def get_version_from_git():
    """Get version from git tags, matching cmake/modules/GetVersionFromGit.cmake.

    Tag format: vMAJOR.MINOR.PATCH[+LOCAL]. Per PEP 440 the .devN segment must
    sit between the public release and the +local label, so the tag is split
    on '+' before the dev counter is inserted.

    Override mechanism: if TTLANG_VERSION_OVERRIDE is set in the environment,
    it is returned verbatim. Used by workflows to stamp wheels built from a
    branch with a caller-supplied PEP 440 version when no matching git tag
    exists.
    """
    version_override = _version_override()
    if version_override:
        return version_override
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
    except (subprocess.CalledProcessError, OSError) as error:
        raise SystemExit(
            "failed to derive tt-lang version from git; set "
            f"{_VERSION_OVERRIDE_ENV} when building outside a tagged checkout"
        ) from error

    base, sep, local = tag.partition("+")
    local_suffix = f"+{local}" if sep else ""
    if commits and commits != "0":
        return f"{base}.dev{commits}{local_suffix}"
    return f"{base}{local_suffix}"


def _validate_ttnn_dep_mode_version(version):
    mode = _ttnn_dep_mode()
    if mode not in ("external", "bundled"):
        return

    try:
        parsed_version = Version(version)
    except InvalidVersion as error:
        raise SystemExit(f"invalid {_VERSION_OVERRIDE_ENV} {version!r}") from error

    if (
        not parsed_version.is_devrelease
        and not parsed_version.is_prerelease
        and not _allow_final_internal_version()
    ):
        raise SystemExit(
            f"TTLANG_TTNN_DEP_MODE={mode} requires a non-final version "
            "such as 0.71.0.dev20260525 or 0.71.0rc1"
        )
    if mode == "external" and parsed_version.local != "light":
        raise SystemExit(
            "TTLANG_TTNN_DEP_MODE=external requires a +light local version "
            "such as 0.71.0.dev20260525+light"
        )


def _require_ttnn_dep_mode_version_override():
    mode = _ttnn_dep_mode()
    if mode in ("external", "bundled") and not _version_override():
        raise SystemExit(
            f"TTLANG_TTNN_DEP_MODE={mode} requires {_VERSION_OVERRIDE_ENV} "
            "so internal wheels cannot be confused with PyPI release wheels"
        )


class TTLangExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            if "ttl" in ext.name:
                self.build_(ext)
            else:
                raise RuntimeError(f"Unknown extension: {ext.name}")

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
        bundled_ttnn_extension = install_dir / "ttnn" / "_ttnn.so"
        for so_file in glob.glob(str(install_dir / "**/*.so"), recursive=True):
            rpath = (
                "$ORIGIN/build/lib"
                if pathlib.Path(so_file) == bundled_ttnn_extension
                else "$ORIGIN"
            )
            self.spawn(["patchelf", "--set-rpath", rpath, so_file])

    def _remove_bundled_ttnn(self, install_dir):
        """Remove stale bundled payloads left by earlier wheel builds."""
        for package_name in ("ttnn", "tracy"):
            package_dir = install_dir / package_name
            if package_dir.exists():
                shutil.rmtree(package_dir)

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

            for env_var in (
                "TTLANG_EXTERNAL_TT_METAL_DIR",
                "TTLANG_EXTERNAL_TT_METAL_BUILD_DIR",
                "TTLANG_ACCEPT_TTMETAL_MISMATCH",
                "TTLANG_PYTHON_VENV",
            ):
                value = os.environ.get(env_var, "")
                if value:
                    cmake_args.append(f"-D{env_var}={value}")

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

        if _ttnn_dep_mode() == "bundled":
            copy_bundled_ttnn(BUNDLED_TT_METAL_ROOT, install_dir)
        else:
            self._remove_bundled_ttnn(install_dir)

        # Post-install: strip binaries and fix RPATH for wheel distribution
        self._strip_binaries(install_dir)
        self._fix_rpath(install_dir)


ttlang_c = TTLangExtension("ttl")

sys.path.insert(0, str(REPO_ROOT / "packaging"))
from bundled_ttnn import (  # noqa: E402
    TTNN_RUNTIME_REQUIREMENTS,
    copy_bundled_ttnn,
    resolve_tt_metal_root,
    stage_bundled_ttnn_python_packages,
)
from rewrite_readme import absolutize_readme_images, ref_for_version  # noqa: E402

_require_ttnn_dep_mode_version_override()
_version = get_version_from_git()
_validate_ttnn_dep_mode_version(_version)

BUNDLED_TT_METAL_ROOT = None
_bundled_packages = []
_bundled_package_dir = {}
if _ttnn_dep_mode() == "bundled":
    try:
        BUNDLED_TT_METAL_ROOT = resolve_tt_metal_root(REPO_ROOT)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    _bundled_metadata = stage_bundled_ttnn_python_packages(
        BUNDLED_TT_METAL_ROOT,
        REPO_ROOT / "build" / "bundled-ttnn-python",
        REPO_ROOT,
    )
    _bundled_packages = _bundled_metadata.packages
    _bundled_package_dir = _bundled_metadata.package_dir

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
        "ttl._pipenets",
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
    ]
    + _bundled_packages,
    package_dir={
        "ttl": "python/ttl",
        "ttl._pipenets": "python/ttl/_pipenets",
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
    }
    | _bundled_package_dir,
    ext_modules=[ttlang_c],
    cmdclass={"build_ext": CMakeBuild, "sdist": NoSdist},
    zip_safe=False,
    long_description=readme,
    long_description_content_type="text/markdown",
)
