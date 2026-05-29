# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for packaging helper scripts used by wheel workflows."""

from __future__ import annotations

import datetime
import os
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from conftest import REPO_ROOT  # noqa: E402

SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"
CHECK_WHEEL_TTNN_METADATA = SCRIPTS_DIR / "check-wheel-ttnn-metadata.py"
CHECK_LIGHT_METAPACKAGE = SCRIPTS_DIR / "check-light-metapackage.py"
COMPUTE_NIGHTLY_VERSION = SCRIPTS_DIR / "compute-nightly-version.py"
CHECK_INSTALLED_TTNN = SCRIPTS_DIR / "check-installed-ttnn.py"
CHECK_BUNDLED_PAYLOAD = SCRIPTS_DIR / "check-wheel-bundled-payload.py"
PUBLISH_S3_PYPI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-s3-pypi.yml"


def _run_script(
    script: Path,
    *args: str,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_wheel(dist_dir: Path, filename: str, metadata: str) -> Path:
    wheel_path = dist_dir / filename
    dist_info = filename.split("-", 1)[0] + "-0.0.0.dist-info"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        wheel.writestr(f"{dist_info}/METADATA", metadata)
    return wheel_path


def test_s3_schedule_publishes_bundled_and_light_wheels() -> None:
    workflow = PUBLISH_S3_PYPI_WORKFLOW.read_text()

    assert (
        "github.event_name == 'workflow_dispatch' && inputs.wheel_variant || ''"
    ) in workflow
    assert "EVENT_NAME: ${{ github.event_name }}" in workflow


def test_s3_stable_tags_publish_clean_version_wheels() -> None:
    workflow = PUBLISH_S3_PYPI_WORKFLOW.read_text()

    assert "push:" in workflow
    assert "tags:" in workflow
    assert "- 'v[0-9]+.[0-9]+.[0-9]+'" in workflow


def test_check_wheel_ttnn_metadata_matches_requirement_name(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_wheel(
        dist_dir,
        "tt_lang-0.71.0.dev20260525-py3-none-any.whl",
        "Metadata-Version: 2.1\nRequires-Dist: ttnn-foo >= 1\n",
    )

    result = _run_script(
        CHECK_WHEEL_TTNN_METADATA, "--mode", "pypi", "--dist-dir", str(dist_dir)
    )

    assert result.returncode != 0
    assert "default wheel metadata must require ttnn" in result.stderr


def test_check_wheel_ttnn_metadata_rejects_external_payload(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    wheel_path = _write_wheel(
        dist_dir,
        "tt_lang-0.71.0.dev20260525+light-py3-none-any.whl",
        "Metadata-Version: 2.1\n",
    )
    with zipfile.ZipFile(wheel_path, "a") as wheel:
        wheel.writestr("ttnn/__init__.py", "")

    result = _run_script(
        CHECK_WHEEL_TTNN_METADATA, "--mode", "external", "--dist-dir", str(dist_dir)
    )

    assert result.returncode != 0
    assert "external wheel must not bundle a ttnn payload" in result.stderr


def test_check_light_metapackage_parses_requires_dist(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_wheel(
        dist_dir,
        "tt_lang_light-0.71.0.dev20260525-py3-none-any.whl",
        (
            "Metadata-Version: 2.1\n"
            "Requires-Dist: tt-lang == 0.71.0.dev20260525+light ; "
            'python_version >= "3.12"\n'
        ),
    )

    result = _run_script(
        CHECK_LIGHT_METAPACKAGE,
        "--dist-dir",
        str(dist_dir),
        "--expect-ttlang-version",
        "0.71.0.dev20260525+light",
    )

    assert result.returncode == 0, result.stderr


def test_compute_nightly_version_uses_latest_stable_tag(
    tmp_path: Path,
) -> None:
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "file.txt").write_text("first\n")
    subprocess.run(["git", "add", "file.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "first"], cwd=tmp_path, check=True)
    subprocess.run(["git", "tag", "v1.2.3"], cwd=tmp_path, check=True)

    subprocess.run(["git", "checkout", "-b", "release"], cwd=tmp_path, check=True)
    (tmp_path / "file.txt").write_text("release\n")
    subprocess.run(["git", "commit", "-am", "release"], cwd=tmp_path, check=True)
    subprocess.run(["git", "tag", "v1.2.4"], cwd=tmp_path, check=True)

    subprocess.run(["git", "checkout", "main"], cwd=tmp_path, check=True)
    (tmp_path / "file.txt").write_text("second\n")
    subprocess.run(["git", "commit", "-am", "second"], cwd=tmp_path, check=True)

    result = _run_script(COMPUTE_NIGHTLY_VERSION, cwd=tmp_path)
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"1.2.4.dev{today}"


def _write_fake_ttnn(root: Path, *, with_native_libs: bool) -> None:
    package = root / "ttnn"
    package.mkdir()
    (package / "__init__.py").write_text("")
    if with_native_libs:
        (package / "_ttnn.so").write_bytes(b"")
        build_lib = package / "build" / "lib"
        build_lib.mkdir(parents=True)
        (build_lib / "_ttnncpp.so").write_bytes(b"")
        (build_lib / "libtt_metal.so").write_bytes(b"")


def _env_with_pythonpath(path: Path) -> dict[str, str]:
    env = {**os.environ}
    env["PYTHONPATH"] = str(path)
    return env


def _env_with_pythonpath_and_ldd_output(
    path: Path,
    *,
    stdout: str,
    stderr: str = "",
    exit_code: int = 0,
) -> dict[str, str]:
    env = _env_with_pythonpath(path)
    script = (
        "import sys\n"
        f"sys.stdout.write({stdout!r})\n"
        f"sys.stderr.write({stderr!r})\n"
        f"raise SystemExit({exit_code})\n"
    )
    env["TTLANG_LDD_COMMAND"] = shlex.join([sys.executable, "-c", script])
    return env


def _env_with_pythonpath_and_ldd(path: Path) -> dict[str, str]:
    ttnncpp_path = path / "ttnn" / "build" / "lib" / "_ttnncpp.so"
    return _env_with_pythonpath_and_ldd_output(
        path,
        stdout=f"\t_ttnncpp.so => {ttnncpp_path} (0x00000000)\n",
    )


def test_check_installed_ttnn_pypi_mode_is_noop() -> None:
    result = _run_script(CHECK_INSTALLED_TTNN, "--mode", "pypi")
    assert result.returncode == 0, result.stderr


def test_check_installed_ttnn_external_passes_when_ttnn_absent(
    tmp_path: Path,
) -> None:
    result = _run_script(
        CHECK_INSTALLED_TTNN,
        "--mode",
        "external",
        env=_env_with_pythonpath(tmp_path),
    )
    assert result.returncode == 0, result.stderr


def test_check_installed_ttnn_external_fails_when_ttnn_present(
    tmp_path: Path,
) -> None:
    _write_fake_ttnn(tmp_path, with_native_libs=False)

    result = _run_script(
        CHECK_INSTALLED_TTNN,
        "--mode",
        "external",
        env=_env_with_pythonpath(tmp_path),
    )

    assert result.returncode != 0
    assert "external wheel unexpectedly installed ttnn" in result.stderr


def test_check_installed_ttnn_bundled_passes_with_required_files(
    tmp_path: Path,
) -> None:
    _write_fake_ttnn(tmp_path, with_native_libs=True)

    result = _run_script(
        CHECK_INSTALLED_TTNN,
        "--mode",
        "bundled",
        env=_env_with_pythonpath_and_ldd(tmp_path),
    )

    assert result.returncode == 0, result.stderr


def test_check_installed_ttnn_bundled_fails_when_files_missing(
    tmp_path: Path,
) -> None:
    _write_fake_ttnn(tmp_path, with_native_libs=False)

    result = _run_script(
        CHECK_INSTALLED_TTNN,
        "--mode",
        "bundled",
        env=_env_with_pythonpath(tmp_path),
    )

    assert result.returncode != 0
    assert "bundled ttnn is missing files" in result.stderr


def test_check_installed_ttnn_bundled_fails_when_ldd_exits_nonzero(
    tmp_path: Path,
) -> None:
    _write_fake_ttnn(tmp_path, with_native_libs=True)
    env = _env_with_pythonpath_and_ldd_output(
        tmp_path,
        stdout="",
        stderr="ldd: cannot read object\n",
        exit_code=1,
    )

    result = _run_script(CHECK_INSTALLED_TTNN, "--mode", "bundled", env=env)

    assert result.returncode != 0
    assert "ldd failed for bundled ttnn extension" in result.stderr


def test_check_installed_ttnn_bundled_fails_on_unresolved_libraries(
    tmp_path: Path,
) -> None:
    _write_fake_ttnn(tmp_path, with_native_libs=True)
    env = _env_with_pythonpath_and_ldd_output(
        tmp_path,
        stdout="\tlibmissing.so => not found\n",
    )

    result = _run_script(CHECK_INSTALLED_TTNN, "--mode", "bundled", env=env)

    assert result.returncode != 0
    assert "unresolved libraries" in result.stderr


def test_check_installed_ttnn_bundled_fails_when_ttnncpp_resolves_elsewhere(
    tmp_path: Path,
) -> None:
    _write_fake_ttnn(tmp_path, with_native_libs=True)
    # ldd reports _ttnncpp.so resolved to a system path rather than the bundled one.
    env = _env_with_pythonpath_and_ldd_output(
        tmp_path,
        stdout="\t_ttnncpp.so => /usr/lib/_ttnncpp.so (0x00000000)\n",
    )

    result = _run_script(CHECK_INSTALLED_TTNN, "--mode", "bundled", env=env)

    assert result.returncode != 0
    assert "does not resolve _ttnncpp.so from" in result.stderr


def _write_bundled_wheel(dist_dir: Path, *, complete: bool) -> Path:
    wheel_path = _write_wheel(
        dist_dir,
        "tt_lang-0.71.0.dev20260525-py3-none-any.whl",
        "Metadata-Version: 2.1\n",
    )
    with zipfile.ZipFile(wheel_path, "a") as wheel:
        wheel.writestr("ttnn/__init__.py", "")
        wheel.writestr("ttnn/_ttnn.so", b"")
        if complete:
            wheel.writestr("ttnn/build/lib/_ttnncpp.so", b"")
            wheel.writestr("ttnn/build/lib/libtt_metal.so", b"")
    return wheel_path


def test_check_wheel_bundled_payload_accepts_complete_wheel(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_bundled_wheel(dist_dir, complete=True)

    result = _run_script(CHECK_BUNDLED_PAYLOAD, "--dist-dir", str(dist_dir))

    assert result.returncode == 0, result.stderr


def test_check_wheel_bundled_payload_rejects_missing_files(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_bundled_wheel(dist_dir, complete=False)

    result = _run_script(CHECK_BUNDLED_PAYLOAD, "--dist-dir", str(dist_dir))

    assert result.returncode != 0
    assert "bundled wheel is missing" in result.stderr


def test_check_wheel_bundled_payload_rejects_empty_dist(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    result = _run_script(CHECK_BUNDLED_PAYLOAD, "--dist-dir", str(dist_dir))

    assert result.returncode != 0
    assert "expected one tt-lang wheel" in result.stderr


def test_internal_wheel_metadata_fails_when_git_version_cannot_be_derived(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.path.insert(0, str(REPO_ROOT / "packaging"))
    import internal_wheel_metadata

    monkeypatch.delenv("TTLANG_VERSION_OVERRIDE", raising=False)

    def fail_git(*_args: object, **_kwargs: object) -> str:
        raise subprocess.CalledProcessError(1, ["git"])

    monkeypatch.setattr(subprocess, "check_output", fail_git)

    with pytest.raises(SystemExit, match="failed to derive internal wheel version"):
        internal_wheel_metadata.get_version(REPO_ROOT)
