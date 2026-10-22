# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CPU-only PyTorch policy for the packaged toolchain, not user environments."""

import json
import os
import re
import runpy
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from conftest import REPO_ROOT

CONTAINERS = REPO_ROOT / ".github" / "containers"


@pytest.mark.parametrize(
    "cuda,hip,packages,error",
    [
        (None, None, ["torch", "numpy"], None),
        ("13.0", None, ["torch"], "Expected CPU-only PyTorch"),
        (None, "7.0", ["torch"], "Expected CPU-only PyTorch"),
        (None, None, ["torch", "nvidia-cublas-cu13"], "nvidia-cublas-cu13"),
        (None, None, ["torch", "NVIDIA_CUDNN_CU13"], "NVIDIA_CUDNN_CU13"),
        (None, None, ["torch", "cuda-toolkit"], "cuda-toolkit"),
        (None, None, ["torch", "cuda-bindings"], "cuda-bindings"),
        (None, None, ["torch", "triton"], "triton"),
        (None, None, ["torch", "pytorch-triton"], "pytorch-triton"),
        (None, None, ["torch", "pytorch-triton-rocm"], "pytorch-triton-rocm"),
        (None, None, ["torch", "PyTorch_TrItOn_ROCm"], "PyTorch_TrItOn_ROCm"),
    ],
)
def test_cpu_torch_check(monkeypatch, capsys, cuda, hip, packages, error):
    # A CUDA build reports no available GPU on CPU CI, but must still fail.
    torch = SimpleNamespace(
        __version__="2.0.0+cpu",
        version=SimpleNamespace(cuda=cuda, hip=hip),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        "importlib.metadata.distributions",
        lambda: [SimpleNamespace(metadata={"Name": name}) for name in packages],
    )
    if error:
        with pytest.raises(SystemExit, match=error):
            runpy.run_path(str(CONTAINERS / "check-cpu-torch.py"), run_name="__main__")
    else:
        runpy.run_path(str(CONTAINERS / "check-cpu-torch.py"), run_name="__main__")
        assert "CPU-only PyTorch 2.0.0+cpu" in capsys.readouterr().out


@pytest.mark.parametrize("failure", ["", "venv", "pip", "check"])
def test_prepare_toolchain_venv(tmp_path, failure):
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    toolchain = tmp_path / "toolchain with spaces"
    venv_python = toolchain / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    log = tmp_path / "calls.jsonl"
    stub = f"""#!{sys.executable}
import json, os, sys
with open(os.environ['CALL_LOG'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
step = sys.argv[2] if sys.argv[1] == '-m' else 'check'
sys.exit(23 if step == os.environ['FAIL_STEP'] else 0)
"""
    for executable in (mock_bin / "python3.12", venv_python):
        executable.write_text(stub)
        executable.chmod(0o755)
    result = subprocess.run(
        ["bash", str(CONTAINERS / "prepare-toolchain-venv.sh"), str(toolchain)],
        env={
            **os.environ,
            "PATH": f"{mock_bin}:{os.environ['PATH']}",
            "CALL_LOG": str(log),
            "FAIL_STEP": failure,
        },
        capture_output=True,
        text=True,
    )
    expected = [
        ["-m", "venv", str(toolchain / "venv")],
        [
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
            "torch",
        ],
        [str(CONTAINERS / "check-cpu-torch.py")],
    ]
    count = {"venv": 1, "pip": 2, "check": 3, "": 3}[failure]
    assert [json.loads(line) for line in log.read_text().splitlines()] == expected[
        :count
    ]
    assert result.returncode == (23 if failure else 0), result.stderr


def test_container_build_uses_and_checks_cpu_torch():
    workflow = (REPO_ROOT / ".github/workflows/call-build-docker.yml").read_text()
    assert workflow.index("prepare-toolchain-venv.sh") < workflow.index(
        "bash scripts/build-and-install.sh --configure-only"
    )
    dockerfile = (CONTAINERS / "Dockerfile").read_text()
    for target in ("ird", "dist"):
        stage = dockerfile.split(f"FROM ${{BASE_IMAGE}} AS {target}\n")[1]
        stage = stage.split("\nFROM ")[0]
        check_command = "$TTLANG_TOOLCHAIN_DIR/venv/bin/python /tmp/check-cpu-torch.py"
        assert stage.index("-m pip install") < stage.index(check_command)
        check_run = next(
            line
            for line in stage.replace("\\\n", "").splitlines()
            if check_command in line
        )
        assert check_run.startswith(
            "RUN --mount=type=bind,source=.github/containers/check-cpu-torch.py,"
            "target=/tmp/check-cpu-torch.py "
        )
        assert "COPY .github/containers/check-cpu-torch.py" not in stage
        assert "rm /tmp/check-cpu-torch.py" not in stage
    uplift_paths = (REPO_ROOT / ".github/scripts/uplift-paths.sh").read_text()
    for helper in ("check-cpu-torch.py", "prepare-toolchain-venv.sh"):
        assert f".github/containers/{helper}" in uplift_paths


def emule_stage(name):
    dockerfile = (CONTAINERS / "Dockerfile.emule").read_text()
    return dockerfile.split(f" AS {name}\n", 1)[1].split("\nFROM ", 1)[0]


def test_emule_image_replaces_donor_venv_with_locked_cpu_torch():
    venv = emule_stage("venv")
    resolve = emule_stage("venv-resolve")
    runtime = emule_stage("runtime")
    cpu_check = "/opt/ttlang-toolchain/venv/bin/python /tmp/check-cpu-torch.py"
    steps = [
        "rm -rf /opt/ttlang-toolchain/venv",
        "python3.12 -m venv /opt/ttlang-toolchain/venv",
        "-r /tmp/emule-requirements.lock",
        "--find-links /tmp/wheels torch",
        "-r /tmp/requirements/requirements-test.txt",
        "check-venv",
        cpu_check,
        "bash /tmp/trim-emule-venv.sh",
    ]
    assert [venv.index(step) for step in steps] == sorted(
        venv.index(step) for step in steps
    )
    prepare = "bash /tmp/prepare-toolchain-venv.sh /opt/ttlang-toolchain"
    assert (
        resolve.index(prepare)
        < resolve.index("-r /tmp/requirements/requirements-test.txt")
        < resolve.index(cpu_check)
    )
    # The native build keeps the donor venv it was configured with.
    for stage in ("toolchain", "metal-fetch", "metal-source", "build"):
        assert "rm -rf /opt/ttlang-toolchain/venv" not in emule_stage(stage)
    assert "COPY --from=venv /opt/ttlang-toolchain/venv " in runtime
    for stage in (venv, resolve, runtime):
        assert "source=check-cpu-torch.py,target=/tmp/check-cpu-torch.py" in stage
        assert "download.pytorch.org" not in stage
    assert runtime.index("--without-pip --symlinks") < runtime.index(cpu_check)
    assert runtime.index("-m pip check") < runtime.index(cpu_check)


@pytest.mark.parametrize(
    "failure", ["", "venv", "download", "torch", "install", "verify", "check", "trim"]
)
def test_emule_locked_venv_stops_on_failures(tmp_path, failure):
    section = emule_stage("venv")
    command = (
        "rm -rf /opt/ttlang-toolchain/venv"
        + section.split("rm -rf /opt/ttlang-toolchain/venv", 1)[1].split("\n\n", 1)[0]
    )
    toolchain = tmp_path / "toolchain"
    helpers = tmp_path / "helpers"
    log = tmp_path / "calls.jsonl"
    helpers.mkdir()
    (toolchain / "venv").mkdir(parents=True)
    (toolchain / "venv/donor").write_text("donor\n")
    venv_python = f"""#!{sys.executable}
import json, os, sys
step = (
    'download' if 'download' in sys.argv
    else 'torch' if sys.argv[-1] == 'torch'
    else 'install' if 'install' in sys.argv
    else 'verify' if 'check-venv' in sys.argv
    else 'check'
)
with open(os.environ['CALL_LOG'], 'a') as stream:
    stream.write(json.dumps(step) + '\\n')
sys.exit(23 if step == os.environ['FAIL_STEP'] else 0)
"""
    (helpers / "python3.12").write_text(
        '#!/usr/bin/env bash\nset -eu\n[ "$1 $2" = "-m venv" ] && [ ! -e "$3" ]\n'
        'printf \'"venv"\\n\' >> "$CALL_LOG"\n'
        '[ "$FAIL_STEP" != venv ] || exit 23\n'
        'mkdir -p "$3/bin"\n'
        f'cp "{helpers}/venv-python" "$3/bin/python"\n'
        'chmod +x "$3/bin/python"\n'
        'ln -s python "$3/bin/python3"\n'
        'ln -s python "$3/bin/python3.12"\n'
    )
    (helpers / "python3.12").chmod(0o755)
    (helpers / "venv-python").write_text(venv_python)
    (helpers / "trim-emule-venv.sh").write_text(
        'printf \'"trim"\\n\' >> "$CALL_LOG"\n' '[ "$FAIL_STEP" != trim ] || exit 23\n'
    )
    # One pass, so a tmp_path under /tmp cannot be rewritten a second time.
    paths = {"/opt/ttlang-toolchain": str(toolchain), "/tmp/": f"{helpers}/"}
    command = re.sub(
        "|".join(map(re.escape, paths)), lambda match: paths[match[0]], command
    )
    result = subprocess.run(
        ["bash", "-e", "-c", command],
        env={
            **os.environ,
            "PATH": f"{helpers}{os.pathsep}{os.environ['PATH']}",
            "CALL_LOG": str(log),
            "FAIL_STEP": failure,
        },
        capture_output=True,
        text=True,
    )
    expected = ["venv", "download", "torch", "install", "verify", "check", "trim"]
    if failure:
        expected = expected[: expected.index(failure) + 1]
    assert [json.loads(line) for line in log.read_text().splitlines()] == expected
    assert result.returncode == (23 if failure else 0), result.stderr
    assert not (toolchain / "venv/donor").exists()
    if not failure:
        assert not any((toolchain / "venv/bin").iterdir())


@pytest.mark.parametrize(
    "document", [".github/containers/README.md", "docs/sphinx/build.md"]
)
@pytest.mark.parametrize("failure", [False, True])
def test_documented_toolchain_preparation(tmp_path, document, failure):
    blocks = (REPO_ROOT / document).read_text().split("```bash\n")[1:]
    commands = [
        block.split("```")[0]
        for block in blocks
        if "prepare-toolchain-venv.sh" in block.split("```")[0]
    ]
    assert len(commands) == 1
    containers = tmp_path / ".github" / "containers"
    containers.mkdir(parents=True)
    (containers / "prepare-toolchain-venv.sh").write_text(
        'echo "prepare $*"\n' + ("exit 23\n" if failure else "")
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "build-and-install.sh").write_text(
        'echo "configure $TTLANG_TOOLCHAIN_DIR $*"\n'
    )
    result = subprocess.run(
        ["bash", "-c", commands[0]],
        cwd=tmp_path,
        env={**os.environ, "TTLANG_TOOLCHAIN_DIR": "/unrelated-toolchain"},
        text=True,
        capture_output=True,
    )
    expected = ["prepare /opt/ttlang-toolchain"]
    if not failure:
        expected.append("configure /opt/ttlang-toolchain --configure-only")
    assert result.stdout.splitlines() == expected
    assert result.returncode == (23 if failure else 0), result.stderr


@pytest.mark.parametrize("source", ["current", "historical", "broken"])
def test_configure_preserves_historical_source_overrides(tmp_path, source):
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/call-build-docker.yml").read_text()
    )
    steps = workflow["jobs"]["build-images"]["steps"]
    configure = next(
        step["run"] for step in steps if step.get("name") == "Configure tt-lang"
    )
    arguments = shlex.split(configure)
    command = arguments[arguments.index("-c") + 1]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "build-and-install.sh").write_text('echo "configure $*"\n')
    if source != "historical":
        containers = tmp_path / ".github" / "containers"
        containers.mkdir(parents=True)
        (containers / "prepare-toolchain-venv.sh").write_text(
            'echo "prepare $*"\n' + ("exit 23\n" if source == "broken" else "")
        )
    result = subprocess.run(
        ["bash", "-e", "-c", command],
        cwd=tmp_path,
        env={**os.environ, "TTLANG_TOOLCHAIN_DIR": "/test toolchain"},
        text=True,
        capture_output=True,
    )
    expected = [] if source == "historical" else ["prepare /test toolchain"]
    if source != "broken":
        expected.append("configure --configure-only")
    assert result.stdout.splitlines() == expected
    assert result.returncode == (23 if source == "broken" else 0), result.stderr
