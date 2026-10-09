# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CPU-only PyTorch policy for the packaged toolchain, not user environments."""

import json
import os
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
        assert stage.index("-m pip install") < stage.index(
            "$TTLANG_TOOLCHAIN_DIR/venv/bin/python /tmp/check-cpu-torch.py"
        )
    uplift_paths = (REPO_ROOT / ".github/scripts/uplift-paths.sh").read_text()
    for helper in ("check-cpu-torch.py", "prepare-toolchain-venv.sh"):
        assert f".github/containers/{helper}" in uplift_paths


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
