# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Docker simulator CLI without a Docker installation."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / "scripts" / "tt-lang-emule.py"


@pytest.fixture
def cli(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("ttlang_emule_cli", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "CONFIG_PATH", tmp_path / ".ttlang-sim" / "emule.json")
    for name in tuple(os.environ):
        if name.startswith("TTLANG_EMULE_"):
            monkeypatch.delenv(name)
    return module


def invoke(cli, monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", [str(CLI), *map(str, arguments)])
    return cli.main()


@pytest.mark.parametrize("exit_code", [0, 17])
def test_setup_changes_saved_settings_only_after_success(cli, monkeypatch, exit_code):
    previous = {"schema_version": 1, "image": "runtime:previous", "jobs": 2}
    cli.save_settings(previous)
    saved = cli.CONFIG_PATH.read_bytes()
    source = cli.REPO_ROOT / "emulator checkout"
    source.mkdir()
    calls = []
    monkeypatch.setattr(cli, "prepare_image", lambda environment: None)

    def smoke(command, environment):
        assert cli.CONFIG_PATH.read_bytes() == saved
        if "validate" in command:
            return 0
        calls.append((command, environment))
        return exit_code

    monkeypatch.setattr(cli, "run_command", smoke)

    result = invoke(cli, monkeypatch, "setup", "--source", source, "--jobs", "4")

    assert result == exit_code
    assert len(calls) == 1
    command, environment = calls[0]
    assert command[-3:] == ["--backend", "emule", "--smoke-test"]
    assert environment["TTLANG_EMULE_RUNTIME_SOURCE_DIR"] == str(source)
    assert environment["TTLANG_EMULE_JOBS"] == "4"
    assert "TTLANG_EMULE_IMAGE" not in environment
    if exit_code:
        assert cli.CONFIG_PATH.read_bytes() == saved
    else:
        assert json.loads(cli.CONFIG_PATH.read_text()) == {
            "schema_version": 1,
            "source": str(source),
            "jobs": 4,
        }


def test_failed_first_setup_does_not_create_configuration(cli, monkeypatch):
    monkeypatch.setattr(cli, "prepare_image", lambda environment: None)
    monkeypatch.setattr(cli, "run_command", lambda command, environment: 9)

    assert invoke(cli, monkeypatch, "setup", "--image", "runtime:broken") == 9
    assert not cli.CONFIG_PATH.exists()


@pytest.mark.parametrize("jobs", ["0", "many"])
@pytest.mark.parametrize("command", ["setup", "smoke"])
def test_invalid_environment_jobs_stop_before_docker_or_smoke(
    cli, monkeypatch, capsys, jobs, command
):
    cli.save_settings({"image": "runtime:previous", "jobs": 2})
    saved = cli.CONFIG_PATH.read_bytes()
    monkeypatch.setenv("TTLANG_EMULE_JOBS", jobs)

    def unexpected_process(*arguments, **keywords):
        pytest.fail("invalid parallelism must be rejected before starting a process")

    monkeypatch.setattr(cli.subprocess, "run", unexpected_process)

    assert invoke(cli, monkeypatch, command) == 2
    assert "TTLANG_EMULE_JOBS" in capsys.readouterr().err
    assert cli.CONFIG_PATH.read_bytes() == saved


@pytest.mark.parametrize(
    "build_settings",
    [
        {"TTLANG_EMULE_RUNTIME_SOURCE_DIR": "source"},
        {"TTLANG_EMULE_RUNTIME_SOURCE_URL": "https://example.com/emulator.git"},
        {"TTLANG_EMULE_REBUILD": "1"},
        {"TTLANG_EMULE_RUNTIME_SOURCE_DIR": "source", "TTLANG_EMULE_REBUILD": "1"},
    ],
)
def test_local_build_selection_does_not_try_pulling_missing_image(
    cli, monkeypatch, build_settings
):
    monkeypatch.setenv("TTLANG_EMULE_IMAGE", "runtime:local-build")
    for name, value in build_settings.items():
        monkeypatch.setenv(name, value)
    calls = []

    def unexpected_docker(*arguments, **keywords):
        pytest.fail("a source build must not inspect or pull its output image")

    def launcher(command, environment):
        calls.append((command, environment))
        return 0

    monkeypatch.setattr(cli.subprocess, "run", unexpected_docker)
    monkeypatch.setattr(cli, "run_command", launcher)

    assert invoke(cli, monkeypatch, "smoke") == 0
    assert len(calls) == 1
    command, environment = calls[0]
    assert command[-3:] == ["--backend", "emule", "--smoke-test"]
    assert environment["TTLANG_EMULE_IMAGE"] == "runtime:local-build"
    for name, value in build_settings.items():
        assert environment[name] == value


def test_invalid_source_stops_before_docker_and_preserves_saved_settings(
    cli, monkeypatch
):
    cli.save_settings({"image": "runtime:previous"})
    saved = cli.CONFIG_PATH.read_bytes()
    source = cli.REPO_ROOT / "invalid emulator"
    source.mkdir()
    calls = []

    def validate(command, environment):
        calls.append(command)
        assert "validate" in command
        assert command[-2:] == ["--emulator-source", str(source)]
        return 11

    def unexpected_prepare(environment):
        pytest.fail("an invalid emulator source must not reach Docker")

    monkeypatch.setattr(cli, "run_command", validate)
    monkeypatch.setattr(cli, "prepare_image", unexpected_prepare)

    assert invoke(cli, monkeypatch, "setup", "--source", source) == 11
    assert len(calls) == 1
    assert cli.CONFIG_PATH.read_bytes() == saved


@pytest.mark.parametrize("selected", ["source", "source_url", "image"])
def test_explicit_runtime_replaces_saved_and_environment_alternatives(
    cli, monkeypatch, selected
):
    source = cli.REPO_ROOT / "new emulator"
    source.mkdir()
    values = {
        "source": str(source),
        "source_url": "https://example.com/emulator.git",
        "image": "runtime:selected",
    }
    cli.save_settings({"image": "runtime:saved", "jobs": 2})
    for key in ("source", "source_url", "image"):
        monkeypatch.setenv(cli.SETTINGS[key], "previous-" + key)
    monkeypatch.setattr(cli, "prepare_image", lambda environment: None)
    environments = []

    def smoke(command, environment):
        environments.append(environment)
        return 0

    monkeypatch.setattr(cli, "run_command", smoke)

    result = invoke(
        cli, monkeypatch, "setup", "--" + selected.replace("_", "-"), values[selected]
    )

    assert result == 0
    saved = json.loads(cli.CONFIG_PATH.read_text())
    assert saved == {"schema_version": 1, selected: values[selected], "jobs": 2}
    for key in ("source", "source_url", "image"):
        assert environments[0].get(cli.SETTINGS[key]) == (
            values[selected] if key == selected else None
        )


def test_saved_runtime_and_jobs_are_used_by_later_commands(cli, monkeypatch):
    cli.save_settings({"image": "runtime:saved", "jobs": 3})
    environments = []
    monkeypatch.setattr(cli, "prepare_image", lambda environment: None)

    def smoke(command, environment):
        environments.append(environment)
        return 0

    monkeypatch.setattr(cli, "run_command", smoke)

    assert invoke(cli, monkeypatch, "smoke") == 0
    assert environments[0]["TTLANG_EMULE_IMAGE"] == "runtime:saved"
    assert environments[0]["TTLANG_EMULE_JOBS"] == "3"


def test_environment_runtime_and_jobs_override_saved_settings(cli, monkeypatch):
    cli.save_settings({"image": "runtime:saved", "jobs": 3})
    monkeypatch.setenv(
        "TTLANG_EMULE_RUNTIME_SOURCE_URL", "https://example.com/override.git"
    )
    monkeypatch.setenv("TTLANG_EMULE_JOBS", "7")
    environments = []
    monkeypatch.setattr(cli, "prepare_image", lambda environment: None)

    def smoke(command, environment):
        environments.append(environment)
        return 0

    monkeypatch.setattr(cli, "run_command", smoke)

    assert invoke(cli, monkeypatch, "smoke") == 0
    assert "TTLANG_EMULE_IMAGE" not in environments[0]
    assert environments[0]["TTLANG_EMULE_RUNTIME_SOURCE_URL"] == (
        "https://example.com/override.git"
    )
    assert environments[0]["TTLANG_EMULE_JOBS"] == "7"
    assert json.loads(cli.CONFIG_PATH.read_text())["image"] == "runtime:saved"


def test_setup_persists_the_successfully_tested_environment_override(cli, monkeypatch):
    cli.save_settings({"image": "runtime:saved"})
    source = cli.REPO_ROOT / "override emulator"
    source.mkdir()
    monkeypatch.setenv("TTLANG_EMULE_RUNTIME_SOURCE_DIR", str(source))
    monkeypatch.setattr(cli, "prepare_image", lambda environment: None)
    monkeypatch.setattr(cli, "run_command", lambda command, environment: 0)

    assert invoke(cli, monkeypatch, "setup") == 0
    assert json.loads(cli.CONFIG_PATH.read_text()) == {
        "schema_version": 1,
        "source": str(source),
    }


def test_missing_script_is_rejected_before_accessing_docker(cli, monkeypatch, capsys):
    cli.save_settings({"image": "runtime:saved"})

    def unexpected_process(*arguments, **keywords):
        pytest.fail("a missing script must not start Docker or the launcher")

    monkeypatch.setattr(cli.subprocess, "run", unexpected_process)

    assert (
        invoke(
            cli, monkeypatch, "--launch", "/fake/runner", cli.REPO_ROOT / "missing.py"
        )
        == 2
    )
    assert "script not found" in capsys.readouterr().err


def test_failed_image_pull_does_not_launch_or_save_configuration(cli, monkeypatch):
    calls = []

    def docker(command, **keywords):
        calls.append(command)
        assert command[0] == "docker"
        code = 0 if command[1:] == ["info"] else 1
        return subprocess.CompletedProcess(command, code)

    monkeypatch.setattr(cli.subprocess, "run", docker)

    assert invoke(cli, monkeypatch, "setup", "--image", "runtime:missing") == 2
    assert calls == [
        ["docker", "info"],
        ["docker", "image", "inspect", "runtime:missing"],
        ["docker", "pull", "--platform", "linux/amd64", "runtime:missing"],
    ]
    assert not cli.CONFIG_PATH.exists()


@pytest.mark.parametrize("dirty", [False, True])
def test_each_test_run_keeps_separate_reports_and_host_provenance(
    cli, monkeypatch, dirty
):
    revision = "a" * 40
    calls = []
    returncodes = iter([0, 19])
    monkeypatch.setattr(cli, "prepare_image", lambda environment: None)
    cli.save_settings({"image": "runtime:tested"})

    def git(command, **keywords):
        assert command[:3] == ["git", "-C", str(cli.REPO_ROOT)]
        if command[3:] == ["rev-parse", "HEAD"]:
            output = revision + "\n"
        else:
            assert command[3:] == ["status", "--porcelain"]
            output = " M python/ttl/ttl_api.py\n" if dirty else ""
        return subprocess.CompletedProcess(command, 0, stdout=output)

    def run(command, environment):
        calls.append((command, environment.copy()))
        return next(returncodes)

    monkeypatch.setattr(cli.subprocess, "run", git)
    monkeypatch.setattr(cli, "run_command", run)
    reports_parent = cli.REPO_ROOT / "test reports"
    arguments = [
        "test",
        "--reports-dir",
        reports_parent,
        "--suite",
        "me2e",
        "--suite",
        "python-lit",
    ]

    assert invoke(cli, monkeypatch, *arguments) == 0
    assert invoke(cli, monkeypatch, *arguments) == 19
    directories = [
        Path(environment["TTLANG_EMULE_REPORT_DIR"]) for _, environment in calls
    ]
    assert directories[0] != directories[1]
    for (command, environment), reports in zip(calls, directories):
        assert reports.parent == reports_parent
        assert command[-7:] == [
            "--",
            "--reports-dir",
            "/ttlang-reports",
            "--suite",
            "me2e",
            "--suite",
            "python-lit",
        ]
        assert command[3] == str(
            cli.REPO_ROOT / "scripts" / "run-tt-lang-emule-suite.py"
        )
        provenance = json.loads((reports / "invocation.json").read_text())
        assert provenance["compiler_commit"] == revision
        assert provenance["compiler_dirty"] is dirty
        assert provenance["suites"] == ["me2e", "python-lit"]
        assert provenance["runtime_image"] == "runtime:tested"
        assert provenance["source_directory"] == str(cli.REPO_ROOT)


@pytest.fixture
def launcher_checkout(tmp_path):
    root = tmp_path / "source checkout"
    (root / "bin").mkdir(parents=True)
    (root / "scripts").mkdir()
    shutil.copy2(REPO_ROOT / "bin" / "tt-lang-sim", root / "bin" / "tt-lang-sim")
    shutil.copy2(CLI, root / "scripts" / "tt-lang-emule.py")
    runner = root / "runner.py"
    runner.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "payload = {'arguments': sys.argv[1:], 'environment': "
        "{key: value for key, value in os.environ.items() if key.startswith('TTLANG_EMULE_')}}\n"
        "pathlib.Path(os.environ['LAUNCH_LOG']).write_text(json.dumps(payload))\n"
        "sys.exit(int(os.environ.get('LAUNCH_RESULT', '0')))\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("TTLANG_EMULE_") and name != "TTLANG_SIM_BACKEND"
    }
    environment.update(
        TTLANG_EMULE_HOST_PYTHON=sys.executable,
        TTLANG_EMULE_RUNNER=str(runner),
        TTLANG_EMULE_DOCKER=str(root / "docker-must-not-run"),
        PYTHON=sys.executable,
        LAUNCH_LOG=str(root / "launch.json"),
    )
    return root, environment


@pytest.mark.parametrize("command", [[], ["setup"], ["smoke"], ["examples"], ["test"]])
def test_launcher_help_needs_no_docker_or_saved_configuration(
    launcher_checkout, command
):
    root, environment = launcher_checkout
    config = root / ".ttlang-sim" / "emule.json"
    config.parent.mkdir()
    config.write_text("invalid configuration", encoding="utf-8")

    result = subprocess.run(
        [str(root / "bin" / "tt-lang-sim"), "emule", *command, "--help"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert "--launch" not in result.stdout
    if not command:
        assert "{setup,smoke,examples,test}" in result.stdout
    assert not Path(environment["LAUNCH_LOG"]).exists()


def test_launcher_preserves_program_arguments_and_exit_status(launcher_checkout):
    root, environment = launcher_checkout
    script = root / "program with spaces.py"
    script.touch()
    arguments = [
        "--backend",
        "python",
        "--image",
        "image with spaces",
        "$(false)",
        "--",
        "",
    ]
    environment["LAUNCH_RESULT"] = "23"

    result = subprocess.run(
        [
            str(root / "bin" / "tt-lang-sim"),
            "--backend=emule",
            str(script),
            "--",
            *arguments,
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 23, result.stderr
    assert json.loads(Path(environment["LAUNCH_LOG"]).read_text())["arguments"] == [
        str(script),
        *arguments,
    ]


def save_launcher_settings(root, settings):
    config = root / ".ttlang-sim" / "emule.json"
    config.parent.mkdir(exist_ok=True)
    config.write_text(json.dumps({"schema_version": 1, **settings}), encoding="utf-8")


def enable_fake_docker(root, environment):
    docker = root / "docker.py"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "with pathlib.Path(os.environ['DOCKER_LOG']).open('a') as output:\n"
        "    output.write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    environment["TTLANG_EMULE_DOCKER"] = str(docker)
    environment["DOCKER_LOG"] = str(root / "docker.jsonl")


def run_launcher(root, environment, *arguments):
    return subprocess.run(
        [str(root / "bin" / "tt-lang-sim"), *arguments],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("runtime", ["source", "image"])
@pytest.mark.parametrize(
    "arguments",
    [
        ["--backend=emule", "program.py"],
        ["--backend", "emule", "program.py"],
        ["program.py", "--backend=emule"],
        ["program.py", "--backend", "emule"],
    ],
)
def test_normal_launcher_uses_saved_settings(launcher_checkout, runtime, arguments):
    root, environment = launcher_checkout
    script = root / "program.py"
    script.touch()
    value = "runtime:saved" if runtime == "image" else str(root / "emulator")
    save_launcher_settings(root, {runtime: value, "jobs": 3})
    enable_fake_docker(root, environment)
    arguments = [str(script) if value == "program.py" else value for value in arguments]

    result = run_launcher(root, environment, *arguments)

    assert result.returncode == 0, result.stderr
    payload = json.loads(Path(environment["LAUNCH_LOG"]).read_text())
    assert payload["arguments"] == [str(script)]
    runtime_variable = (
        "TTLANG_EMULE_IMAGE"
        if runtime == "image"
        else "TTLANG_EMULE_RUNTIME_SOURCE_DIR"
    )
    assert payload["environment"][runtime_variable] == value
    assert payload["environment"]["TTLANG_EMULE_JOBS"] == "3"
    docker_log = Path(environment["DOCKER_LOG"])
    if runtime == "image":
        calls = [json.loads(line) for line in docker_log.read_text().splitlines()]
        assert calls == [["info"], ["image", "inspect", "runtime:saved"]]
    else:
        assert not docker_log.exists()


def test_default_emule_backend_uses_saved_settings(launcher_checkout):
    root, environment = launcher_checkout
    script = root / "program.py"
    script.touch()
    source = str(root / "saved emulator")
    save_launcher_settings(root, {"source": source})
    environment["TTLANG_SIM_BACKEND"] = "emule"

    result = run_launcher(root, environment, str(script))

    assert result.returncode == 0, result.stderr
    payload = json.loads(Path(environment["LAUNCH_LOG"]).read_text())
    assert payload["arguments"] == [str(script)]
    assert payload["environment"]["TTLANG_EMULE_RUNTIME_SOURCE_DIR"] == source


def test_normal_launcher_environment_selection_replaces_saved_runtime(
    launcher_checkout,
):
    root, environment = launcher_checkout
    script = root / "program.py"
    script.touch()
    save_launcher_settings(root, {"image": "runtime:saved", "jobs": 2})
    environment["TTLANG_EMULE_RUNTIME_SOURCE_URL"] = "https://example.com/override.git"
    environment["TTLANG_EMULE_JOBS"] = "5"

    result = run_launcher(root, environment, "--backend=emule", str(script))

    assert result.returncode == 0, result.stderr
    effective = json.loads(Path(environment["LAUNCH_LOG"]).read_text())["environment"]
    assert "TTLANG_EMULE_IMAGE" not in effective
    assert (
        effective["TTLANG_EMULE_RUNTIME_SOURCE_URL"]
        == "https://example.com/override.git"
    )
    assert effective["TTLANG_EMULE_JOBS"] == "5"


@pytest.mark.parametrize(
    "image_option",
    [["--runtime-image", "runtime:explicit"], ["--runtime-image=runtime:explicit"]],
)
def test_normal_launcher_explicit_image_overrides_saved_and_environment_images(
    launcher_checkout, image_option
):
    root, environment = launcher_checkout
    script = root / "program.py"
    script.touch()
    save_launcher_settings(root, {"image": "runtime:saved"})
    environment["TTLANG_EMULE_IMAGE"] = "runtime:environment"
    enable_fake_docker(root, environment)

    result = run_launcher(
        root, environment, str(script), "--backend=emule", *image_option
    )

    assert result.returncode == 0, result.stderr
    effective = json.loads(Path(environment["LAUNCH_LOG"]).read_text())["environment"]
    assert effective["TTLANG_EMULE_IMAGE"] == "runtime:explicit"
    calls = [
        json.loads(line)
        for line in Path(environment["DOCKER_LOG"]).read_text().splitlines()
    ]
    assert calls == [["info"], ["image", "inspect", "runtime:explicit"]]


@pytest.mark.parametrize("backend", ["python", "emule"])
def test_top_level_help_ignores_invalid_saved_configuration(launcher_checkout, backend):
    root, environment = launcher_checkout
    config = root / ".ttlang-sim" / "emule.json"
    config.parent.mkdir()
    config.write_text("invalid configuration", encoding="utf-8")
    simulation = root / "python" / "sim" / "ttlang_sim.py"
    simulation.parent.mkdir(parents=True)
    (simulation.parent / "__init__.py").touch()
    simulation.write_text("print('usage: tt-lang-sim')\n", encoding="utf-8")

    result = run_launcher(root, environment, "--backend=" + backend, "--help")

    assert result.returncode == 0, result.stderr
    assert "usage: tt-lang-sim" in result.stdout
    assert not Path(environment["LAUNCH_LOG"]).exists()


def test_python_execution_ignores_invalid_saved_emule_configuration(launcher_checkout):
    root, environment = launcher_checkout
    config = root / ".ttlang-sim" / "emule.json"
    config.parent.mkdir()
    config.write_text("invalid configuration", encoding="utf-8")
    simulation = root / "python" / "sim" / "ttlang_sim.py"
    simulation.parent.mkdir(parents=True)
    (simulation.parent / "__init__.py").touch()
    simulation.write_text(
        "import sys\nassert sys.argv[1:] == ['program.py']\nprint('python backend selected')\n",
        encoding="utf-8",
    )
    environment["TTLANG_SIM_BACKEND"] = "emule"

    result = run_launcher(root, environment, "--backend=python", "program.py")

    assert result.returncode == 0, result.stderr
    assert "python backend selected" in result.stdout
    assert not Path(environment["LAUNCH_LOG"]).exists()


def test_normal_launcher_rejects_missing_script_before_docker(launcher_checkout):
    root, environment = launcher_checkout
    save_launcher_settings(root, {"image": "runtime:saved"})

    result = run_launcher(
        root, environment, "--backend=emule", str(root / "missing.py")
    )

    assert result.returncode == 2
    assert "script not found" in result.stderr
    assert not Path(environment["LAUNCH_LOG"]).exists()


def test_launcher_keeps_direct_runner_fallback_without_management_helper(
    launcher_checkout,
):
    root, environment = launcher_checkout
    (root / "scripts" / "tt-lang-emule.py").unlink()
    script = root / "program.py"
    script.touch()

    result = run_launcher(root, environment, "--backend=emule", str(script))

    assert result.returncode == 0, result.stderr
    assert json.loads(Path(environment["LAUNCH_LOG"]).read_text())["arguments"] == [
        str(script)
    ]
