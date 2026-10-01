# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "emule_ci_worker",
    Path(__file__).resolve().parents[2] / "scripts/emule-ci/worker.py",
)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)
IMAGE = "sha256:" + "a" * 64


@pytest.fixture
def environment():
    return {
        "TTLANG_CI_DOCKER": "/usr/bin/docker",
        "TTLANG_EMULE_IMAGE": IMAGE,
        "TTLANG_CI_CONTAINER": "ttlang-ci-serial-abc-test",
        "TTLANG_CI_TOKEN": "ttlang-ci-serial-abc",
        "TTLANG_CI_CPUS": "4",
        "TTLANG_CI_MEMORY_GIB": "12",
        "TTLANG_EMULE_BUILD_VOLUME": "ttlang-ci-serial-abc-build",
        "TTLANG_EMULE_CACHE_VOLUME": "ttlang-ci-serial-abc-cache",
        "TTLANG_CI_SOURCE": "/source",
        "TTLANG_CI_GIT_COMMON": "/repo/.git",
        "TTLANG_CI_RESULTS": "/results/private/container",
    }


def launcher_arguments(environment):
    return [
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "--entrypoint",
        "/workspace/scripts/tt-lang-emule-entrypoint.sh",
        "--mount",
        "type=bind,src=/source,dst=/workspace",
        "--mount",
        "type=bind,src=/repo/.git,dst=/repo/.git,readonly",
        "--mount",
        f"type=volume,src={environment['TTLANG_EMULE_BUILD_VOLUME']},dst=/ttlang-build",
        "--mount",
        f"type=volume,src={environment['TTLANG_EMULE_CACHE_VOLUME']},dst=/tt-metal-cache",
        "-e",
        "TTLANG_EMULE_INSTALL=1",
        "--workdir",
        "/workspace",
        IMAGE,
        "scripts/emule-ci/benchmark.py",
        "run",
    ]


def summary():
    return {
        "manifest_digest": "a" * 64,
        "shard": 1,
        "planned_group_count": 2,
        "planned_case_count": 3,
        "group_count": 2,
        "complete_group_count": 2,
        "group_status_counts": {"passed": 2},
        "wall_seconds": 5.5,
        "generic_op_launches": 3,
        "generic_op_returns": 3,
        "runtime_execution_observed": True,
    }


def test_proxy_hardens_existing_launcher_without_replacing_it(environment):
    command = worker.proxy_command(launcher_arguments(environment), environment)
    for option in (
        "--pull=never",
        "--network=none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--cpus=4",
        "--memory=12g",
        "--memory-swap=12g",
        "--pids-limit=512",
        "GIT_CONFIG_VALUE_0=/workspace",
        "PIP_NO_INDEX=1",
        "PYTHONDONTWRITEBYTECODE=1",
        "TTLANG_EMULE_CI_THREADS=4",
        "TTLANG_EMULE_CI_CACHE=/tt-metal-cache",
        "OMP_NUM_THREADS=4",
        "/workspace/build:rw,exec,nosuid,size=512m",
        "type=bind,src=/source,dst=/workspace,readonly",
        "type=bind,src=/results/private/container,dst=/emule-results",
    ):
        assert option in command
    assert command[-3:] == [IMAGE, "scripts/emule-ci/benchmark.py", "run"]


@pytest.mark.parametrize(
    "operation", [["pull", "image"], ["build", "."], ["volume", "rm", "other"]]
)
def test_proxy_rejects_non_launcher_mutations(environment, operation):
    with pytest.raises(ValueError):
        worker.proxy_command(operation, environment)


@pytest.mark.parametrize(
    "extra",
    [
        ["--privileged"],
        ["--device", "/dev/tenstorrent"],
        ["--network", "host"],
        ["--mount", "type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock"],
        ["--mount", "type=bind,src=/auth,dst=/root/.docker"],
        ["--mount", "type=volume,src=existing-user-volume,dst=/ttlang-build"],
        ["-e", "SECRET=value"],
        ["-e", "TTLANG_EMULE_INSTALL"],
    ],
)
def test_proxy_rejects_unexpected_privileges_and_mounts(environment, extra):
    with pytest.raises(ValueError):
        worker.proxy_command(["run", *extra, IMAGE], environment)


@pytest.mark.parametrize(
    "key,value",
    [
        ("TTLANG_EMULE_IMAGE", "mutable:tag"),
        ("TTLANG_CI_CPUS", "5"),
        ("TTLANG_CI_MEMORY_GIB", "13"),
        ("TTLANG_EMULE_CACHE_VOLUME", "existing-user-volume"),
        ("TTLANG_CI_SOURCE", "/source,extra=mount"),
    ],
)
def test_proxy_rejects_invalid_worker_configuration(environment, key, value):
    environment[key] = value
    with pytest.raises(ValueError):
        worker.proxy_command(launcher_arguments(environment), environment)


def test_volume_measurement_mounts_are_read_only(environment):
    command = worker.proxy_command(["measure"], environment)
    assert "type=volume,src=ttlang-ci-serial-abc-build,dst=/build,readonly" in command
    assert "type=volume,src=ttlang-ci-serial-abc-cache,dst=/cache,readonly" in command
    assert command[-3:] == [IMAGE, "-c", "du -sk /build /cache"]


def test_stats_expose_only_numeric_resource_fields():
    value = worker.numeric_stats(
        json.dumps(
            {
                "CPUPerc": "123.4%",
                "MemUsage": "1.25GiB / 12GiB",
                "Name": "sensitive-container-name",
                "arbitrary": "secret",
            }
        )
    )
    assert value == {"cpu_percent": 123.4, "memory_bytes": 1342177280}


@pytest.mark.parametrize(
    "cpu,memory", [("NaN%", "1B / 2B"), ("-1%", "1B / 2B"), ("0%", "private/path")]
)
def test_invalid_stats_are_not_published(cpu, memory):
    with pytest.raises(ValueError):
        worker.numeric_stats(json.dumps({"CPUPerc": cpu, "MemUsage": memory}))


def test_public_summary_accepts_only_documented_sanitized_contract():
    assert worker.public_test_summary(summary()) == summary()
    value = summary()
    value["command"] = "/private/source.py"
    with pytest.raises(ValueError):
        worker.public_test_summary(value)


@pytest.mark.parametrize(
    "key,value",
    [
        ("manifest_digest", "/private/path"),
        ("wall_seconds", float("nan")),
        ("group_count", True),
        ("runtime_execution_observed", "yes"),
        ("group_status_counts", {"secret": 1}),
    ],
)
def test_public_summary_rejects_untrusted_values(key, value):
    data = summary()
    data[key] = value
    with pytest.raises(ValueError):
        worker.public_test_summary(data)


def test_dry_run_does_not_query_docker_or_create_output(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("dry run must not execute a subprocess")

    monkeypatch.setattr(worker.subprocess, "run", forbidden)
    output = tmp_path / "output"
    assert (
        worker.main(["--image", "local:runtime", "--output", str(output), "--dry-run"])
        == 0
    )
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["stages"] == [
        "install",
        "prepare",
        "collect",
        "test",
    ]


@pytest.mark.parametrize(
    "options",
    [
        ["--cpus", "5"],
        ["--memory-gib", "13"],
        ["--timeout-seconds", "18001"],
        ["--disk-floor-gib", "0"],
        ["--plan", "plan.json"],
        ["--shard", "1"],
        ["--worker", "has spaces"],
    ],
)
def test_cli_rejects_unbounded_or_incomplete_options(options):
    with pytest.raises(SystemExit):
        worker.arguments(["--image", "local:runtime", "--output", "/results", *options])


@pytest.fixture
def instance(tmp_path, monkeypatch):
    monkeypatch.setattr(worker.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(
        worker.shutil, "disk_usage", lambda _: SimpleNamespace(free=200 * 1024**3)
    )
    args = worker.arguments(
        ["--image", "local:runtime", "--output", str(tmp_path / "output")]
    )
    value = worker.Worker(args)
    (value.private / "container/inputs").mkdir(parents=True)
    return value


def test_missing_image_stops_before_volume_creation_or_install(instance, monkeypatch):
    calls = []

    def capture(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, "", "image unavailable")

    monkeypatch.setattr(instance, "capture", capture)
    with pytest.raises(RuntimeError, match="already exist"):
        instance.prepare()
    assert len(calls) == 1
    assert calls[0][1:3] == ["image", "inspect"]


def test_dirty_source_stops_before_volume_creation(instance, monkeypatch):
    outputs = iter([IMAGE + "\n", "a" * 40 + "\n", "/repo/.git\n", " M changed.py\n"])
    calls = []

    def capture(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, next(outputs), "")

    monkeypatch.setattr(instance, "capture", capture)
    with pytest.raises(RuntimeError, match="commit the experiment"):
        instance.prepare()
    assert not any("volume" in command for command in calls)


@pytest.mark.parametrize("test_summary,expected", [(None, 1), (summary(), 0)])
def test_success_requires_sanitized_runtime_execution_evidence(
    instance, monkeypatch, test_summary, expected
):
    def prepare():
        instance.revision = "a" * 40
        instance.environment["TTLANG_EMULE_IMAGE"] = IMAGE

    monkeypatch.setattr(instance, "prepare", prepare)
    monkeypatch.setattr(instance, "sizes", lambda _: None)
    monkeypatch.setattr(instance, "stage", lambda *args: 0)
    if test_summary:
        report = instance.private / "container/run/summary.json"
        report.parent.mkdir()
        report.write_text(json.dumps(test_summary))
    assert instance.execute() == expected


def test_failed_install_does_not_run_tests(instance, monkeypatch):
    def prepare():
        instance.revision = "a" * 40
        instance.environment["TTLANG_EMULE_IMAGE"] = IMAGE

    calls = []
    monkeypatch.setattr(instance, "prepare", prepare)
    monkeypatch.setattr(instance, "sizes", lambda _: None)
    monkeypatch.setattr(
        instance, "stage", lambda name, command: calls.append(name) or 1
    )
    assert instance.execute() == 1
    assert calls == ["install"]


@pytest.mark.parametrize("supplied_manifest", [False, True])
def test_test_tools_are_prepared_before_every_execution(
    instance, monkeypatch, supplied_manifest
):
    def prepare():
        instance.revision = "a" * 40
        instance.environment["TTLANG_EMULE_IMAGE"] = IMAGE

    if supplied_manifest:
        instance.args.manifest = instance.output / "manifest.json"
        instance.args.manifest.write_text("{}")
    calls = []
    monkeypatch.setattr(instance, "prepare", prepare)
    monkeypatch.setattr(instance, "sizes", lambda _: None)
    monkeypatch.setattr(
        instance, "stage", lambda name, command: calls.append((name, command)) or 0
    )
    instance.execute()
    assert [name for name, _ in calls] == [
        "install",
        "prepare",
        *([] if supplied_manifest else ["collect"]),
        "test",
    ]
    assert calls[1][1][-2:] == ["scripts/emule-ci/worker.py", "--prepare-test-tools"]


def test_missing_test_tools_stop_before_collection(instance, monkeypatch):
    def prepare():
        instance.revision = "a" * 40
        instance.environment["TTLANG_EMULE_IMAGE"] = IMAGE

    calls = []
    monkeypatch.setattr(instance, "prepare", prepare)
    monkeypatch.setattr(instance, "sizes", lambda _: None)
    monkeypatch.setattr(
        instance,
        "stage",
        lambda name, command: calls.append(name) or int(name == "prepare"),
    )
    assert instance.execute() == 1
    assert calls == ["install", "prepare"]


def test_prepare_uses_existing_cmake_test_tools_target():
    assert worker.test_tools_command(
        {"TT_METAL_EMULE_MODE": "1", "TTLANG_EMULE_CI_THREADS": "2"}
    ) == [
        "cmake",
        "--build",
        "/ttlang-build",
        "--target",
        "ttlang-test-tools",
        "--parallel",
        "2",
    ]
    with pytest.raises(ValueError):
        worker.test_tools_command({"TTLANG_EMULE_CI_THREADS": "2"})


def test_exited_docker_client_still_cleans_up_owned_container(instance, monkeypatch):
    class ExitedProcess:
        returncode = 1

        def poll(self):
            return self.returncode

    monkeypatch.setattr(
        worker.subprocess, "Popen", lambda *args, **kwargs: ExitedProcess()
    )
    cleaned = []
    monkeypatch.setattr(instance, "stop_owned_container", cleaned.append)
    assert instance.stage("test", ["failed-docker-client"]) == 1
    assert cleaned == [instance.token + "-test"]


def test_timeout_cleanup_does_not_remove_unowned_container(instance, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "another-owner\n", "")

    monkeypatch.setattr(worker.subprocess, "run", run)
    instance.stop_owned_container(instance.token + "-test")
    assert len(calls) == 1
    assert "rm" not in calls[0]


def test_long_hosted_timeout_is_explicit_and_local_default_stays_bounded():
    arguments = ["--image", "local:runtime", "--output", "/results"]
    assert worker.arguments(arguments).timeout_seconds == 2700
    assert (
        worker.arguments(arguments + ["--timeout-seconds", "18000"]).timeout_seconds
        == 18000
    )


def test_low_disk_stops_before_image_inspection(instance, monkeypatch):
    monkeypatch.setattr(
        worker.shutil, "disk_usage", lambda _: SimpleNamespace(free=1024)
    )
    monkeypatch.setattr(
        instance, "prepare", lambda: pytest.fail("must not start Docker")
    )
    with pytest.raises(RuntimeError, match="disk"):
        instance.execute()
    assert instance.summary["low_disk"] is True


def test_low_disk_sample_is_recorded_before_stopping(instance, monkeypatch):
    instance.started = worker.time.monotonic()
    instance.environment["TTLANG_CI_CONTAINER"] = instance.token + "-test"
    monkeypatch.setattr(
        worker.shutil, "disk_usage", lambda _: SimpleNamespace(free=1024)
    )
    monkeypatch.setattr(
        instance,
        "capture",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 1, "", ""),
    )
    with pytest.raises(RuntimeError, match="disk"):
        instance.sample("test")
    assert instance.summary["low_disk"] is True
    assert instance.summary["samples"][-1]["host_disk_free_bytes"] == 1024


def test_private_parent_protects_container_writable_child(tmp_path, monkeypatch):
    monkeypatch.setattr(worker.Worker, "execute", lambda self: 0)
    output = tmp_path / "result"
    original_umask = os.umask(0o077)
    try:
        assert worker.main(["--image", "local:runtime", "--output", str(output)]) == 0
    finally:
        os.umask(original_umask)
    assert output.stat().st_mode & 0o777 == 0o700
    assert (output / "private").stat().st_mode & 0o777 == 0o700
    assert (output / "private/container").stat().st_mode & 0o777 == 0o777
    assert (output / "private/container/inputs").stat().st_mode & 0o777 == 0o755


def test_unignored_build_mountpoint_is_never_created(instance, monkeypatch, tmp_path):
    instance.source = tmp_path / "source"
    instance.source.mkdir()
    answers = iter([(0, IMAGE), (0, "a" * 40), (0, "/repo/.git"), (0, ""), (1, "")])

    def capture(command, **kwargs):
        code, output = next(answers)
        return subprocess.CompletedProcess(command, code, output, "")

    monkeypatch.setattr(instance, "capture", capture)
    with pytest.raises(RuntimeError, match="ignored directory"):
        instance.prepare()
    assert not (instance.source / "build").exists()
