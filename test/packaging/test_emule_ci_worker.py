# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import builtins
import json
import os
from pathlib import Path
import subprocess
import sys
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
    suites = {
        suite: {
            "planned_groups": 0,
            "planned_cases": 0,
            "groups": 0,
            "complete_groups": 0,
            "status_counts": {},
            "outcome_counts": {},
        }
        for suite in worker.SUITES
    }
    suites["pytest"] = {
        "planned_groups": 2,
        "planned_cases": 3,
        "groups": 2,
        "complete_groups": 2,
        "status_counts": {"passed": 2},
        "outcome_counts": {"passed": 2, "unsupported": 1},
    }
    return {
        "manifest_digest": "a" * 64,
        "selection": "complete",
        "shard": 1,
        "planned_group_count": 2,
        "planned_case_count": 3,
        "group_count": 2,
        "complete_group_count": 2,
        "group_status_counts": {"passed": 2},
        "case_outcome_counts": {"passed": 2, "unsupported": 1},
        "suite_counts": suites,
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


def test_preflight_uses_image_python_without_compiler_or_device(environment):
    command = worker.proxy_command(["preflight"], environment)
    assert "--entrypoint=/opt/ttlang-toolchain/venv/bin/python" in command
    assert "type=bind,src=/source,dst=/workspace,readonly" in command
    assert (
        "PYTHONPATH=/opt/tt-emule-runtime/tt-metal/ttnn:/opt/tt-emule-runtime/tt-metal/tools"
        in command
    )
    assert "--network=none" in command
    assert "--device" not in command
    assert not any("type=volume" in part for part in command)
    assert command[-3:] == [
        IMAGE,
        "/workspace/scripts/emule-ci/worker.py",
        "--runtime-preflight",
    ]


@pytest.fixture
def runtime_modules(monkeypatch):
    torch = SimpleNamespace(version=SimpleNamespace(cuda=None, hip=None))
    memory_config = SimpleNamespace(
        experimental_set_range_lockstep_allocation=lambda _: None
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules, "ttnn", SimpleNamespace(MemoryConfig=memory_config)
    )
    monkeypatch.setitem(sys.modules, "lit", SimpleNamespace())
    return torch, memory_config


def test_runtime_preflight_checks_packages_without_allocating(
    runtime_modules, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs))
        or subprocess.CompletedProcess(command, 0),
    )
    assert worker.runtime_preflight() == 0
    assert calls == [
        ([sys.executable, "-m", "pip", "check"], {"check": False, "timeout": 60})
    ]


@pytest.mark.parametrize("accelerator", ["cuda", "hip"])
def test_runtime_preflight_rejects_accelerator_torch(runtime_modules, accelerator):
    setattr(runtime_modules[0].version, accelerator, "version")
    assert worker.runtime_preflight() == worker.PreflightFailure.CPU_ONLY_TORCH_REQUIRED


def test_runtime_preflight_rejects_missing_binding_before_pip_check(
    runtime_modules, monkeypatch
):
    del runtime_modules[1].experimental_set_range_lockstep_allocation
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("pip must not run"),
    )
    assert (
        worker.runtime_preflight() == worker.PreflightFailure.RANGE_LOCKSTEP_API_MISSING
    )


def test_runtime_preflight_preserves_pip_check_failure(runtime_modules, monkeypatch):
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 1),
    )
    assert worker.runtime_preflight() == worker.PreflightFailure.PIP_CHECK_FAILED


@pytest.mark.parametrize("module", ["torch", "ttnn", "pytest", "lit"])
@pytest.mark.parametrize("exception", [ImportError, OSError, RuntimeError])
def test_runtime_preflight_import_failures_have_fixed_cli_exit(
    runtime_modules, monkeypatch, module, exception, capsys
):
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == module:
            raise exception("private-import-error-sentinel")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    assert (
        worker.main(["--runtime-preflight"])
        == worker.PreflightFailure.RUNTIME_IMPORT_FAILED
    )
    output = capsys.readouterr()
    assert "private-import-error-sentinel" not in output.out + output.err


@pytest.mark.parametrize(
    "exception,expected",
    [
        (
            subprocess.TimeoutExpired("private-pip-command-sentinel", 60),
            worker.PreflightFailure.PIP_CHECK_TIMED_OUT,
        ),
        (
            OSError("private-pip-error-sentinel"),
            worker.PreflightFailure.PIP_CHECK_FAILED,
        ),
    ],
)
def test_runtime_preflight_pip_errors_have_fixed_cli_exit(
    runtime_modules, monkeypatch, exception, expected, capsys
):
    def failing_run(*args, **kwargs):
        raise exception

    monkeypatch.setattr(worker.subprocess, "run", failing_run)
    assert worker.main(["--runtime-preflight"]) == expected
    output = capsys.readouterr()
    assert "sentinel" not in output.out + output.err


@pytest.fixture
def smoke_harness(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "benchmark", Path(worker.__file__).with_name("benchmark.py")
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    monkeypatch.setitem(sys.modules, "benchmark", benchmark)
    monkeypatch.setattr(
        worker,
        "Path",
        lambda value: (
            tmp_path / value.lstrip("/")
            if value.startswith(("/workspace", "/ttlang-build", "/emule-results"))
            else Path(value)
        ),
    )
    monkeypatch.setattr(benchmark, "validate_source", lambda *args: None)
    monkeypatch.setenv("TT_METAL_EMULE_MODE", "1")
    monkeypatch.setenv("TT_METAL_MOCK_CLUSTER_DESC_PATH", "/mock-descriptor.yaml")
    monkeypatch.setenv("TTLANG_EMULE_CI_CACHE", str(tmp_path / "cache"))
    for key in ("TTLANG_COMPILE_ONLY", "TTLANG_SIM_ONLY", "TT_METAL_SIMULATOR"):
        monkeypatch.delenv(key, raising=False)
    return benchmark


@pytest.mark.parametrize(
    "code,outcomes,returns,expected",
    [
        (0, {"passed": 1}, 1, 0),
        (0, {"skipped": 1}, 1, 1),
        (0, {"passed": 1}, 0, 1),
        (1, {"failed": 1}, 1, 1),
    ],
)
def test_smoke_requires_numerical_pass_and_observed_execution(
    smoke_harness, monkeypatch, tmp_path, code, outcomes, returns, expected
):
    calls = []

    def run(args, env):
        calls.append((args, env))
        manifest = smoke_harness.read_json(args.manifest)
        assert manifest["version"] == 2 and manifest["selection"] == "complete"
        assert manifest["groups"] == [
            {
                "id": "pytest:python/pipe/test_broadcast_2d.py",
                "kind": "pytest",
                "path": "python/pipe/test_broadcast_2d.py",
                "cases": ["python/pipe/test_broadcast_2d.py::test_broadcast_2d"],
                "markers": {"multi_device": []},
            }
        ]
        smoke_harness.write_json(
            args.output / "results.json",
            {
                "manifest_digest": smoke_harness.digest(manifest),
                "shard": 1,
                "groups": [
                    {
                        "id": "pytest:python/pipe/test_broadcast_2d.py",
                        "outcomes": outcomes,
                        "complete": True,
                        "status": "passed" if code == 0 else "failed",
                        "wall_seconds": 1,
                        "launch_counts": {"launch": 1, "returned": returns},
                    }
                ],
            },
        )
        return code

    monkeypatch.setattr(smoke_harness, "run", run)
    assert worker.runtime_smoke("a" * 40, IMAGE) == expected
    args, env = calls[0]
    assert env["TT_METAL_CACHE"] == str(tmp_path / "cache/smoke")
    assert env["TT_EMULE_JIT_CACHE_DIR"] == str(tmp_path / "cache/smoke/emule-jit")
    args.output = tmp_path / "emule-results/run"
    assert smoke_harness.environment(args)["TT_METAL_CACHE"] == str(
        tmp_path / "cache/run"
    )


def test_smoke_refuses_reused_cache(smoke_harness, tmp_path, monkeypatch):
    (tmp_path / "cache/smoke").mkdir(parents=True)
    monkeypatch.setattr(
        smoke_harness, "run", lambda *args: pytest.fail("must not execute")
    )
    with pytest.raises(RuntimeError, match="fresh cache"):
        worker.runtime_smoke("a" * 40, IMAGE)


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
        ("selection", "not multi_device"),
        ("case_outcome_counts", {"passed": "/private/log"}),
        ("case_outcome_counts", {"private_reason": 1}),
        ("suite_counts", {}),
        ("suite_counts", {"python/private_test.py": {}}),
    ],
)
def test_public_summary_rejects_untrusted_values(key, value):
    data = summary()
    data[key] = value
    with pytest.raises(ValueError):
        worker.public_test_summary(data)


@pytest.mark.parametrize(
    "key,value",
    [
        ("planned_cases", -1),
        ("groups", "2"),
        ("status_counts", {"private": 1}),
        ("outcome_counts", {"unsupported": 1.5}),
        ("log", "/private/path"),
    ],
)
def test_public_summary_rejects_untrusted_suite_counts(key, value):
    data = summary()
    data["suite_counts"]["mlir"][key] = value
    with pytest.raises(ValueError):
        worker.public_test_summary(data)


def worker_summary(**overrides):
    value = {
        "schema_version": 1,
        "publication_blocked": True,
        "status": "passed",
        "cpus": 4,
        "memory_limit_bytes": 12 * 1024**3,
        "timeout_seconds": 13200,
        "disk_floor_bytes": 2 * 1024**3,
        "low_disk": False,
        "stages": {
            "preflight": {"exit_code": 0, "seconds": 12.2},
            "test": {"exit_code": 1, "seconds": 7000.6},
        },
        "samples": [{"stage": "test", "host_disk_free_bytes": 1}],
        "smoke": summary(),
        "tests": summary(),
    }
    value.update(overrides)
    return value


def test_benchmark_public_summary_satisfies_the_worker_contract(smoke_harness):
    benchmark = smoke_harness
    groups = [
        {
            "id": "mlir:ttlang",
            "complete": True,
            "status": "passed",
            "wall_seconds": 90.0,
            "outcomes": {"passed": 589, "unsupported": 2},
            "lit_codes": {"PASS": 589, "UNSUPPORTED": 2},
        },
        {
            "id": "pytest:python/pipe/test_broadcast_2d.py",
            "complete": True,
            "status": "failed",
            "wall_seconds": 12.5,
            "outcomes": {"passed": 3, "failed": 1},
            "launch_counts": {"launch": 4, "returned": 3},
            "command": ["/private/python", "-m", "pytest"],
            "report_error": "private diagnostic",
        },
        {
            "id": "lit:python/add.py",
            "complete": False,
            "status": "infrastructure_error",
            "wall_seconds": 600.0,
            "timed_out": True,
            "outcomes": {"not_run": 1},
        },
    ]
    results = {
        "manifest_digest": "c" * 64,
        "shard": 1,
        "planned_group_count": 4,
        "planned_case_count": 600,
        "planned_suite_counts": {
            suite: {"groups": 1, "cases": 1} for suite in benchmark.SUITES
        },
        "groups": groups,
    }
    for value in (results, {"manifest_digest": "c" * 64, "shard": 1, "groups": []}):
        public = benchmark.public_summary(value)
        assert worker.public_test_summary(public) == public
        assert "private" not in json.dumps(public)
    public = benchmark.public_summary(results)
    rendered = worker.report_summary(
        {"schema_version": 1, "status": "failed", "stages": {}, "tests": public}
    )
    assert "| mlir | 1 | 1 | 1 | 1 | passed 589, unsupported 2 | passed 1 |" in rendered
    assert "| lit | 1 | 1 | 0 | 1 | not_run 1 | infrastructure_error 1 |" in rendered
    assert "private" not in rendered


def test_report_renders_per_suite_table_from_public_fields_only():
    report = worker.report_summary(worker_summary())
    assert report.startswith("Worker status: `passed`\n")
    assert "Stage `preflight`: exit 0, 12 s" in report
    assert "Stage `test`: exit 1, 7001 s" in report
    assert "Representative smoke: runtime execution observed True" in report
    assert "3 planned cases in 2 groups; 2 groups completed" in report
    assert "| Suite | Planned groups | Run | Complete | Planned cases" in report
    assert "| pytest | 2 | 2 | 2 | 3 | passed 2, unsupported 1 | passed 2 |" in report
    assert "| mlir | 0 | 0 | 0 | 0 | - | - |" in report
    assert "All suites: cases passed 2, unsupported 1; groups passed 2." in report
    assert "`unsupported` counts cases" in report
    assert "private" not in report


def test_report_without_inventory_names_the_fixed_preflight_reason():
    value = worker_summary(
        status="failed",
        failure_reason="cpu_only_torch_required",
        stages={"preflight": {"exit_code": 21, "seconds": 1}},
    )
    del value["smoke"], value["tests"]
    report = worker.report_summary(value)
    assert "Preflight failure: `cpu_only_torch_required`" in report
    assert "Complete compiler inventory: no validated summary was recorded." in report
    assert "| Suite |" not in report


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": 2},
        {"status": "private failure text"},
        {"failure_reason": "Traceback: /private/path"},
        {"stages": {"test": {"exit_code": 1, "seconds": "7000s"}}},
        {"stages": {"test": {"exit_code": "1", "seconds": 1}}},
        {"tests": {**summary(), "command": "/private/source.py"}},
        {"smoke": {**summary(), "log": "/private/log"}},
    ],
)
def test_report_refuses_undocumented_or_private_fields(overrides):
    with pytest.raises(ValueError):
        worker.report_summary(worker_summary(**overrides))


def test_report_cli_prints_markdown_and_fails_closed(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(worker_summary()))
    result = subprocess.run(
        [sys.executable, worker.__file__, "--report-summary", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == worker.report_summary(worker_summary())
    path.write_text(json.dumps(worker_summary(tests={"private": "/private/path"})))
    result = subprocess.run(
        [sys.executable, worker.__file__, "--report-summary", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "/private/path" not in result.stdout + result.stderr
    for arguments in ([], [str(path), "extra"], [str(tmp_path / "missing.json")]):
        result = subprocess.run(
            [sys.executable, worker.__file__, "--report-summary", *arguments],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2 and result.stdout == ""


@pytest.mark.parametrize(
    "options",
    [
        ["--cpus", "5"],
        ["--memory-gib", "13"],
        ["--timeout-seconds", "18001"],
        ["--disk-floor-gib", "0"],
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
    (value.private / "container").mkdir(parents=True)
    report = value.private / "container/smoke/summary.json"
    report.parent.mkdir()
    report.write_text(json.dumps(summary()))
    return value


def test_missing_image_stops_before_install(instance, monkeypatch):
    calls = []

    def capture(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, "", "image unavailable")

    monkeypatch.setattr(instance, "capture", capture)
    with pytest.raises(RuntimeError, match="already exist"):
        instance.prepare()
    assert len(calls) == 1
    assert calls[0][1:3] == ["image", "inspect"]


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
        instance,
        "stage",
        lambda name, command: calls.append(name) or int(name == "install"),
    )
    assert instance.execute() == 1
    assert calls == ["preflight", "install"]


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("preflight", ["preflight"]),
        ("smoke", ["preflight", "install", "prepare", "smoke"]),
    ],
)
def test_preflight_or_smoke_failure_stops_before_benchmark(
    instance, monkeypatch, failure, expected
):
    def prepare():
        instance.revision = "a" * 40
        instance.environment["TTLANG_EMULE_IMAGE"] = IMAGE

    calls = []
    monkeypatch.setattr(instance, "prepare", prepare)
    monkeypatch.setattr(instance, "sizes", lambda _: None)
    monkeypatch.setattr(
        instance,
        "stage",
        lambda name, command: calls.append(name) or int(name == failure),
    )
    assert instance.execute() == 1
    assert calls == expected


@pytest.mark.parametrize(
    "code,reason",
    [
        (20, "runtime_import_failed"),
        (21, "cpu_only_torch_required"),
        (22, "range_lockstep_api_missing"),
        (23, "pip_check_failed"),
        (24, "pip_check_timed_out"),
        (1, "preflight_failure"),
        (124, "preflight_failure"),
        (255, "preflight_failure"),
    ],
)
def test_preflight_summary_publishes_only_fixed_reason(
    tmp_path, monkeypatch, capsys, code, reason
):
    output = tmp_path / "result"
    calls = []

    def prepare(self):
        self.revision = "a" * 40
        self.environment["TTLANG_EMULE_IMAGE"] = IMAGE

    def stage(self, name, command):
        calls.append(name)
        (self.private / "preflight.log").write_text("private-preflight-error-sentinel")
        return code

    monkeypatch.setattr(worker.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(
        worker.shutil, "disk_usage", lambda _: SimpleNamespace(free=200 * 1024**3)
    )
    monkeypatch.setattr(worker.Worker, "prepare", prepare)
    monkeypatch.setattr(worker.Worker, "sizes", lambda *args: None)
    monkeypatch.setattr(worker.Worker, "stage", stage)
    original_umask = os.umask(0o077)
    try:
        assert worker.main(["--image", "local:runtime", "--output", str(output)]) == 1
    finally:
        os.umask(original_umask)
    public = capsys.readouterr().out
    assert json.loads(public)["failure_reason"] == reason
    assert json.loads((output / "summary.json").read_text())["failure_reason"] == reason
    assert "private-preflight-error-sentinel" not in public
    assert (
        output / "private/preflight.log"
    ).read_text() == "private-preflight-error-sentinel"
    assert calls == ["preflight"]


def test_smoke_without_runtime_evidence_stops_before_collection(instance, monkeypatch):
    def prepare():
        instance.revision = "a" * 40
        instance.environment["TTLANG_EMULE_IMAGE"] = IMAGE

    report = summary()
    report["runtime_execution_observed"] = False
    (instance.private / "container/smoke/summary.json").write_text(json.dumps(report))
    calls = []
    monkeypatch.setattr(instance, "prepare", prepare)
    monkeypatch.setattr(instance, "sizes", lambda _: None)
    monkeypatch.setattr(
        instance, "stage", lambda name, command: calls.append(name) or 0
    )
    assert instance.execute() == 1
    assert calls == ["preflight", "install", "prepare", "smoke"]


def test_test_tools_are_prepared_before_every_execution(instance, monkeypatch):
    def prepare():
        instance.revision = "a" * 40
        instance.environment["TTLANG_EMULE_IMAGE"] = IMAGE

    calls = []
    monkeypatch.setattr(instance, "prepare", prepare)
    monkeypatch.setattr(instance, "sizes", lambda _: None)
    monkeypatch.setattr(
        instance, "stage", lambda name, command: calls.append((name, command)) or 0
    )
    instance.execute()
    assert [name for name, _ in calls] == [
        "preflight",
        "install",
        "prepare",
        "smoke",
        "collect",
        "test",
    ]
    assert calls[2][1][-2:] == ["scripts/emule-ci/worker.py", "--prepare-test-tools"]
    assert calls[4][1][-2:] == ["--output", "/emule-results/collection"]
    command = calls[5][1]
    assert command[command.index("--manifest") + 1] == (
        "/emule-results/collection/manifest.json"
    )


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
    assert calls == ["preflight", "install", "prepare"]


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


def test_unignored_build_mountpoint_is_never_created(instance, monkeypatch, tmp_path):
    instance.source = tmp_path / "source"
    instance.source.mkdir()
    answers = iter([(0, IMAGE), (0, "a" * 40), (0, "/repo/.git"), (1, "")])

    def capture(command, **kwargs):
        code, output = next(answers)
        return subprocess.CompletedProcess(command, code, output, "")

    monkeypatch.setattr(instance, "capture", capture)
    with pytest.raises(RuntimeError, match="ignored directory"):
        instance.prepare()
    assert not (instance.source / "build").exists()
