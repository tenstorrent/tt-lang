# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts/emule-ci"
SPEC = importlib.util.spec_from_file_location(
    "emule_benchmark", SCRIPT_DIR / "benchmark.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def manifest():
    return benchmark.make_manifest(
        [
            "me2e/test_ops.py::TestAdd::test_build_module",
            "me2e/test_ops.py::TestAdd::test_execute",
            "me2e/test_ops.py::TestAdd::test_validate_golden",
            "python/test_add.py::test_add[1]",
            "python/test_add.py::test_add[2]",
            "python/test_matmul.py::test_matmul",
        ],
        "-- Available Tests --\n  TTLang :: python/add.py\n",
        "a" * 40,
        "sha256:" + "b" * 64,
    )


def results(inventory):
    return {
        "manifest_digest": benchmark.digest(inventory),
        "shard": 1,
        "groups": [
            {
                "id": group["id"],
                "wall_seconds": float(index + 1),
                "complete": True,
                "status": "passed",
                "launch_counts": {"launch": 1, "returned": 1},
            }
            for index, group in enumerate(inventory["groups"])
        ],
    }


def test_collection_preserves_ordered_me2e_stages():
    inventory = manifest()
    assert len(inventory["groups"]) == 4
    stages = inventory["groups"][0]["cases"]
    assert [stage.rsplit("::", 1)[1] for stage in stages] == [
        "test_build_module",
        "test_execute",
        "test_validate_golden",
    ]
    assert len(inventory["groups"][1]["cases"]) == 2


def test_balance_is_deterministic_complete_and_disjoint():
    inventory = manifest()
    timing = results(inventory)
    plan = benchmark.balance(inventory, timing, 3)
    assert plan == benchmark.balance(inventory, timing, 3)
    assert plan["estimated_seconds"] == [4.0, 3.0, 3.0]
    groups = [
        group
        for index in range(1, 4)
        for group in benchmark.selected_groups(inventory, plan, index)
    ]
    assert sorted(group["id"] for group in groups) == sorted(
        group["id"] for group in inventory["groups"]
    )
    assert inventory["groups"][0] in groups


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "2"])
def test_balance_rejects_invalid_timings(value):
    inventory = manifest()
    timing = results(inventory)
    timing["groups"][0]["wall_seconds"] = value
    with pytest.raises(ValueError, match="finite positive"):
        benchmark.balance(inventory, timing, 3)


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "duplicate",
        "incomplete",
        "infrastructure",
        "wrong_revision",
        "no_execution",
    ],
)
def test_balance_rejects_invalid_baseline(problem):
    inventory = manifest()
    timing = results(inventory)
    if problem == "missing":
        timing["groups"].pop()
    elif problem == "duplicate":
        timing["groups"].append(timing["groups"][0])
    elif problem == "incomplete":
        timing["groups"][0]["complete"] = False
    elif problem == "infrastructure":
        timing["groups"][0]["status"] = "infrastructure_error"
    elif problem == "wrong_revision":
        timing["manifest_digest"] = "wrong"
    else:
        for group in timing["groups"]:
            group["launch_counts"] = {}
    with pytest.raises(ValueError):
        benchmark.balance(inventory, timing, 3)


def test_completed_test_failures_remain_in_timing_baseline():
    inventory = manifest()
    timing = results(inventory)
    timing["groups"][0]["status"] = "failed"
    assert len(benchmark.balance(inventory, timing, 3)["shards"]) == 3


@pytest.mark.parametrize(
    "problem", ["duplicate", "missing", "unknown", "empty", "digest"]
)
def test_rejects_invalid_shard_coverage(problem):
    inventory = manifest()
    plan = benchmark.balance(inventory, results(inventory), 3)
    if problem == "duplicate":
        plan["shards"][0].append(plan["shards"][1][0])
    elif problem == "missing":
        plan["shards"][0].pop()
    elif problem == "unknown":
        plan["shards"][0][0] = "pytest:python/missing.py"
    elif problem == "empty":
        plan["shards"].append([])
    else:
        plan["manifest_digest"] = "wrong"
    with pytest.raises(ValueError):
        benchmark.selected_groups(inventory, plan, 1)


@pytest.mark.parametrize(
    "changed", ["revision", "runtime", "group_order", "case_order", "parameters"]
)
def test_plan_rejects_changed_recollection(changed):
    inventory = manifest()
    plan = benchmark.balance(inventory, results(inventory), 3)
    if changed == "revision":
        inventory["revision"] = "c" * 40
    elif changed == "runtime":
        inventory["runtime_id"] = "sha256:" + "c" * 64
    elif changed == "group_order":
        inventory["groups"].reverse()
    elif changed == "case_order":
        inventory["groups"][0]["cases"].reverse()
    else:
        inventory["groups"][1]["cases"][0] = "python/test_add.py::test_add[changed]"
    benchmark.validate_manifest(inventory)
    with pytest.raises(ValueError, match="plan belongs to another manifest"):
        benchmark.selected_groups(inventory, plan, 1)


def test_native_abort_is_not_complete_coverage():
    group = manifest()["groups"][0]
    events = [{"kind": "collection", "nodes": group["cases"]}]
    events += [{"kind": "finish", "nodeid": node} for node in group["cases"][:-1]]
    assert not benchmark.pytest_coverage(group, events)
    events.append({"kind": "finish", "nodeid": group["cases"][-1]})
    assert benchmark.pytest_coverage(group, events)
    events.append(events[-1])
    assert not benchmark.pytest_coverage(group, events)


def test_public_summary_excludes_commands_paths_and_errors():
    timing = results(manifest())
    timing["groups"][0].update(
        command=["secret"], report_error="private log", log="/private/path"
    )
    summary = benchmark.public_summary(timing)
    assert summary["runtime_execution_observed"]
    assert summary["wall_seconds"] == 10.0
    assert "private" not in json.dumps(summary)
    assert "secret" not in json.dumps(summary)
    assert "python/" not in json.dumps(summary)


@pytest.mark.parametrize(
    "path",
    [
        "../python/test_a.py",
        "/python/test_a.py",
        "python/../test_a.py",
        "python//test_a.py",
        "python/test_\na.py",
        "python/test_\ra.py",
        "python/test_\ta.py",
        "python/test_\x1ba.py",
        "python/test_ a.py",
        "python/test_:a.py",
    ],
)
def test_rejects_unsafe_test_paths(path):
    with pytest.raises(ValueError, match="invalid test group"):
        benchmark.make_manifest(
            [path + "::test_a"],
            "  TTLang :: python/add.py",
            "a" * 40,
            "sha256:" + "b" * 64,
        )


def test_command_records_launch_failure_and_timeout(tmp_path):
    failure = benchmark.run_command(
        [str(tmp_path / "missing")], tmp_path / "missing.log", 5, os.environ, tmp_path
    )
    assert failure["exit_code"] is None
    assert "error" in failure
    timeout = benchmark.run_command(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        tmp_path / "timeout.log",
        0.1,
        os.environ,
        tmp_path,
    )
    assert timeout["timed_out"]
    assert timeout["exit_code"] < 0


def test_observer_collects_without_importing_ttnn(tmp_path):
    (tmp_path / "test_sample.py").write_text(
        "def test_first(): pass\ndef test_second(): pass\n"
    )
    (tmp_path / "ttnn.py").write_text(
        "raise RuntimeError('collection must not import ttnn')\n"
    )
    events = tmp_path / "events.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "observer",
            "-c",
            os.devnull,
            "--rootdir",
            str(tmp_path),
            str(tmp_path / "test_sample.py"),
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(SCRIPT_DIR) + os.pathsep + str(tmp_path),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTEST_ADDOPTS": "",
            "TTLANG_EMULE_CI_EVENTS": str(events),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert benchmark.read_events(events) == [
        {
            "kind": "collection",
            "nodes": ["test_sample.py::test_first", "test_sample.py::test_second"],
        }
    ]


def test_environment_rejects_compile_only(monkeypatch, tmp_path):
    monkeypatch.setenv("TT_METAL_EMULE_MODE", "1")
    monkeypatch.setenv("TT_METAL_MOCK_CLUSTER_DESC_PATH", "descriptor.yaml")
    monkeypatch.setenv("TTLANG_COMPILE_ONLY", "1")
    with pytest.raises(ValueError, match="activate"):
        benchmark.environment(SimpleNamespace(output=tmp_path))


@pytest.mark.parametrize(
    "runtime_id", ["latest", "local-image:v1", "sha256:123", "a@sha256:" + "g" * 64]
)
def test_rejects_mutable_runtime_identity(runtime_id):
    inventory = manifest()
    inventory["runtime_id"] = runtime_id
    with pytest.raises(ValueError, match="immutable"):
        benchmark.validate_manifest(inventory)


@pytest.mark.parametrize(
    "head,dirty", [("b" * 40, ""), ("a" * 40, " M test/python/test_add.py\n")]
)
def test_source_revision_and_cleanliness_are_verified(
    monkeypatch, tmp_path, head, dirty
):
    responses = iter([SimpleNamespace(stdout=head), SimpleNamespace(stdout=dirty)])
    monkeypatch.setattr(
        benchmark.subprocess, "run", lambda *args, **kwargs: next(responses)
    )
    with pytest.raises(ValueError):
        benchmark.validate_source(tmp_path, "a" * 40)


def test_unrecognized_lit_entry_is_not_silently_omitted():
    with pytest.raises(ValueError, match="unexpected lit"):
        benchmark.make_manifest(
            ["python/test_add.py::test_add"],
            "  TTLang :: python/add.py\n  OtherSuite :: python/matmul.py\n",
            "a" * 40,
            "sha256:" + "b" * 64,
        )


@pytest.mark.parametrize(
    "observed,status,expected",
    [(False, "passed", 2), (True, "failed", 1), (True, "passed", 0)],
)
def test_run_requires_coverage_and_runtime_execution(
    monkeypatch, tmp_path, observed, status, expected
):
    inventory = manifest()
    benchmark.write_json(tmp_path / "manifest.json", inventory)
    args = SimpleNamespace(
        source=tmp_path,
        build=tmp_path,
        revision=inventory["revision"],
        runtime_id=inventory["runtime_id"],
        manifest=tmp_path / "manifest.json",
        plan=None,
        shard=1,
        output=tmp_path,
        timeout=600,
        group_timeout=7200,
        lit="llvm-lit",
    )

    def fake_run(command, output, limit, env, source):
        if "pytest" in command:
            path = Path(command[-1]).relative_to(tmp_path / "test").as_posix()
            group = next(
                group for group in inventory["groups"] if group["path"] == path
            )
            events = [{"kind": "collection", "nodes": group["cases"]}]
            events += [{"kind": "finish", "nodeid": node} for node in group["cases"]]
            if observed:
                events += [{"kind": "launch"}, {"kind": "returned"}]
            Path(env["TTLANG_EMULE_CI_EVENTS"]).write_text(
                "\n".join(json.dumps(event) for event in events)
            )
        else:
            benchmark.write_json(
                Path(command[command.index("-o") + 1]),
                {"tests": [{"name": "TTLang :: python/add.py", "code": "PASS"}]},
            )
        return {
            "exit_code": int(status == "failed"),
            "timed_out": False,
            "wall_seconds": 1.0,
        }

    monkeypatch.setattr(benchmark, "run_command", fake_run)
    assert benchmark.run(args, {}) == expected
    summary = benchmark.read_json(tmp_path / "summary.json")
    assert summary["runtime_execution_observed"] is observed
    assert summary["planned_group_count"] == summary["complete_group_count"] == 4
    assert summary["planned_case_count"] == 7


def test_lit_build_paths_work_with_separate_execution_root(tmp_path):
    pytest.importorskip("lit.main")
    source, build, execution = (tmp_path / name for name in ("source", "build", "exec"))
    (source / "python").mkdir(parents=True)
    build.mkdir()
    (source / "python/add.py").write_text("# RUN: echo pass > %t\n")
    (build / "lit.site.cfg.py").write_text(
        "import os, lit.formats\n"
        "config.name = 'TTLang'\n"
        "config.suffixes = ['.py']\n"
        "config.test_format = lit.formats.ShTest(False)\n"
        f"config.test_source_root = {str(source)!r}\n"
        "config.test_exec_root = os.environ['TTLANG_LIT_TEST_EXEC_ROOT']\n"
        "os.makedirs(os.path.join(config.test_exec_root, 'python', 'Output'), exist_ok=True)\n"
    )
    command = [sys.executable, "-c", "from lit.main import main; main()"]
    env = {**os.environ, "TTLANG_LIT_TEST_EXEC_ROOT": str(execution)}
    collected = subprocess.run(
        command + ["--show-tests", str(build / "python")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert "TTLang :: python/add.py" in collected.stdout
    output = tmp_path / "lit.json"
    executed = subprocess.run(
        command + ["-j1", "-o", str(output), str(build / "python/add.py")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert executed.returncode == 0, executed.stdout + executed.stderr
    assert benchmark.read_json(output)["tests"][0]["code"] == "PASS"
    assert list((execution / "python/Output").iterdir())


@pytest.mark.parametrize("where", ["root", "group"])
def test_private_manifest_fields_cannot_be_exported(where):
    inventory = manifest()
    target = inventory if where == "root" else inventory["groups"][0]
    target["private_diagnostic"] = "must not be uploaded"
    with pytest.raises(ValueError, match="unexpected.*fields"):
        benchmark.balance(inventory, results(inventory), 3)


def test_balance_exports_only_plan_without_private_parameters(tmp_path):
    inventory = manifest()
    sentinel = "PRIVATE_RUNTIME_PARAMETER"
    inventory["groups"][1]["cases"][0] = f"python/test_add.py::test_add[{sentinel}]"
    benchmark.write_json(tmp_path / "input.json", inventory)
    benchmark.write_json(tmp_path / "results.json", results(inventory))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "benchmark.py"),
            "balance",
            "--manifest",
            str(tmp_path / "input.json"),
            "--results",
            str(tmp_path / "results.json"),
            "--output",
            str(tmp_path / "plan.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    exported = benchmark.read_json(tmp_path / "plan.json")
    assert set(exported) == {"manifest_digest", "shards", "estimated_seconds"}
    assert exported["manifest_digest"] == benchmark.digest(inventory)
    assert sentinel not in json.dumps(exported) + result.stdout + result.stderr
    assert "test_add[" not in json.dumps(exported)
    assert sorted(key for shard in exported["shards"] for key in shard) == sorted(
        group["id"] for group in inventory["groups"]
    )
    assert set(path.name for path in tmp_path.iterdir()) == {
        "input.json",
        "results.json",
        "plan.json",
    }


def test_balance_refuses_obsolete_public_manifest_export(tmp_path):
    inventory = manifest()
    benchmark.write_json(tmp_path / "input.json", inventory)
    benchmark.write_json(tmp_path / "results.json", results(inventory))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "benchmark.py"),
            "balance",
            "--manifest",
            str(tmp_path / "input.json"),
            "--results",
            str(tmp_path / "results.json"),
            "--output",
            str(tmp_path / "plan.json"),
            "--public-manifest",
            str(tmp_path / "public.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unrecognized arguments: --public-manifest" in result.stderr
    assert not (tmp_path / "public.json").exists()
    assert not (tmp_path / "plan.json").exists()


@pytest.mark.parametrize(
    "problem",
    [None, "revision", "dirty", "unknown", "untracked", "symlink", "gitlink"],
)
def test_public_plan_requires_clean_exact_source_and_tracked_regular_files(
    tmp_path, monkeypatch, problem
):
    inventory = manifest()
    source = tmp_path / "source"
    source.mkdir()
    tree = {
        "test/" + group["path"]: ("100755" if index == 0 else "100644", "blob")
        for index, group in enumerate(inventory["groups"])
    }
    path = "test/" + inventory["groups"][0]["path"]
    if problem in ("unknown", "untracked"):
        del tree[path]
        if problem == "untracked":
            untracked = source / path
            untracked.parent.mkdir(parents=True)
            untracked.write_text("untracked source does not prove public provenance")
    elif problem == "symlink":
        tree[path] = ("120000", "blob")
    elif problem == "gitlink":
        tree[path] = ("160000", "commit")
    tree_output = "".join(
        f"{mode} {kind} {'d' * 40}\t{path}\0" for path, (mode, kind) in tree.items()
    )
    answers = iter(
        [
            "c" * 40 if problem == "revision" else inventory["revision"],
            " M changed.py\n" if problem == "dirty" else "",
            tree_output,
        ]
    )
    calls = []

    def git(command, **kwargs):
        calls.append(command)
        assert kwargs == {"capture_output": True, "text": True, "check": True}
        return subprocess.CompletedProcess(command, 0, next(answers), "")

    monkeypatch.setattr(benchmark.subprocess, "run", git)
    benchmark.write_json(tmp_path / "input.json", inventory)
    benchmark.write_json(tmp_path / "results.json", results(inventory))
    output = tmp_path / "plan.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "balance",
            "--manifest",
            str(tmp_path / "input.json"),
            "--results",
            str(tmp_path / "results.json"),
            "--output",
            str(output),
            "--public-source",
            str(source),
        ],
    )
    if problem:
        with pytest.raises(SystemExit) as error:
            benchmark.main()
        assert error.value.code == 2
        assert not output.exists()
    else:
        assert benchmark.main() == 0
        assert benchmark.read_json(output) == benchmark.balance(
            inventory, results(inventory), 3
        )
    if problem not in ("revision", "dirty"):
        assert calls[-1][-6:] == [
            "ls-tree",
            "-r",
            "-z",
            inventory["revision"],
            "--",
            "test",
        ]


@pytest.mark.parametrize("threads", ["1", "2", "3", "4"])
def test_environment_uses_explicit_bounded_threads_and_isolated_cache(
    monkeypatch, tmp_path, threads
):
    for key in ("TTLANG_COMPILE_ONLY", "TTLANG_SIM_ONLY", "TT_METAL_SIMULATOR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TT_METAL_EMULE_MODE", "1")
    monkeypatch.setenv("TT_METAL_MOCK_CLUSTER_DESC_PATH", "descriptor.yaml")
    monkeypatch.setenv("TTLANG_EMULE_CI_THREADS", threads)
    monkeypatch.setenv("TTLANG_EMULE_CI_CACHE", "/tt-metal-cache")
    env = benchmark.environment(SimpleNamespace(output=tmp_path / "run"))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert env[key] == threads
    assert env["TT_METAL_CACHE"] == "/tt-metal-cache/run"
    assert env["TT_EMULE_JIT_CACHE_DIR"] == "/tt-metal-cache/run/emule-jit"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


@pytest.mark.parametrize("threads", ["0", "5", "-1", "1.5", "many"])
def test_environment_rejects_unbounded_threads(monkeypatch, tmp_path, threads):
    for key in (
        "TTLANG_COMPILE_ONLY",
        "TTLANG_SIM_ONLY",
        "TT_METAL_SIMULATOR",
        "TTLANG_EMULE_CI_CACHE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TT_METAL_EMULE_MODE", "1")
    monkeypatch.setenv("TT_METAL_MOCK_CLUSTER_DESC_PATH", "descriptor.yaml")
    monkeypatch.setenv("TTLANG_EMULE_CI_THREADS", threads)
    with pytest.raises(ValueError):
        benchmark.environment(SimpleNamespace(output=tmp_path / "run"))
