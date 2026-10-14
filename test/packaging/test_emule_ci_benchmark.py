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


DEVICE_NODES = [
    "me2e/test_ops.py::TestAdd::test_build_module",
    "me2e/test_ops.py::TestAdd::test_execute",
    "me2e/test_ops.py::TestAdd::test_validate_golden",
    "python/test_add.py::test_add[1]",
    "python/test_add.py::test_add[2]",
    "python/test_matmul.py::test_matmul",
    "python/fabric/test_mesh.py::test_mesh",
]
DEVICE_MARKERS = {"python/fabric/test_mesh.py::test_mesh": ["multi_device"]}
PACKAGING_NODES = [
    "packaging/test_wheel.py::test_metadata",
    "packaging/test_wheel.py::test_layout[a b]",
]
MLIR_TESTS = "-- Available Tests --\n  TTLang :: ttlang/Dialect/TTL/IR/ops.mlir\n  TTLang :: ttlang/Dialect/TTL/IR/ops_invalid.mlir\n"
BINDINGS_TESTS = (
    "-- Available Tests --\n  TTLang :: bindings/python/kernel_headers.py\n"
)
LIT_TESTS = "-- Available Tests --\n  TTLang :: python/add.py\n"


def collected(**overrides):
    inventory = {
        "pytest": {"nodes": list(DEVICE_NODES), "markers": dict(DEVICE_MARKERS)},
        "packaging": {"nodes": list(PACKAGING_NODES)},
        "lit": LIT_TESTS,
        "mlir": MLIR_TESTS,
        "bindings": BINDINGS_TESTS,
    }
    inventory.update(overrides)
    return inventory


def manifest(**overrides):
    return benchmark.make_manifest(
        collected(**overrides), "a" * 40, "sha256:" + "b" * 64
    )


def group(inventory, group_id):
    return next(group for group in inventory["groups"] if group["id"] == group_id)


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


def test_collection_covers_every_suite_without_selection():
    inventory = manifest()
    assert inventory["version"] == 2 and inventory["selection"] == "complete"
    assert [group["id"] for group in inventory["groups"]] == [
        "mlir:ttlang",
        "bindings:bindings/python",
        "packaging:packaging",
        "pytest:me2e/test_ops.py",
        "pytest:python/test_add.py",
        "pytest:python/test_matmul.py",
        "pytest:python/fabric/test_mesh.py",
        "lit:python/add.py",
    ]
    assert group(inventory, "mlir:ttlang")["cases"] == [
        "TTLang :: ttlang/Dialect/TTL/IR/ops.mlir",
        "TTLang :: ttlang/Dialect/TTL/IR/ops_invalid.mlir",
    ]
    assert group(inventory, "packaging:packaging")["cases"] == PACKAGING_NODES
    assert [benchmark.suite_of(group["id"]) for group in inventory["groups"]] == [
        "mlir",
        "bindings",
        "packaging",
        "me2e",
        "pytest",
        "pytest",
        "pytest",
        "lit",
    ]
    assert "markers" not in group(inventory, "mlir:ttlang")
    assert group(inventory, "pytest:python/fabric/test_mesh.py")["markers"] == {
        "multi_device": ["python/fabric/test_mesh.py::test_mesh"],
        "compile_only": [],
    }
    assert group(inventory, "pytest:python/test_add.py")["markers"] == {
        "multi_device": [],
        "compile_only": [],
    }


def test_collection_preserves_ordered_me2e_stages():
    inventory = manifest()
    stages = group(inventory, "pytest:me2e/test_ops.py")["cases"]
    assert [stage.rsplit("::", 1)[1] for stage in stages] == [
        "test_build_module",
        "test_execute",
        "test_validate_golden",
    ]
    assert len(group(inventory, "pytest:python/test_add.py")["cases"]) == 2


@pytest.mark.parametrize(
    "suite", ["pytest", "packaging", "lit", "mlir", "bindings", "me2e"]
)
def test_every_compiler_suite_must_collect_tests(suite):
    empty = {"nodes": []}
    if suite == "me2e":
        override = {
            "pytest": {"nodes": [n for n in DEVICE_NODES if not n.startswith("me2e/")]}
        }
    elif suite in ("pytest", "packaging"):
        override = {suite: empty}
    else:
        override = {suite: "-- Available Tests --\n"}
    with pytest.raises(ValueError, match="every compiler suite|invalid test group"):
        manifest(**override)


@pytest.mark.parametrize(
    "entry",
    [
        "  TTLang :: python/add.py",
        "  TTLang :: ttlang/../escape.mlir",
        "  TTLang :: ttlang/Dialect/ops.txt",
        "  Other :: ttlang/Dialect/ops.mlir",
    ],
)
def test_directory_suites_reject_entries_outside_their_tree(entry):
    with pytest.raises(ValueError, match="unexpected lit collection entry"):
        manifest(mlir="-- Available Tests --\n" + entry + "\n")


def test_classified_cases_must_belong_to_their_group():
    inventory = manifest()
    group(inventory, "pytest:python/test_add.py")["markers"]["multi_device"] = [
        "python/test_matmul.py::test_matmul"
    ]
    with pytest.raises(ValueError, match="classified cases"):
        benchmark.validate_manifest(inventory)
    inventory = manifest()
    del group(inventory, "pytest:python/test_add.py")["markers"]["compile_only"]
    with pytest.raises(ValueError, match="classification markers"):
        benchmark.validate_manifest(inventory)


def test_outcomes_account_for_unsupported_and_never_run_cases():
    inventory = manifest()
    mesh = group(inventory, "pytest:python/fabric/test_mesh.py")
    add = group(inventory, "pytest:python/test_add.py")
    skipped = [
        {
            "kind": "report",
            "nodeid": case,
            "phase": "setup",
            "outcome": "skipped",
            "expected_failure": False,
        }
        for case in mesh["cases"] + add["cases"][:1]
    ]
    assert benchmark.pytest_outcomes(mesh, skipped) == {"unsupported": 1}
    assert benchmark.pytest_outcomes(add, skipped) == {"skipped": 1, "not_run": 1}
    xfail = [dict(skipped[0], expected_failure=True)]
    assert benchmark.pytest_outcomes(mesh, xfail) == {"xfail": 1}
    mlir = group(inventory, "mlir:ttlang")
    tests = [
        {"name": mlir["cases"][0], "code": "UNSUPPORTED"},
        {"name": mlir["cases"][1], "code": "UNRESOLVED"},
    ]
    assert benchmark.lit_outcomes(mlir, tests) == {"unsupported": 1, "incomplete": 1}
    assert not benchmark.lit_coverage(mlir, tests)
    tests = [{"name": name, "code": "PASS"} for name in reversed(mlir["cases"])]
    assert benchmark.lit_coverage(mlir, tests)
    assert benchmark.lit_outcomes(mlir, tests[:1]) == {"passed": 1, "not_run": 1}


def test_public_summary_reports_suites_outcomes_and_unsupported_cases():
    inventory = manifest()
    timing = results(inventory)
    for entry in timing["groups"]:
        entry["outcomes"] = {"passed": 1}
    timing["groups"][-2]["outcomes"] = {"unsupported": 1}
    timing["planned_suite_counts"] = {
        suite: {"groups": 1, "cases": 2} for suite in benchmark.SUITES
    }
    summary = benchmark.public_summary(timing)
    assert summary["selection"] == "complete"
    assert summary["case_outcome_counts"] == {"passed": 7, "unsupported": 1}
    assert set(summary["suite_counts"]) == set(benchmark.SUITES)
    assert summary["suite_counts"]["pytest"] == {
        "planned_groups": 1,
        "planned_cases": 2,
        "groups": 3,
        "complete_groups": 3,
        "status_counts": {"passed": 3},
        "outcome_counts": {"passed": 2, "unsupported": 1},
    }
    assert summary["suite_counts"]["mlir"]["groups"] == 1
    assert "ttlang" not in json.dumps(summary["suite_counts"])


def test_balance_is_deterministic_complete_and_disjoint():
    inventory = manifest()
    timing = results(inventory)
    plan = benchmark.balance(inventory, timing, 3)
    assert plan == benchmark.balance(inventory, timing, 3)
    assert plan["estimated_seconds"] == [13.0, 12.0, 11.0]
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
        group(inventory, "pytest:me2e/test_ops.py")["cases"].reverse()
    else:
        group(inventory, "pytest:python/test_add.py")["cases"][
            0
        ] = "python/test_add.py::test_add[changed]"
    benchmark.validate_manifest(inventory)
    with pytest.raises(ValueError, match="plan belongs to another manifest"):
        benchmark.selected_groups(inventory, plan, 1)


def test_native_abort_is_not_complete_coverage():
    stages = group(manifest(), "pytest:me2e/test_ops.py")
    events = [{"kind": "collection", "nodes": stages["cases"]}]
    events += [{"kind": "finish", "nodeid": node} for node in stages["cases"][:-1]]
    assert not benchmark.pytest_coverage(stages, events)
    events.append({"kind": "finish", "nodeid": stages["cases"][-1]})
    assert benchmark.pytest_coverage(stages, events)
    events.append(events[-1])
    assert not benchmark.pytest_coverage(stages, events)


def test_public_summary_excludes_commands_paths_and_errors():
    timing = results(manifest())
    timing["groups"][0].update(
        command=["secret"], report_error="private log", log="/private/path"
    )
    summary = benchmark.public_summary(timing)
    assert summary["runtime_execution_observed"]
    assert summary["wall_seconds"] == 36.0
    assert "private" not in json.dumps(summary)
    assert "secret" not in json.dumps(summary)
    assert "python/" not in json.dumps(summary)
    assert "ttlang" not in json.dumps(summary)


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
        manifest(pytest={"nodes": [*DEVICE_NODES, path + "::test_a"]})


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
            "markers": {},
        }
    ]


def test_observer_records_classification_markers(tmp_path):
    (tmp_path / "test_sample.py").write_text(
        "import pytest\n"
        "pytestmark = pytest.mark.multi_device\n"
        "@pytest.mark.compile_only\n"
        "@pytest.mark.slow\n"
        "def test_first(): pass\n"
        "def test_second(): pass\n"
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
    assert benchmark.read_events(events)[0]["markers"] == {
        "test_sample.py::test_first": ["compile_only", "multi_device"],
        "test_sample.py::test_second": ["multi_device"],
    }
    inventory = benchmark.collection_inventory(events)
    assert inventory["nodes"] == [
        "test_sample.py::test_first",
        "test_sample.py::test_second",
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


@pytest.mark.parametrize(
    "entry",
    [
        "  OtherSuite :: python/matmul.py",
        "  TTLang :: ttlang/Dialect/ops.mlir",
        "  TTLang :: python/matmul.mlir",
    ],
)
def test_unrecognized_lit_entry_is_not_silently_omitted(entry):
    with pytest.raises(ValueError, match="unexpected lit"):
        manifest(lit=LIT_TESTS + entry + "\n")


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

    commands = []

    def fake_run(command, output, limit, env, source):
        commands.append(command)
        path = Path(command[-1]).relative_to(tmp_path / "test").as_posix()
        selected = next(group for group in inventory["groups"] if group["path"] == path)
        if "pytest" in command:
            events = [{"kind": "collection", "nodes": selected["cases"]}]
            events += [
                {
                    "kind": "report",
                    "nodeid": node,
                    "phase": "call",
                    "outcome": "passed",
                    "expected_failure": False,
                }
                for node in selected["cases"]
            ]
            events += [{"kind": "finish", "nodeid": node} for node in selected["cases"]]
            if observed and selected["kind"] == "pytest":
                events += [{"kind": "launch"}, {"kind": "returned"}]
            Path(env["TTLANG_EMULE_CI_EVENTS"]).write_text(
                "\n".join(json.dumps(event) for event in events)
            )
        else:
            benchmark.write_json(
                Path(command[command.index("-o") + 1]),
                {
                    "tests": [
                        {"name": case, "code": "PASS"} for case in selected["cases"]
                    ]
                },
            )
        return {
            "exit_code": int(status == "failed"),
            "timed_out": False,
            "wall_seconds": 1.0,
        }

    monkeypatch.setattr(benchmark, "run_command", fake_run)
    assert benchmark.run(args, {"TTLANG_EMULE_CI_THREADS": "3"}) == expected
    summary = benchmark.read_json(tmp_path / "summary.json")
    assert summary["runtime_execution_observed"] is observed
    assert summary["planned_group_count"] == summary["complete_group_count"] == 8
    assert summary["planned_case_count"] == 13
    assert summary["case_outcome_counts"] == {"passed": 13}
    assert summary["suite_counts"]["mlir"]["planned_cases"] == 2
    assert summary["suite_counts"]["me2e"]["outcome_counts"] == {"passed": 3}
    results = benchmark.read_json(tmp_path / "results.json")
    assert [entry["suite"] for entry in results["groups"]] == [
        "mlir",
        "bindings",
        "packaging",
        "me2e",
        "pytest",
        "pytest",
        "pytest",
        "lit",
    ]
    assert results["groups"][0]["lit_codes"] == {"PASS": 2}
    by_kind = {
        selected["kind"]: command
        for selected, command in zip(inventory["groups"], commands)
    }
    # The only "-m" is "python -m pytest": no marker selection excludes cases.
    assert by_kind["pytest"].count("-m") == 1 and by_kind["pytest"][1] == "-m"
    assert "-j3" in by_kind["mlir"] and "-j3" in by_kind["bindings"]
    assert "-j1" in by_kind["lit"]
    assert by_kind["packaging"][by_kind["packaging"].index("-c") + 1] == os.devnull
    assert "observer" in by_kind["packaging"] and "observer" in by_kind["pytest"]
    assert "--reruns=0" in by_kind["packaging"]


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
    group(inventory, "pytest:python/test_add.py")["cases"][
        0
    ] = f"python/test_add.py::test_add[{sentinel}]"
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
    tree = {}
    for index, entry in enumerate(inventory["groups"]):
        if entry["kind"] in benchmark.DIRECTORY_GROUPS:
            # Directory suites are proven by their tracked member files.
            member = entry["cases"][0].removeprefix("TTLang :: ").split("::", 1)[0]
            tree["test/" + member] = ("100644", "blob")
        else:
            tree["test/" + entry["path"]] = (
                "100755" if index == 3 else "100644",
                "blob",
            )
    path = "test/" + group(inventory, "pytest:me2e/test_ops.py")["path"]
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
