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


MLIR_UNSUPPORTED = ["TTLang :: ttlang/Dialect/TTL/IR/ops_invalid.mlir"]


def lit_collection(tests, expected_unsupported=()):
    return {"tests": tests, "expected_unsupported": list(expected_unsupported)}


def collected(**overrides):
    inventory = {
        "pytest": {"nodes": list(DEVICE_NODES), "markers": dict(DEVICE_MARKERS)},
        "packaging": {"nodes": list(PACKAGING_NODES)},
        "lit": lit_collection(LIT_TESTS),
        "mlir": lit_collection(MLIR_TESTS, MLIR_UNSUPPORTED),
        "bindings": lit_collection(BINDINGS_TESTS),
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
    assert group(inventory, "mlir:ttlang")["markers"] == {
        "expected_unsupported": MLIR_UNSUPPORTED
    }
    assert group(inventory, "lit:python/add.py")["markers"] == {
        "expected_unsupported": []
    }
    assert group(inventory, "packaging:packaging")["markers"] == {
        "multi_device": [],
    }
    assert group(inventory, "pytest:python/fabric/test_mesh.py")["markers"] == {
        "multi_device": ["python/fabric/test_mesh.py::test_mesh"],
    }
    assert group(inventory, "pytest:python/test_add.py")["markers"] == {
        "multi_device": [],
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
    "suite,message",
    [
        ("pytest", "every compiler suite"),
        ("me2e", "every compiler suite"),
        ("lit", "every compiler suite"),
        ("packaging", "invalid test group"),
        ("mlir", "invalid test group"),
        ("bindings", "invalid test group"),
    ],
)
def test_every_compiler_suite_must_collect_tests(suite, message):
    if suite == "me2e":
        override = {
            "pytest": {"nodes": [n for n in DEVICE_NODES if not n.startswith("me2e/")]}
        }
    elif suite in ("pytest", "packaging"):
        override = {suite: {"nodes": []}}
    else:
        override = {suite: lit_collection("-- Available Tests --\n")}
    with pytest.raises(ValueError, match=message):
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
        manifest(mlir=lit_collection("-- Available Tests --\n" + entry + "\n"))


@pytest.mark.parametrize("suite", ["lit", "mlir"])
def test_expected_unsupported_lit_tests_must_be_collected(suite):
    stray = ["TTLang :: python/missing.py"]
    tests = {"lit": LIT_TESTS, "mlir": MLIR_TESTS}[suite]
    with pytest.raises(ValueError, match="expected unsupported"):
        manifest(**{suite: lit_collection(tests, stray)})
    with pytest.raises(ValueError, match="expected unsupported"):
        manifest(**{suite: {"tests": tests, "expected_unsupported": "all"}})


def test_classified_cases_must_belong_to_their_group():
    inventory = manifest()
    group(inventory, "pytest:python/test_add.py")["markers"]["multi_device"] = [
        "python/test_matmul.py::test_matmul"
    ]
    with pytest.raises(ValueError, match="classified cases"):
        benchmark.validate_manifest(inventory)
    inventory = manifest()
    del group(inventory, "pytest:python/test_add.py")["markers"]["multi_device"]
    with pytest.raises(ValueError, match="classification markers"):
        benchmark.validate_manifest(inventory)
    inventory = manifest()
    group(inventory, "mlir:ttlang")["markers"] = {"multi_device": []}
    with pytest.raises(ValueError, match="classification markers"):
        benchmark.validate_manifest(inventory)
    inventory = manifest()
    del group(inventory, "lit:python/add.py")["markers"]
    with pytest.raises(ValueError, match="unexpected test group fields"):
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
    supported, expected = mlir["cases"]
    assert mlir["markers"]["expected_unsupported"] == [expected]
    # UNSUPPORTED is a verdict only where the --no-execute probe predicted it;
    # anywhere else it means the runtime features went missing.
    tests = [
        {"name": expected, "code": "UNSUPPORTED"},
        {"name": supported, "code": "UNSUPPORTED"},
    ]
    assert benchmark.lit_outcomes(mlir, tests) == {"unsupported": 1, "incomplete": 1}
    assert not benchmark.lit_coverage(mlir, tests)
    tests = [
        {"name": expected, "code": "UNSUPPORTED"},
        {"name": supported, "code": "UNRESOLVED"},
    ]
    assert benchmark.lit_outcomes(mlir, tests) == {"unsupported": 1, "incomplete": 1}
    assert not benchmark.lit_coverage(mlir, tests)
    tests = [
        {"name": expected, "code": "UNSUPPORTED"},
        {"name": supported, "code": "FLAKYPASS"},
    ]
    assert not benchmark.lit_coverage(mlir, tests)
    tests = [{"name": name, "code": "PASS"} for name in reversed(mlir["cases"])]
    assert benchmark.lit_coverage(mlir, tests)
    assert benchmark.lit_outcomes(mlir, tests[:1]) == {"passed": 1, "not_run": 1}
    tests = [{"name": expected, "code": "XPASS"}, {"name": supported, "code": "XFAIL"}]
    assert benchmark.lit_coverage(mlir, tests)
    assert benchmark.lit_outcomes(mlir, tests) == {"failed": 1, "xfail": 1}


def test_lit_report_tolerates_absence_but_not_malformed_reports(tmp_path):
    assert benchmark.lit_report(tmp_path / "missing.json") == []
    benchmark.write_json(tmp_path / "report.json", {"tests": [{"name": "x"}]})
    with pytest.raises(ValueError, match="invalid lit report"):
        benchmark.lit_report(tmp_path / "report.json")
    (tmp_path / "truncated.json").write_text('{"tests": [')
    with pytest.raises(ValueError):
        benchmark.lit_report(tmp_path / "truncated.json")


def test_truncated_final_event_line_does_not_discard_recorded_outcomes(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text('{"kind": "collection", "nodes": ["a::b"]}\n{"kind": "fin')
    assert benchmark.read_events(events) == [{"kind": "collection", "nodes": ["a::b"]}]
    events.write_text('{"kind": "col\n{"kind": "finish", "nodeid": "a::b"}\n')
    with pytest.raises(ValueError):
        benchmark.read_events(events)


def test_lit_features_require_the_feature_line():
    listing = (
        "-- Test Suites --\n  TTLang - 155 tests\n    Source Root: /private/src\n"
        "    Available Features: optimized system-linux tt-device ttnn\n"
        "    Available Substitutions: %python => /private/python\n"
    )
    assert benchmark.lit_features(listing) >= {"ttnn", "tt-device", "optimized"}
    with pytest.raises(ValueError, match="available features"):
        benchmark.lit_features("-- Test Suites --\n  TTLang - 155 tests\n")


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
        "test_sample.py::test_first": ["multi_device"],
        "test_sample.py::test_second": ["multi_device"],
    }
    inventory = benchmark.collection_inventory(events)
    assert inventory["nodes"] == [
        "test_sample.py::test_first",
        "test_sample.py::test_second",
    ]


@pytest.mark.parametrize("observe", ["0", "1", None])
def test_observer_imports_runtime_only_for_device_suites(tmp_path, observe):
    (tmp_path / "test_sample.py").write_text("def test_first(): pass\n")
    (tmp_path / "ttnn.py").write_text(
        "raise RuntimeError('host-only suites must not import ttnn')\n"
    )
    events = tmp_path / "events.jsonl"
    env = {
        **os.environ,
        "PYTHONPATH": str(SCRIPT_DIR) + os.pathsep + str(tmp_path),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_ADDOPTS": "",
        "TTLANG_EMULE_CI_EVENTS": str(events),
    }
    env.pop("TTLANG_EMULE_CI_OBSERVE_RUNTIME", None)
    if observe is not None:
        env["TTLANG_EMULE_CI_OBSERVE_RUNTIME"] = observe
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "observer",
            "-c",
            os.devnull,
            "--rootdir",
            str(tmp_path),
            str(tmp_path / "test_sample.py"),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if observe == "1":
        assert result.returncode != 0
        assert "must not import ttnn" in result.stdout + result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        kinds = [event["kind"] for event in benchmark.read_events(events)]
        assert kinds == ["collection", "report", "report", "report", "finish"]


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
        manifest(lit=lit_collection(LIT_TESTS + entry + "\n"))


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
        output=tmp_path,
        timeout=600,
        group_timeout=7200,
        lit="llvm-lit",
    )

    commands = []
    observe_flags = []

    def fake_run(command, output, limit, env, source):
        commands.append(command)
        observe_flags.append(env.get("TTLANG_EMULE_CI_OBSERVE_RUNTIME"))
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
    # Only the device suites observe ttnn.generic_op; packaging stays host-only.
    assert [
        flag
        for selected, flag in zip(inventory["groups"], observe_flags)
        if selected["kind"] in ("pytest", "packaging")
    ] == ["0", "1", "1", "1", "1"]


@pytest.mark.parametrize("failure", ["timeout", "malformed", "events_missing"])
def test_lost_group_reports_still_account_for_every_planned_case(
    monkeypatch, tmp_path, failure
):
    inventory = manifest()
    benchmark.write_json(tmp_path / "manifest.json", inventory)
    args = SimpleNamespace(
        source=tmp_path,
        build=tmp_path,
        revision=inventory["revision"],
        runtime_id=inventory["runtime_id"],
        manifest=tmp_path / "manifest.json",
        output=tmp_path,
        timeout=600,
        group_timeout=7200,
        lit="llvm-lit",
    )

    def fake_run(command, output, limit, env, source):
        path = Path(command[-1]).relative_to(tmp_path / "test").as_posix()
        selected = next(group for group in inventory["groups"] if group["path"] == path)
        timed_out = False
        if selected["kind"] == "mlir":
            # The group timeout killed lit (no report), lit wrote garbage, or
            # pytest never produced its event file.
            if failure == "timeout":
                timed_out = True
            elif failure == "malformed":
                Path(command[command.index("-o") + 1]).write_text("{not json")
        elif "pytest" in command and failure != "events_missing":
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
            events += [{"kind": "launch"}, {"kind": "returned"}]
            Path(env["TTLANG_EMULE_CI_EVENTS"]).write_text(
                "\n".join(json.dumps(event) for event in events)
            )
        elif "pytest" not in command:
            benchmark.write_json(
                Path(command[command.index("-o") + 1]),
                {"tests": [{"name": c, "code": "PASS"} for c in selected["cases"]]},
            )
        return {"exit_code": 0, "timed_out": timed_out, "wall_seconds": 1.0}

    monkeypatch.setattr(benchmark, "run_command", fake_run)
    assert benchmark.run(args, {}) == (2 if failure == "events_missing" else 1)
    summary = benchmark.read_json(tmp_path / "summary.json")
    results = benchmark.read_json(tmp_path / "results.json")
    mlir = results["groups"][0]
    assert summary["planned_case_count"] == sum(summary["case_outcome_counts"].values())
    if failure == "events_missing":
        for entry in results["groups"]:
            if entry["suite"] in ("pytest", "me2e", "packaging"):
                assert entry["status"] == "infrastructure_error"
                planned = len(group(inventory, entry["id"])["cases"])
                assert entry["outcomes"] == {"not_run": planned}
        assert summary["runtime_execution_observed"] is False
    else:
        assert mlir["status"] == "infrastructure_error" and mlir["complete"] is False
        assert mlir["outcomes"] == {"not_run": 2}
        assert summary["suite_counts"]["mlir"]["outcome_counts"] == {"not_run": 2}
        assert summary["suite_counts"]["mlir"]["status_counts"] == {
            "infrastructure_error": 1
        }
        assert ("report_error" in mlir) == (failure == "malformed")
    assert "private" not in json.dumps(summary)


def test_collect_builds_the_complete_inventory_and_gates_lit_features(
    monkeypatch, tmp_path
):
    args = SimpleNamespace(
        source=tmp_path / "src",
        build=tmp_path / "build",
        revision="a" * 40,
        runtime_id="sha256:" + "b" * 64,
        output=tmp_path / "out",
        lit="llvm-lit",
    )
    args.output.mkdir()
    listings = {
        "python": (LIT_TESTS, ["TTLang :: python/add.py"]),
        "ttlang": (MLIR_TESTS, []),
        "bindings/python": (BINDINGS_TESTS, []),
    }
    features = {"value": "optimized system-linux tt-device ttnn"}
    commands = []

    def fake_run(command, output, limit, env, source):
        commands.append(command)
        if command[0] == "llvm-lit":
            suite = Path(command[-1]).relative_to(args.build / "test").as_posix()
            tests, unsupported = listings[suite]
            if "--show-tests" in command:
                output.write_text(tests)
            elif "--show-suites" in command:
                output.write_text(
                    "-- Test Suites --\n  TTLang - 1 tests\n"
                    f"    Available Features: {features['value']}\n"
                )
            else:
                assert "--no-execute" in command and "-j1" in command
                names = benchmark.lit_names(tests, suite.rstrip("/") + "/")
                benchmark.write_json(
                    Path(command[command.index("-o") + 1]),
                    {
                        "tests": [
                            {
                                "name": "TTLang :: " + name,
                                "code": (
                                    "UNSUPPORTED"
                                    if "TTLang :: " + name in unsupported
                                    else "PASS"
                                ),
                            }
                            for name in names
                        ]
                    },
                )
                return {"exit_code": 1, "timed_out": False, "wall_seconds": 1.0}
        else:
            assert "--collect-only" in command
            nodes = (
                PACKAGING_NODES if command[-1].endswith("packaging") else DEVICE_NODES
            )
            markers = {} if command[-1].endswith("packaging") else DEVICE_MARKERS
            Path(env["TTLANG_EMULE_CI_EVENTS"]).write_text(
                json.dumps({"kind": "collection", "nodes": nodes, "markers": markers})
            )
        return {"exit_code": 0, "timed_out": False, "wall_seconds": 1.0}

    monkeypatch.setattr(benchmark, "run_command", fake_run)
    benchmark.collect(args, {"TTLANG_EMULE_CI_THREADS": "1"})
    produced = benchmark.read_json(args.output / "manifest.json")
    expected = manifest(lit=lit_collection(LIT_TESTS, ["TTLang :: python/add.py"]))
    expected["groups"] = [
        (
            {**entry, "markers": {"expected_unsupported": []}}
            if entry["kind"] == "mlir"
            else entry
        )
        for entry in expected["groups"]
    ]
    assert produced == expected
    assert group(produced, "lit:python/add.py")["markers"] == {
        "expected_unsupported": ["TTLang :: python/add.py"]
    }
    assert sum(command[0] == "llvm-lit" for command in commands) == 7
    # Without the runtime features every Python lit test would silently become
    # UNSUPPORTED, so collection refuses to produce an inventory at all.
    features["value"] = "optimized system-linux"
    args.output = tmp_path / "out2"
    args.output.mkdir()
    with pytest.raises(ValueError, match="emulator runtime features"):
        benchmark.collect(args, {})
    assert not (args.output / "manifest.json").exists()


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
def test_manifest_rejects_unexpected_fields(where):
    inventory = manifest()
    target = inventory if where == "root" else inventory["groups"][0]
    target["private_diagnostic"] = "must not be accepted"
    with pytest.raises(ValueError, match="unexpected.*fields"):
        benchmark.validate_manifest(inventory)


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
