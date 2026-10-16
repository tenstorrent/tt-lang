# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts/emule-ci"
SPEC = importlib.util.spec_from_file_location(
    "emule_inventory", SCRIPT_DIR / "inventory.py"
)
emule_inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(emule_inventory)


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
APPENDED_NODES = [
    "bindings/python/test_ttl_ttcore_integration.py::test_context",
    "examples/test_example_scripts.py::test_example[eltwise_add.py]",
    "examples/test_example_scripts.py::test_example[multidevice_all_reduce.py]",
    "tutorial/test_tutorials.py::test_tutorial[matmul-tutorial/step_1.py]",
]
APPENDED_MARKERS = {
    "examples/test_example_scripts.py::test_example[multidevice_all_reduce.py]": [
        "multi_device"
    ]
}
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
        "appended": {"nodes": list(APPENDED_NODES), "markers": dict(APPENDED_MARKERS)},
        "lit": lit_collection(LIT_TESTS),
        "mlir": lit_collection(MLIR_TESTS, MLIR_UNSUPPORTED),
        "bindings": lit_collection(BINDINGS_TESTS),
    }
    inventory.update(overrides)
    return inventory


def manifest(**overrides):
    return emule_inventory.make_manifest(
        collected(**overrides), "a" * 40, "sha256:" + "b" * 64
    )


def group(inventory, group_id):
    return next(group for group in inventory["groups"] if group["id"] == group_id)


def write_xfail(source, tests):
    path = source / emule_inventory.XFAIL_LIST
    path.parent.mkdir(parents=True, exist_ok=True)
    emule_inventory.write_json(path, {"schema_version": 1, "tests": tests})
    return path


def results(inventory):
    return {
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
    assert [group["id"] for group in inventory["groups"]] == [
        "mlir:ttlang",
        "bindings:bindings/python",
        "packaging:packaging",
        "pytest:me2e/test_ops.py",
        "pytest:python/test_add.py",
        "pytest:python/test_matmul.py",
        "pytest:python/fabric/test_mesh.py",
        "lit:python/add.py",
        "pytest:bindings/python/test_ttl_ttcore_integration.py",
        "pytest:examples/test_example_scripts.py",
        "pytest:tutorial/test_tutorials.py",
    ]
    assert group(inventory, "mlir:ttlang")["cases"] == [
        "TTLang :: ttlang/Dialect/TTL/IR/ops.mlir",
        "TTLang :: ttlang/Dialect/TTL/IR/ops_invalid.mlir",
    ]
    assert group(inventory, "packaging:packaging")["cases"] == PACKAGING_NODES
    assert [emule_inventory.suite_of(group["id"]) for group in inventory["groups"]] == [
        "mlir",
        "bindings",
        "packaging",
        "me2e",
        "pytest",
        "pytest",
        "pytest",
        "lit",
        "bindings",
        "examples",
        "tutorial",
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
    assert group(inventory, "pytest:examples/test_example_scripts.py")["markers"] == {
        "multi_device": list(APPENDED_MARKERS),
    }


@pytest.mark.parametrize("path", ["sim/test_a.py", "pythonic/test_a.py"])
def test_pytest_groups_outside_known_suites_are_rejected(path):
    with pytest.raises(ValueError, match="known compiler suites"):
        manifest(appended={"nodes": [*APPENDED_NODES, path + "::test_a"]})


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
        ("examples", "every compiler suite"),
        ("tutorial", "every compiler suite"),
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
    elif suite in ("examples", "tutorial"):
        nodes = [n for n in APPENDED_NODES if not n.startswith(suite + "/")]
        override = {"appended": {"nodes": nodes}}
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
        emule_inventory.validate_manifest(inventory)


@pytest.mark.parametrize("stale", [-1, "1", True, 1.0])
def test_manifest_rejects_an_invalid_stale_xfail_count(stale):
    with pytest.raises(ValueError, match="stale"):
        emule_inventory.validate_manifest({**manifest(), "stale_xfail_count": stale})


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
    assert emule_inventory.pytest_outcomes(mesh, skipped) == {"unsupported": 1}
    assert emule_inventory.pytest_outcomes(add, skipped) == {"skipped": 1, "not_run": 1}
    xfail = [dict(skipped[0], expected_failure=True)]
    assert emule_inventory.pytest_outcomes(mesh, xfail) == {"xfail": 1}
    # A case that began without any report aborted mid-test.
    started = skipped + [{"kind": "start", "nodeid": add["cases"][1]}]
    assert emule_inventory.pytest_outcomes(add, started) == {
        "skipped": 1,
        "incomplete": 1,
    }
    mlir = group(inventory, "mlir:ttlang")
    supported, expected = mlir["cases"]
    assert mlir["markers"]["expected_unsupported"] == [expected]
    # UNSUPPORTED is a verdict only where the --no-execute probe predicted it;
    # anywhere else it means the runtime features went missing.
    tests = [
        {"name": expected, "code": "UNSUPPORTED"},
        {"name": supported, "code": "UNSUPPORTED"},
    ]
    assert emule_inventory.lit_outcomes(mlir, tests, {}) == {
        "unsupported": 1,
        "incomplete": 1,
    }
    assert not emule_inventory.lit_coverage(mlir, tests, {})
    tests = [
        {"name": expected, "code": "UNSUPPORTED"},
        {"name": supported, "code": "UNRESOLVED"},
    ]
    assert emule_inventory.lit_outcomes(mlir, tests, {}) == {
        "unsupported": 1,
        "incomplete": 1,
    }
    assert not emule_inventory.lit_coverage(mlir, tests, {})
    tests = [
        {"name": expected, "code": "UNSUPPORTED"},
        {"name": supported, "code": "FLAKYPASS"},
    ]
    assert not emule_inventory.lit_coverage(mlir, tests, {})
    tests = [{"name": name, "code": "PASS"} for name in reversed(mlir["cases"])]
    assert emule_inventory.lit_coverage(mlir, tests, {})
    assert emule_inventory.lit_outcomes(mlir, tests[:1], {}) == {
        "passed": 1,
        "not_run": 1,
    }
    tests = [{"name": expected, "code": "XPASS"}, {"name": supported, "code": "XFAIL"}]
    assert emule_inventory.lit_coverage(mlir, tests, {})
    assert emule_inventory.lit_outcomes(mlir, tests, {}) == {"failed": 1, "xfail": 1}


def test_listed_lit_cases_are_non_strict_and_not_run_cases_are_xfail():
    mlir = group(manifest(), "mlir:ttlang")
    ops, invalid = mlir["cases"]
    # A listed pass (XPASS) is a non-strict pass; an unlisted XPASS still fails.
    tests = [{"name": ops, "code": "XPASS"}, {"name": invalid, "code": "XFAIL"}]
    listed = {ops: True, invalid: True}
    assert emule_inventory.lit_coverage(mlir, tests, listed)
    assert emule_inventory.lit_outcomes(mlir, tests, listed) == {
        "passed": 1,
        "xfail": 1,
    }
    assert emule_inventory.lit_outcomes(mlir, tests, {invalid: True}) == {
        "failed": 1,
        "xfail": 1,
    }
    # A case that must not run is an xfail without a report, and fails coverage
    # if lit reported it anyway.
    listed = {invalid: False}
    tests = [{"name": ops, "code": "PASS"}]
    assert emule_inventory.lit_coverage(mlir, tests, listed)
    assert emule_inventory.lit_outcomes(mlir, tests, listed) == {
        "passed": 1,
        "xfail": 1,
    }
    tests.append({"name": invalid, "code": "FAIL"})
    assert not emule_inventory.lit_coverage(mlir, tests, listed)
    args = SimpleNamespace(lit="llvm-lit", timeout=300, build=Path("/build"))
    command = emule_inventory.group_command(
        args, mlir, Path("/out/group"), {}, {ops: True, invalid: False}
    )
    assert "--xfail=" + ops in command and "--allow-empty-runs" in command
    excluded = next(arg for arg in command if arg.startswith("--filter-out="))
    pattern = re.compile(excluded.partition("=")[2], re.IGNORECASE)
    assert pattern.search(invalid) and not pattern.search(ops)
    assert not pattern.search(invalid + ".orig")
    assert command[-3:] == ["-o", "/out/group.lit.json", "/build/test/ttlang"]
    command = emule_inventory.group_command(args, mlir, Path("/out/group"), {}, {})
    assert not any(arg.startswith(("--xfail", "--filter-out")) for arg in command)


@pytest.mark.parametrize(
    "contents",
    [
        "[]",
        "{not json",
        '{"schema_version": 2, "tests": {}}',
        '{"schema_version": 1, "tests": []}',
        '{"schema_version": 1, "tests": {}, "extra": 1}',
        '{"schema_version": 1, "tests": {"a": {"run": true}}}',
        '{"schema_version": 1, "tests": {"a": {"reason": " ", "run": true}}}',
        '{"schema_version": 1, "tests": {"a": {"reason": "r", "run": "no"}}}',
        '{"schema_version": 1, "tests": {"a": {"reason": "r", "run": true, "x": 1}}}',
        '{"schema_version": 1, "tests": {"a": {"reason": "r", "run": true},'
        ' "a": {"reason": "r", "run": false}}}',
    ],
)
def test_malformed_xfail_list_fails_closed(tmp_path, contents):
    (tmp_path / "xfail.json").write_text(contents)
    with pytest.raises(ValueError):
        emule_inventory.xfail_list(tmp_path / "xfail.json")


@pytest.mark.parametrize(
    "reason",
    [
        "see /private/source/test.py",
        "tracked in issue #12",
        "ask @owner",
        "https://example.com/issue",
        "line\nbreak",
        "x" * 161,
    ],
)
def test_xfail_reasons_are_generic_text_only(tmp_path, reason):
    """The public list must not carry paths, references, URLs or names."""
    write_xfail(tmp_path, {"a::b": {"reason": reason, "run": True}})
    with pytest.raises(ValueError, match="malformed"):
        emule_inventory.xfail_list(tmp_path / emule_inventory.XFAIL_LIST)
    generic = "Emulator gap: tile reuse (dst_reg), see note; it's slow - 2x."
    write_xfail(tmp_path, {"a::b": {"reason": generic, "run": False}})
    assert emule_inventory.xfail_list(tmp_path / emule_inventory.XFAIL_LIST) == {
        "a::b": {"reason": generic, "run": False}
    }


def test_checked_in_xfail_list_is_valid():
    assert isinstance(
        emule_inventory.xfail_list(REPO_ROOT / emule_inventory.XFAIL_LIST), dict
    )


def test_lit_report_tolerates_absence_but_not_malformed_reports(tmp_path):
    assert emule_inventory.lit_report(tmp_path / "missing.json") == []
    emule_inventory.write_json(tmp_path / "report.json", {"tests": [{"name": "x"}]})
    with pytest.raises(ValueError, match="invalid lit report"):
        emule_inventory.lit_report(tmp_path / "report.json")
    (tmp_path / "truncated.json").write_text('{"tests": [')
    with pytest.raises(ValueError):
        emule_inventory.lit_report(tmp_path / "truncated.json")


def test_truncated_final_event_line_does_not_discard_recorded_outcomes(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text('{"kind": "collection", "nodes": ["a::b"]}\n{"kind": "fin')
    assert emule_inventory.read_events(events) == [
        {"kind": "collection", "nodes": ["a::b"]}
    ]
    events.write_text('{"kind": "col\n{"kind": "finish", "nodeid": "a::b"}\n')
    with pytest.raises(ValueError):
        emule_inventory.read_events(events)


def test_lit_features_require_the_feature_line():
    listing = (
        "-- Test Suites --\n  TTLang - 155 tests\n    Source Root: /private/src\n"
        "    Available Features: optimized system-linux tt-device ttnn\n"
        "    Available Substitutions: %python => /private/python\n"
    )
    assert emule_inventory.lit_features(listing) >= {"ttnn", "tt-device", "optimized"}
    with pytest.raises(ValueError, match="available features"):
        emule_inventory.lit_features("-- Test Suites --\n  TTLang - 155 tests\n")


def test_public_summary_reports_suites_outcomes_and_unsupported_cases():
    inventory = manifest()
    timing = results(inventory)
    for entry in timing["groups"]:
        entry["outcomes"] = {"passed": 1}
    mesh = [entry["id"] for entry in timing["groups"]].index(
        "pytest:python/fabric/test_mesh.py"
    )
    timing["groups"][mesh]["outcomes"] = {"unsupported": 1}
    timing["planned_suite_counts"] = {
        suite: {"groups": 1, "cases": 2} for suite in emule_inventory.SUITES
    }
    summary = emule_inventory.public_summary(timing)
    assert summary["case_outcome_counts"] == {"passed": 10, "unsupported": 1}
    assert set(summary["suite_counts"]) == set(emule_inventory.SUITES)
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
    assert not emule_inventory.pytest_coverage(stages, events)
    events.append({"kind": "finish", "nodeid": stages["cases"][-1]})
    assert emule_inventory.pytest_coverage(stages, events)
    events.append(events[-1])
    assert not emule_inventory.pytest_coverage(stages, events)


def test_public_summary_excludes_commands_paths_and_errors():
    timing = results(manifest())
    timing["groups"][0].update(
        command=["secret"], report_error="private log", log="/private/path"
    )
    summary = emule_inventory.public_summary(timing)
    assert summary["runtime_execution_observed"]
    assert summary["wall_seconds"] == 66.0
    assert "private" not in json.dumps(summary)
    assert "secret" not in json.dumps(summary)
    assert "python/" not in json.dumps(summary)
    assert "ttlang" not in json.dumps(summary)


@pytest.mark.parametrize(
    "path",
    [
        "/python/test_a.py",
        "python/../test_a.py",
        "python//test_a.py",
    ],
)
def test_rejects_unsafe_test_paths(path):
    with pytest.raises(ValueError, match="invalid test group"):
        manifest(pytest={"nodes": [*DEVICE_NODES, path + "::test_a"]})


def test_command_records_launch_failure_and_timeout(tmp_path):
    failure = emule_inventory.run_command(
        [str(tmp_path / "missing")], tmp_path / "missing.log", 5, os.environ, tmp_path
    )
    assert failure["exit_code"] is None
    assert "error" in failure
    timeout = emule_inventory.run_command(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        tmp_path / "timeout.log",
        0.1,
        os.environ,
        tmp_path,
    )
    assert timeout["timed_out"] and not timeout["stalled"]
    assert timeout["exit_code"] < 0


def test_silent_command_is_stopped_after_the_stall_timeout(tmp_path):
    result = emule_inventory.run_command(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        tmp_path / "silent.log",
        60,
        os.environ,
        tmp_path,
        stall=0.5,
        progress=(tmp_path / "events.jsonl",),
    )
    assert result["stalled"] and not result["timed_out"]
    assert result["exit_code"] < 0
    assert result["wall_seconds"] < 30


@pytest.mark.parametrize("channel", ["events", "log"])
def test_command_that_keeps_reporting_progress_is_not_stopped(tmp_path, channel):
    events = tmp_path / "events.jsonl"
    write = (
        f"open({str(events)!r}, 'a').write('{{}}\\n')"
        if channel == "events"
        else "print('.', flush=True)"
    )
    # Ten writes 0.2 s apart outlast the 0.6 s stall timeout several times over.
    result = emule_inventory.run_command(
        [
            sys.executable,
            "-c",
            f"import time\nfor _ in range(10):\n    {write}\n    time.sleep(0.2)",
        ],
        tmp_path / "progress.log",
        60,
        os.environ,
        tmp_path,
        stall=0.6,
        progress=(events,),
    )
    assert result["exit_code"] == 0
    assert not result["stalled"] and not result["timed_out"]
    assert result["wall_seconds"] >= 1.8


@pytest.mark.parametrize("stall", ["0", "-1"])
def test_run_rejects_a_non_positive_stall_timeout(tmp_path, stall):
    options = ["--source", str(tmp_path), "--build", str(tmp_path)]
    options += ["--revision", "a" * 40, "--runtime-id", "runtime"]
    options += ["--output", str(tmp_path / "out"), "--manifest", str(tmp_path)]
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "inventory.py"), "run", *options]
        + ["--stall-timeout", stall],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "timeouts must be positive" in result.stderr
    assert not (tmp_path / "out").exists()


def run_observer(tmp_path, source, *options, observe=None, xfail=None):
    """Run pytest with the observer plugin where importing ttnn fails loudly."""
    sample = tmp_path / "test_sample.py"
    sample.write_text(source)
    (tmp_path / "ttnn.py").write_text("raise RuntimeError('must not import ttnn')\n")
    events = tmp_path / "events.jsonl"
    env = {
        **os.environ,
        "PYTHONPATH": str(SCRIPT_DIR) + os.pathsep + str(tmp_path),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_ADDOPTS": "",
        "TTLANG_EMULE_CI_EVENTS": str(events),
    }
    env.pop("TTLANG_EMULE_CI_OBSERVE_RUNTIME", None)
    env.pop("TTLANG_EMULE_CI_XFAIL", None)
    if observe is not None:
        env["TTLANG_EMULE_CI_OBSERVE_RUNTIME"] = observe
    if xfail is not None:
        env["TTLANG_EMULE_CI_XFAIL"] = str(write_xfail(tmp_path, xfail))
    plugin = ["-p", "observer", "-c", os.devnull, "--rootdir", str(tmp_path)]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *options, *plugin, str(sample)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, events


def test_observer_records_classification_markers(tmp_path):
    # Collection never imports ttnn, even when runtime observation is requested.
    result, events = run_observer(
        tmp_path,
        "import pytest\n"
        "pytestmark = pytest.mark.multi_device\n"
        "@pytest.mark.compile_only\n"
        "@pytest.mark.slow\n"
        "def test_first(): pass\n"
        "def test_second(): pass\n",
        "--collect-only",
        observe="1",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert emule_inventory.read_events(events)[0]["markers"] == {
        "test_sample.py::test_first": ["multi_device"],
        "test_sample.py::test_second": ["multi_device"],
    }
    inventory = emule_inventory.collection_inventory(events)
    assert inventory["nodes"] == [
        "test_sample.py::test_first",
        "test_sample.py::test_second",
    ]


@pytest.mark.parametrize("observe", ["0", "1", None])
def test_observer_imports_runtime_only_for_device_suites(tmp_path, observe):
    result, events = run_observer(tmp_path, "def test_first(): pass\n", observe=observe)
    if observe == "1":
        assert result.returncode != 0
        assert "must not import ttnn" in result.stdout + result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        recorded = emule_inventory.read_events(events)
        assert [event["kind"] for event in recorded] == [
            "collection",
            "start",
            "report",
            "report",
            "report",
            "finish",
        ]
        assert recorded[0]["markers"] == {}


@pytest.mark.parametrize("reruns", [False, True])
def test_case_that_aborts_the_process_is_attributed_as_incomplete(tmp_path, reruns):
    options = ()
    if reruns:
        pytest.importorskip("pytest_rerunfailures")
        options = ("-p", "pytest_rerunfailures", "--reruns=0")
    result, events = run_observer(
        tmp_path,
        "import os\n"
        "def test_first(): pass\n"
        "def test_aborts(): os._exit(134)\n"
        "def test_never(): pass\n",
        *options,
    )
    assert result.returncode == 134, result.stdout + result.stderr
    recorded = emule_inventory.read_events(events)
    aborted = "test_sample.py::test_aborts"
    # pytest-rerunfailures holds back a case's reports until it finishes; drop
    # them as it would, so only the start event names the aborted case.
    held = [event for event in recorded if event.get("nodeid") != aborted]
    held.append({"kind": "start", "nodeid": aborted})
    sample = {"cases": recorded[0]["nodes"]}
    for observed in (recorded, held):
        outcomes = emule_inventory.pytest_case_outcomes(sample, observed)
        assert outcomes == {
            "test_sample.py::test_first": "passed",
            aborted: "incomplete",
            "test_sample.py::test_never": "not_run",
        }
        assert not emule_inventory.pytest_coverage(sample, observed)


def test_observer_marks_listed_cases_non_strict_and_skips_cases_not_run(tmp_path):
    ran = tmp_path / "ran"
    entry = {"reason": "emulator gap", "run": True}
    result, events = run_observer(
        tmp_path,
        "import pytest\n"
        f"RAN = {str(ran)!r}\n"
        "def test_fails(): assert False\n"
        "def test_passes(): pass\n"
        "@pytest.mark.xfail(strict=True)\n"
        "def test_own_strict(): pass\n"
        "def test_hangs(): open(RAN, 'w').close()\n"
        "def test_unlisted(): assert False\n",
        "-o",
        "xfail_strict=true",
        xfail={
            "test_sample.py::test_fails": entry,
            "test_sample.py::test_passes": entry,
            "test_sample.py::test_own_strict": entry,
            "test_sample.py::test_hangs": {"reason": "hangs", "run": False},
            "TTLang :: python/add.py": entry,
        },
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert not ran.exists()
    recorded = emule_inventory.read_events(events)
    outcomes = {
        case.partition("::")[2]: emule_inventory.pytest_outcomes(
            {"cases": [case]}, recorded
        )
        for case in recorded[0]["nodes"]
    }
    assert outcomes == {
        "test_fails": {"xfail": 1},
        "test_passes": {"passed": 1},
        "test_own_strict": {"passed": 1},
        "test_hangs": {"xfail": 1},
        "test_unlisted": {"failed": 1},
    }


@pytest.mark.parametrize(
    "head,dirty", [("b" * 40, ""), ("a" * 40, " M test/python/test_add.py\n")]
)
def test_source_revision_and_cleanliness_are_verified(
    monkeypatch, tmp_path, head, dirty
):
    responses = iter([SimpleNamespace(stdout=head), SimpleNamespace(stdout=dirty)])
    monkeypatch.setattr(
        emule_inventory.subprocess, "run", lambda *args, **kwargs: next(responses)
    )
    with pytest.raises(ValueError):
        emule_inventory.validate_source(tmp_path, "a" * 40)


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


def run_args(tmp_path, inventory):
    emule_inventory.write_json(tmp_path / "manifest.json", inventory)
    write_xfail(tmp_path, {})
    return SimpleNamespace(
        source=tmp_path,
        build=tmp_path,
        revision=inventory["revision"],
        runtime_id=inventory["runtime_id"],
        manifest=tmp_path / "manifest.json",
        output=tmp_path,
        timeout=600,
        group_timeout=7200,
        stall_timeout=600,
        lit="llvm-lit",
    )


def selected_group(tmp_path, inventory, command):
    path = Path(command[-1]).relative_to(tmp_path / "test").as_posix()
    return next(group for group in inventory["groups"] if group["path"] == path)


def write_passing_report(selected, command, env, runtime):
    """Write the pytest events or lit report of a group whose cases all pass."""
    cases = selected["cases"]
    if "pytest" not in command:
        emule_inventory.write_json(
            Path(command[command.index("-o") + 1]),
            {"tests": [{"name": case, "code": "PASS"} for case in cases]},
        )
        return
    events = [{"kind": "collection", "nodes": cases}]
    events += [
        {
            "kind": "report",
            "nodeid": node,
            "phase": "call",
            "outcome": "passed",
            "expected_failure": False,
        }
        for node in cases
    ]
    events += [{"kind": "finish", "nodeid": node} for node in cases]
    if runtime and selected["kind"] == "pytest":
        events += [{"kind": "launch"}, {"kind": "returned"}]
    Path(env["TTLANG_EMULE_CI_EVENTS"]).write_text(
        "\n".join(json.dumps(event) for event in events)
    )


@pytest.mark.parametrize(
    "observed,status,expected",
    [(False, "passed", 2), (True, "failed", 1), (True, "passed", 0)],
)
def test_run_requires_coverage_and_runtime_execution(
    monkeypatch, tmp_path, observed, status, expected
):
    inventory = manifest()
    args = run_args(tmp_path, inventory)

    commands = []
    observe_flags = []
    script_timeouts = set()
    watchdogs = []

    def fake_run(command, output, limit, env, source, **watchdog):
        commands.append(command)
        watchdogs.append(watchdog)
        observe_flags.append(env.get("TTLANG_EMULE_CI_OBSERVE_RUNTIME"))
        script_timeouts.add(env.get("TUTORIAL_TIMEOUT_SECONDS"))
        script_timeouts.add(env.get("TTLANG_EXAMPLE_TIMEOUT_SECONDS"))
        selected = selected_group(tmp_path, inventory, command)
        write_passing_report(selected, command, env, observed)
        return {
            "exit_code": int(status == "failed"),
            "timed_out": False,
            "wall_seconds": 1.0,
        }

    monkeypatch.setattr(emule_inventory, "run_command", fake_run)
    assert emule_inventory.run(args, {"TTLANG_EMULE_CI_THREADS": "3"}) == expected
    summary = emule_inventory.read_json(tmp_path / "summary.json")
    assert summary["runtime_execution_observed"] is observed
    assert summary["planned_group_count"] == summary["complete_group_count"] == 11
    assert summary["planned_case_count"] == 17
    assert summary["case_outcome_counts"] == {"passed": 17}
    assert summary["suite_counts"]["mlir"]["planned_cases"] == 2
    assert summary["suite_counts"]["me2e"]["outcome_counts"] == {"passed": 3}
    results = emule_inventory.read_json(tmp_path / "results.json")
    assert [entry["suite"] for entry in results["groups"]] == [
        "mlir",
        "bindings",
        "packaging",
        "me2e",
        "pytest",
        "pytest",
        "pytest",
        "lit",
        "bindings",
        "examples",
        "tutorial",
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
    # Only the device suites observe ttnn.generic_op: packaging and the binding
    # file stay host-only, and the scripts run kernels in child processes.
    assert [
        flag
        for selected, flag in zip(inventory["groups"], observe_flags)
        if selected["kind"] in ("pytest", "packaging")
    ] == ["0", "1", "1", "1", "1", "0", "0", "0"]
    assert script_timeouts - {None} == {"600"}
    # Every group is watched for silence through its log and observer events.
    assert watchdogs == [
        {
            "stall": 600,
            "progress": (tmp_path / f"group-{index:04d}.events.jsonl",),
        }
        for index in range(len(inventory["groups"]))
    ]


@pytest.mark.parametrize("failure", ["timeout", "malformed", "events_missing"])
def test_lost_group_reports_still_account_for_every_planned_case(
    monkeypatch, tmp_path, failure
):
    inventory = manifest()
    args = run_args(tmp_path, inventory)

    def fake_run(command, output, limit, env, source, **watchdog):
        selected = selected_group(tmp_path, inventory, command)
        timed_out = False
        if selected["kind"] == "mlir":
            # The group timeout killed lit (no report), lit wrote garbage, or
            # pytest never produced its event file.
            if failure == "timeout":
                timed_out = True
            elif failure == "malformed":
                Path(command[command.index("-o") + 1]).write_text("{not json")
        elif "pytest" not in command or failure != "events_missing":
            write_passing_report(selected, command, env, runtime=True)
        return {"exit_code": 0, "timed_out": timed_out, "wall_seconds": 1.0}

    monkeypatch.setattr(emule_inventory, "run_command", fake_run)
    assert emule_inventory.run(args, {}) == (2 if failure == "events_missing" else 1)
    summary = emule_inventory.read_json(tmp_path / "summary.json")
    results = emule_inventory.read_json(tmp_path / "results.json")
    mlir = results["groups"][0]
    assert summary["planned_case_count"] == sum(summary["case_outcome_counts"].values())
    if failure == "events_missing":
        for entry in results["groups"]:
            if group(inventory, entry["id"])["kind"] in emule_inventory.PYTEST_KINDS:
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


def test_stalled_group_attributes_its_hung_case_and_stops_only_that_group(
    monkeypatch, tmp_path
):
    inventory = manifest()
    args = run_args(tmp_path, inventory)
    args.stall_timeout = 0.5
    real_run = emule_inventory.run_command
    hung = "pytest:me2e/test_ops.py"
    cases = group(inventory, hung)["cases"]
    # Like an emulator hang holding the GIL: the first case passes, the second
    # starts and then nothing more is written, so pytest-timeout never fires.
    events = [{"kind": "collection", "nodes": cases}]
    events += [
        {
            "kind": "report",
            "nodeid": cases[0],
            "phase": "call",
            "outcome": "passed",
            "expected_failure": False,
        },
        {"kind": "finish", "nodeid": cases[0]},
        {"kind": "start", "nodeid": cases[1]},
    ]
    script = (
        "import json, os, time\n"
        f"with open(os.environ['TTLANG_EMULE_CI_EVENTS'], 'a') as output:\n"
        f"    for event in {events!r}:\n"
        "        output.write(json.dumps(event) + '\\n')\n"
        "time.sleep(60)\n"
    )

    def fake_run(command, output, limit, env, source, **watchdog):
        selected = selected_group(tmp_path, inventory, command)
        if selected["id"] != hung:
            write_passing_report(selected, command, env, runtime=True)
            return {"exit_code": 0, "timed_out": False, "wall_seconds": 1.0}
        return real_run(
            [sys.executable, "-c", script], output, limit, env, source, **watchdog
        )

    monkeypatch.setattr(emule_inventory, "run_command", fake_run)
    assert emule_inventory.run(args, dict(os.environ)) == 1
    results = emule_inventory.read_json(tmp_path / "results.json")
    index = [entry["id"] for entry in results["groups"]].index(hung)
    entry = results["groups"][index]
    assert entry["stalled"] and not entry["timed_out"] and entry["exit_code"] < 0
    assert entry["status"] == "infrastructure_error" and not entry["complete"]
    recorded = emule_inventory.read_events(tmp_path / f"group-{index:04d}.events.jsonl")
    assert emule_inventory.pytest_case_outcomes(group(inventory, hung), recorded) == {
        cases[0]: "passed",
        cases[1]: "incomplete",
        cases[2]: "not_run",
    }
    assert entry["outcomes"] == {"passed": 1, "incomplete": 1, "not_run": 1}
    assert {other["status"] for other in results["groups"] if other["id"] != hung} == {
        "passed"
    }
    summary = emule_inventory.read_json(tmp_path / "summary.json")
    assert summary["planned_case_count"] == sum(summary["case_outcome_counts"].values())
    assert summary["group_status_counts"] == {"passed": 10, "infrastructure_error": 1}


@pytest.mark.parametrize("listed", [True, False, None])
def test_run_applies_the_xfail_list_to_lit_groups(monkeypatch, tmp_path, listed):
    inventory = {**manifest(), "stale_xfail_count": 2}
    args = run_args(tmp_path, inventory)
    case = "TTLang :: python/add.py"
    if listed is not None:
        write_xfail(tmp_path, {case: {"reason": "emulator gap", "run": listed}})
    commands = []

    def fake_run(command, output, limit, env, source, **watchdog):
        selected = selected_group(tmp_path, inventory, command)
        if selected["id"] != "lit:python/add.py":
            write_passing_report(selected, command, env, runtime=True)
            return {"exit_code": 0, "timed_out": False, "wall_seconds": 1.0}
        commands.append(command)
        if listed is False:
            # Filtered out: lit runs nothing, writes no report and succeeds.
            return {"exit_code": 0, "timed_out": False, "wall_seconds": 1.0}
        # The case passes, so lit reports XPASS for a listed (or XFAIL) test.
        report = Path(command[command.index("-o") + 1])
        emule_inventory.write_json(report, {"tests": [{"name": case, "code": "XPASS"}]})
        return {"exit_code": 1, "timed_out": False, "wall_seconds": 1.0}

    monkeypatch.setattr(emule_inventory, "run_command", fake_run)
    assert emule_inventory.run(args, {}) == (1 if listed is None else 0)
    summary = emule_inventory.read_json(tmp_path / "summary.json")
    assert summary["xpass_count"] == int(listed is True)
    assert summary["stale_xfail_count"] == 2
    results = emule_inventory.read_json(tmp_path / "results.json")
    lit = next(entry for entry in results["groups"] if entry["suite"] == "lit")
    assert lit["complete"]
    assert (lit["status"], lit["outcomes"]) == {
        True: ("passed", {"passed": 1}),
        False: ("passed", {"xfail": 1}),
        None: ("failed", {"failed": 1}),
    }[listed]
    assert ("--xfail=" + case in commands[0]) is bool(listed)
    assert any(arg.startswith("--filter-out") for arg in commands[0]) is (
        listed is False
    )


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
    write_xfail(args.source, {})
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
                names = emule_inventory.lit_names(tests, suite.rstrip("/") + "/")
                emule_inventory.write_json(
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
            nodes, markers = {
                "packaging": (PACKAGING_NODES, {}),
                "me2e": (DEVICE_NODES, DEVICE_MARKERS),
                "tutorial": (APPENDED_NODES, APPENDED_MARKERS),
            }[Path(command[-1]).name]
            Path(env["TTLANG_EMULE_CI_EVENTS"]).write_text(
                json.dumps({"kind": "collection", "nodes": nodes, "markers": markers})
            )
        return {"exit_code": 0, "timed_out": False, "wall_seconds": 1.0}

    monkeypatch.setattr(emule_inventory, "run_command", fake_run)
    emule_inventory.collect(args, {"TTLANG_EMULE_CI_THREADS": "1"})
    produced = emule_inventory.read_json(args.output / "manifest.json")
    expected = manifest(lit=lit_collection(LIT_TESTS, ["TTLang :: python/add.py"]))
    expected["groups"] = [
        (
            {**entry, "markers": {"expected_unsupported": []}}
            if entry["kind"] == "mlir"
            else entry
        )
        for entry in expected["groups"]
    ]
    assert produced == {**expected, "stale_xfail_count": 0}
    assert group(produced, "lit:python/add.py")["markers"] == {
        "expected_unsupported": ["TTLang :: python/add.py"]
    }
    assert sum(command[0] == "llvm-lit" for command in commands) == 7
    # A listed case that was not collected (renamed or removed on main) does
    # not fail the run; it is counted as stale.
    entry = {"reason": "emulator gap", "run": True}
    write_xfail(
        args.source,
        {"python/test_gone.py::test_gone": entry, "TTLang :: python/add.py": entry},
    )
    args.output = tmp_path / "out-xfail"
    args.output.mkdir()
    emule_inventory.collect(args, {})
    produced = emule_inventory.read_json(args.output / "manifest.json")
    assert produced == {**expected, "stale_xfail_count": 1}
    # Without the runtime features every Python lit test would silently become
    # UNSUPPORTED, so collection refuses to produce an inventory at all.
    features["value"] = "optimized system-linux"
    args.output = tmp_path / "out2"
    args.output.mkdir()
    with pytest.raises(ValueError, match="emulator runtime features"):
        emule_inventory.collect(args, {})
    assert not (args.output / "manifest.json").exists()


def activate_emulator(monkeypatch, **variables):
    for key in (
        "TTLANG_COMPILE_ONLY",
        "TTLANG_SIM_ONLY",
        "TT_METAL_SIMULATOR",
        "TTLANG_EMULE_CI_CACHE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TT_METAL_EMULE_MODE", "1")
    monkeypatch.setenv("TT_METAL_MOCK_CLUSTER_DESC_PATH", "descriptor.yaml")
    for key, value in variables.items():
        monkeypatch.setenv(key, value)


@pytest.mark.parametrize("threads", ["1", "2", "3", "4"])
def test_environment_uses_explicit_bounded_threads_and_isolated_cache(
    monkeypatch, tmp_path, threads
):
    activate_emulator(
        monkeypatch,
        TTLANG_EMULE_CI_THREADS=threads,
        TTLANG_EMULE_CI_CACHE="/tt-metal-cache",
    )
    env = emule_inventory.environment(
        SimpleNamespace(source=tmp_path, output=tmp_path / "run")
    )
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert env[key] == threads
    assert env["TT_METAL_CACHE"] == "/tt-metal-cache/run"
    assert env["TT_EMULE_JIT_CACHE_DIR"] == "/tt-metal-cache/run/emule-jit"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["TTLANG_EMULE_CI_XFAIL"] == str(tmp_path / emule_inventory.XFAIL_LIST)
    # Lit writes its outputs under this run's private directory, not the build.
    assert (tmp_path / "run") in Path(env["TTLANG_LIT_TEST_EXEC_ROOT"]).parents
    monkeypatch.setenv("TTLANG_COMPILE_ONLY", "1")
    with pytest.raises(ValueError, match="activate"):
        emule_inventory.environment(
            SimpleNamespace(source=tmp_path, output=tmp_path / "run")
        )


@pytest.mark.parametrize("threads", ["0", "5", "-1", "1.5", "many"])
def test_environment_rejects_unbounded_threads(monkeypatch, tmp_path, threads):
    activate_emulator(monkeypatch, TTLANG_EMULE_CI_THREADS=threads)
    with pytest.raises(ValueError):
        emule_inventory.environment(
            SimpleNamespace(source=tmp_path, output=tmp_path / "run")
        )


# The wrapper registers multi_device in its own conftest, absent here.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnknownMarkWarning")
def test_example_wrapper_matches_the_hardware_examples_runner():
    """Discovery and tags must not drift from compile-and-run-examples.sh."""
    spec = importlib.util.spec_from_file_location(
        "emule_example_scripts", REPO_ROOT / "test/examples/test_example_scripts.py"
    )
    examples = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(examples)
    runner = (REPO_ROOT / ".github/scripts/compile-and-run-examples.sh").read_text()
    # Reuse the runner's own settings and functions without running any script.
    definitions = re.findall(
        r"(?ms)^(?:(?:SCAN_LINES|SKIP_TAG|XFAIL_TAG)=.*?$|\w+\(\) \{.*?^\}$)", runner
    )
    probe = "\n".join(definitions) + (
        "\ncollect_scripts | while IFS= read -r f; do\n"
        '  tags=""\n'
        '  file_has_tag "$ROOT/$f" "$SKIP_TAG" && tags=skip\n'
        '  file_has_tag "$ROOT/$f" "$XFAIL_TAG" && tags="$tags xfail"\n'
        '  printf "%s:%s\\n" "$f" "$tags"\n'
        "done\n"
    )
    listed = subprocess.run(
        ["bash", "-c", probe],
        env={**os.environ, "ROOT": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    shell = {
        script: set(found.split()) for script, found in re.findall(r"(.+):(.*)", listed)
    }
    marks = {
        param.values[0]: {mark.name: mark for mark in param.marks}
        for param in examples.EXAMPLES
    }
    assert set(marks) == set(shell) and len(marks) == len(examples.EXAMPLES)
    for script, found in shell.items():
        # skip-compiler wins over requires-multi-device, as in the runner.
        unsupported = (
            examples.REQUIRES_MESH_TAG in examples.tags(script) and "skip" not in found
        )
        assert ("multi_device" in marks[script]) == unsupported
        assert ("skip" in marks[script]) == ("skip" in found or unsupported)
        expect_failure = found == {"xfail"} and not unsupported
        assert ("xfail" in marks[script]) == expect_failure
        if expect_failure:
            assert marks[script]["xfail"].kwargs["strict"] is True
