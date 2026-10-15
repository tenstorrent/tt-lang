# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Collect and time the complete compiler test inventory on tt-emule.

The inventory mirrors ``check-ttlang-all``: the MLIR, Python-binding and
packaging suites run as compiler-only directory groups, every ``test/python``
and ``test/me2e`` pytest file runs through the emulator without marker
selection, and every Python lit test runs as its own group. The hardware
example and tutorial scripts and the binding pytest file run last. Cases a test
declares unsupported on the single-device emulator profile (``multi_device``
pytest skips, lit ``UNSUPPORTED``) are reported explicitly instead of being
excluded, and cases that never ran are counted as ``not_run``. Cases the
checked-in emulator xfail list names are non-strict expected failures here only.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import time

SUITES = (
    "mlir",
    "bindings",
    "packaging",
    "pytest",
    "me2e",
    "lit",
    "examples",
    "tutorial",
)
PYTEST_KINDS = ("pytest", "packaging")
LIT_KINDS = ("lit", "mlir", "bindings")
CLASSIFICATION_MARKERS = ("multi_device",)
# lit features that prove the emulator runtime is visible to the Python lit
# suite; without them its tests silently become UNSUPPORTED.
REQUIRED_LIT_FEATURES = ("ttnn", "tt-device")
DIRECTORY_GROUPS = {
    "mlir": "ttlang",
    "bindings": "bindings/python",
    "packaging": "packaging",
}
LIT_SUITES = {"lit": "python/", "mlir": "ttlang/", "bindings": "bindings/python/"}
PYTEST_SUITES = {
    "python/": "pytest",
    "me2e/": "me2e",
    "bindings/": "bindings",
    "examples/": "examples",
    "tutorial/": "tutorial",
}
# Collected separately and run after every other group, cheapest first, so a
# deadline overrun loses only these; lit skips pytest files, so no other CI
# runs the binding file.
APPENDED_PYTEST_PATHS = (
    "bindings/python/test_ttl_ttcore_integration.py",
    "examples",
    "tutorial",
)
LIT_OUTCOMES = {
    "PASS": "passed",
    "XFAIL": "xfail",
    "UNSUPPORTED": "unsupported",
    "FAIL": "failed",
    "XPASS": "failed",
    "NOEXE": "failed",
}
MAX_THREADS = 4
XFAIL_LIST = "config/tt-lang-emule-xfail.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    # Replace atomically: the worker may read a summary while it is rewritten.
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def unique(values):
    if len(values) != len(set(values)):
        raise ValueError("duplicate group or test IDs")


def unique_keys(pairs):
    unique([key for key, _ in pairs])
    return dict(pairs)


def xfail_list(path):
    """Load the emulator-only expected failures: case ID -> reason and run."""
    listing = json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_keys
    )
    if (
        not isinstance(listing, dict)
        or set(listing) != {"schema_version", "tests"}
        or listing["schema_version"] != 1
        or not isinstance(listing["tests"], dict)
        or any(
            not isinstance(entry, dict)
            or set(entry) != {"reason", "run"}
            or not isinstance(entry["reason"], str)
            or not entry["reason"].strip()
            or not isinstance(entry["run"], bool)
            for entry in listing["tests"].values()
        )
    ):
        raise ValueError("malformed emulator xfail list")
    return listing["tests"]


def suite_of(group_id):
    kind, _, path = group_id.partition(":")
    if kind != "pytest":
        return kind
    for prefix, suite in PYTEST_SUITES.items():
        if path.startswith(prefix):
            return suite
    raise ValueError("pytest group outside the known compiler suites")


def validate_group(group):
    """Check one group of the inventory this job collected and then runs."""
    kind, path, cases = group["kind"], PurePosixPath(group["path"]), group["cases"]
    prefix = ("TTLang :: " if kind in LIT_KINDS else "") + group["path"]
    if (
        kind not in (*PYTEST_KINDS, *LIT_KINDS)
        or group["id"] != f"{kind}:{path}"
        or path.is_absolute()
        or ".." in path.parts
        or not cases
        or any(not case.startswith(prefix) for case in cases)
    ):
        raise ValueError("invalid test group")
    if any(not set(marked) <= set(cases) for marked in group["markers"].values()):
        raise ValueError("classified cases must belong to the group")


def validate_manifest(manifest):
    if not manifest["groups"]:
        raise ValueError("manifest requires nonempty groups")
    for group in manifest["groups"]:
        validate_group(group)
    unique([group["id"] for group in manifest["groups"]])
    unique([case for group in manifest["groups"] for case in group["cases"]])


def lit_names(output, prefix):
    """Parse ``llvm-lit --show-tests`` output for one suite directory."""
    names = []
    for line in output.splitlines():
        match = re.fullmatch(r"\s+TTLang :: (\S.*)", line)
        if " :: " in line and not match:
            raise ValueError("unexpected lit collection entry")
        if match:
            name = match.group(1)
            if not name.startswith(prefix) or (
                prefix == LIT_SUITES["lit"] and not name.endswith(".py")
            ):
                raise ValueError("unexpected lit collection entry")
            names.append(name)
    unique(names)
    return names


def pytest_markers(cases, marked):
    if not isinstance(marked, dict):
        raise ValueError("invalid pytest collection inventory")
    return {
        marker: [case for case in cases if marker in marked.get(case, [])]
        for marker in CLASSIFICATION_MARKERS
    }


def pytest_groups(inventory):
    nodes = inventory["nodes"]
    if not isinstance(nodes, list):
        raise ValueError("invalid pytest collection inventory")
    groups = {}
    for node in nodes:
        path = node.split("::", 1)[0]
        key = "pytest:" + path
        group = groups.setdefault(
            key, {"id": key, "kind": "pytest", "path": path, "cases": []}
        )
        group["cases"].append(node)
    for group in groups.values():
        group["markers"] = pytest_markers(group["cases"], inventory.get("markers", {}))
    return list(groups.values())


def lit_markers(cases, expected_unsupported):
    """Cases lit itself reports UNSUPPORTED before execution (``--no-execute``)."""
    if not isinstance(expected_unsupported, list) or not set(
        expected_unsupported
    ) <= set(cases):
        raise ValueError("expected unsupported lit tests must be collected tests")
    return {"expected_unsupported": [c for c in cases if c in expected_unsupported]}


def lit_features(output):
    """Parse the available features from ``llvm-lit --show-suites`` output."""
    features = set()
    found = False
    for line in output.splitlines():
        prefix, separator, listed = line.strip().partition("Available Features:")
        if separator and not prefix:
            found = True
            features.update(listed.split())
    if not found:
        raise ValueError("lit did not report its available features")
    return features


def make_manifest(collected, revision, runtime_id):
    """Build the complete inventory from collection outputs.

    ``collected`` maps ``pytest``, ``packaging`` and ``appended`` to collection
    inventories (``nodes`` plus optional ``markers``), and ``lit``, ``mlir`` and
    ``bindings`` to their ``llvm-lit --show-tests`` output (``tests``) with the
    full names lit reported UNSUPPORTED without executing anything
    (``expected_unsupported``).
    """
    groups = []
    for kind in ("mlir", "bindings"):
        path = DIRECTORY_GROUPS[kind]
        names = lit_names(collected[kind]["tests"], LIT_SUITES[kind])
        cases = ["TTLang :: " + name for name in names]
        groups.append(
            {
                "id": f"{kind}:{path}",
                "kind": kind,
                "path": path,
                "cases": cases,
                "markers": lit_markers(cases, collected[kind]["expected_unsupported"]),
            }
        )
    packaging = collected["packaging"]
    if not isinstance(packaging["nodes"], list):
        raise ValueError("invalid pytest collection inventory")
    groups.append(
        {
            "id": "packaging:" + DIRECTORY_GROUPS["packaging"],
            "kind": "packaging",
            "path": DIRECTORY_GROUPS["packaging"],
            "cases": list(packaging["nodes"]),
            "markers": pytest_markers(packaging["nodes"], packaging.get("markers", {})),
        }
    )
    groups.extend(pytest_groups(collected["pytest"]))
    expected_unsupported = collected["lit"]["expected_unsupported"]
    if not isinstance(expected_unsupported, list):
        raise ValueError("expected unsupported lit tests must be collected tests")
    for name in lit_names(collected["lit"]["tests"], LIT_SUITES["lit"]):
        cases = ["TTLang :: " + name]
        groups.append(
            {
                "id": "lit:" + name,
                "kind": "lit",
                "path": name,
                "cases": cases,
                "markers": lit_markers(
                    cases, [c for c in expected_unsupported if c in cases]
                ),
            }
        )
    unexpected = set(expected_unsupported) - {
        case for group in groups if group["kind"] == "lit" for case in group["cases"]
    }
    if unexpected:
        raise ValueError("expected unsupported lit tests must be collected tests")
    groups.extend(pytest_groups(collected["appended"]))
    manifest = {"revision": revision, "runtime_id": runtime_id, "groups": groups}
    validate_manifest(manifest)
    if {suite_of(group["id"]) for group in groups} != set(SUITES):
        raise ValueError("every compiler suite must collect tests")
    return manifest


def read_events(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    events = []
    for index, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except ValueError:
            # A process killed mid-write leaves at most one truncated final
            # line; the coverage check still fails because its finish event is
            # missing. Corruption anywhere else is an error.
            if index != len(lines) - 1:
                raise
    return events


def collection_inventory(events_path):
    collections = [
        event for event in read_events(events_path) if event["kind"] == "collection"
    ]
    if len(collections) != 1:
        raise ValueError("pytest collection did not report its complete inventory")
    return {
        "nodes": collections[0]["nodes"],
        "markers": collections[0].get("markers", {}),
    }


def pytest_coverage(group, events):
    collections = [event["nodes"] for event in events if event["kind"] == "collection"]
    finished = [event["nodeid"] for event in events if event["kind"] == "finish"]
    return (
        collections == [group["cases"]]
        and len(finished) == len(set(finished))
        and set(finished) == set(group["cases"])
    )


def pytest_outcomes(group, events):
    """Account for every planned case, including cases that never ran."""
    reports = {}
    for event in events:
        if event["kind"] == "report":
            reports.setdefault(event["nodeid"], []).append(event)
    unsupported = set(group.get("markers", {}).get("multi_device", []))
    counts = Counter()
    for case in group["cases"]:
        node_reports = reports.get(case, [])
        if not node_reports:
            outcome = "not_run"
        elif any(report["outcome"] == "failed" for report in node_reports):
            outcome = "failed"
        elif any(report["outcome"] == "skipped" for report in node_reports):
            if any(report["expected_failure"] for report in node_reports):
                outcome = "xfail"
            elif case in unsupported:
                outcome = "unsupported"
            else:
                outcome = "skipped"
        elif any(report["phase"] == "call" for report in node_reports):
            outcome = "passed"
        else:
            outcome = "incomplete"
        counts[outcome] += 1
    return dict(counts)


def lit_outcome(group, case, code, listed):
    """Normalize one lit result code; UNSUPPORTED counts only where expected.

    ``listed`` maps the group's xfail-listed cases to whether they run: lit
    reports a listed failure as XFAIL, a listed pass (non-strict) counts as
    passed, and a listed case that must not run is an xfail without a code.
    """
    if code is None:
        return "xfail" if listed.get(case) is False else "not_run"
    if code == "XPASS" and case in listed:
        return "passed"
    if code == "UNSUPPORTED":
        expected = group.get("markers", {}).get("expected_unsupported", [])
        return "unsupported" if case in expected else "incomplete"
    return LIT_OUTCOMES.get(code, "incomplete")


def lit_coverage(group, tests, listed):
    """Every case meant to run reported exactly once with a verdict lit executed."""
    planned = [case for case in group["cases"] if listed.get(case, True)]
    return sorted(test["name"] for test in tests) == sorted(planned) and all(
        lit_outcome(group, test["name"], test["code"], listed) != "incomplete"
        for test in tests
    )


def lit_outcomes(group, tests, listed):
    codes = {test["name"]: test["code"] for test in tests}
    counts = Counter()
    for case in group["cases"]:
        counts[lit_outcome(group, case, codes.get(case), listed)] += 1
    return dict(counts)


def lit_report(path):
    """The tests a lit JSON report lists, or none when lit left no report."""
    if not path.exists():
        return []
    tests = read_json(path)["tests"]
    if not isinstance(tests, list) or any(
        not isinstance(test.get("name"), str) or not isinstance(test.get("code"), str)
        for test in tests
    ):
        raise ValueError("invalid lit report")
    return tests


def suite_counts(results):
    planned = results.get("planned_suite_counts", {})
    counts = {
        suite: {
            "planned_groups": planned.get(suite, {}).get("groups", 0),
            "planned_cases": planned.get(suite, {}).get("cases", 0),
            "groups": 0,
            "complete_groups": 0,
            "status_counts": {},
            "outcome_counts": {},
        }
        for suite in SUITES
    }
    for group in results["groups"]:
        suite = counts[suite_of(group["id"])]
        suite["groups"] += 1
        suite["complete_groups"] += int(group["complete"])
        suite["status_counts"][group["status"]] = (
            suite["status_counts"].get(group["status"], 0) + 1
        )
        for outcome, count in group.get("outcomes", {}).items():
            suite["outcome_counts"][outcome] = (
                suite["outcome_counts"].get(outcome, 0) + count
            )
    return counts


def public_summary(results):
    groups = results["groups"]
    launches = sum(group.get("launch_counts", {}).get("launch", 0) for group in groups)
    returned = sum(
        group.get("launch_counts", {}).get("returned", 0) for group in groups
    )
    outcomes = Counter()
    for group in groups:
        outcomes.update(group.get("outcomes", {}))
    return {
        "planned_group_count": results.get("planned_group_count", len(groups)),
        "planned_case_count": results.get("planned_case_count", 0),
        "group_count": len(groups),
        "complete_group_count": sum(group["complete"] for group in groups),
        "group_status_counts": dict(Counter(group["status"] for group in groups)),
        "case_outcome_counts": dict(outcomes),
        "suite_counts": suite_counts(results),
        "wall_seconds": sum(group["wall_seconds"] for group in groups),
        "generic_op_launches": launches,
        "generic_op_returns": returned,
        "runtime_execution_observed": returned > 0,
    }


def run_command(command, output, limit, env, source):
    started = time.monotonic()
    result = {"command": command, "exit_code": None, "timed_out": False}
    with output.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                command,
                cwd=source,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as error:
            result["error"] = str(error)
        else:
            try:
                process.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                result["timed_out"] = True
            finally:
                # Thread timeouts or native crashes can leave JIT children alive.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            result["exit_code"] = process.returncode
    result["wall_seconds"] = time.monotonic() - started
    return result


def pytest_command(args):
    """The device suites: every test/python and test/me2e case, no selection."""
    return [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        str(args.build / "test/pytest.ini"),
        "--rootdir=" + str(args.source / "test"),
        "-p",
        "observer",
        "-q",
        "-o",
        "addopts=--order-scope=class",
        "-o",
        "cache_dir=" + str(args.output / "pytest-cache"),
    ]


def packaging_command(args):
    """The packaging suite, invoked as check-ttlang-packaging does."""
    return [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        os.devnull,
        "--rootdir=" + str(args.source / "test"),
        "-p",
        "observer",
        "-q",
        "-o",
        "cache_dir=" + str(args.output / "pytest-cache"),
    ]


def lit_jobs(env):
    threads = int(env.get("TTLANG_EMULE_CI_THREADS", "1"))
    if not 1 <= threads <= MAX_THREADS:
        raise ValueError("emulator CI thread count must be between one and four")
    return threads


def environment(args):
    env = dict(os.environ)
    if (
        env.get("TT_METAL_EMULE_MODE") != "1"
        or not env.get("TT_METAL_MOCK_CLUSTER_DESC_PATH")
        or env.get("TTLANG_COMPILE_ONLY")
        or env.get("TTLANG_SIM_ONLY")
        or env.get("TT_METAL_SIMULATOR")
    ):
        raise ValueError("activate the installed tt-emule compiler environment first")
    env["PYTHONPATH"] = (
        str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
    )
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cache = args.output / "cache"
    if env.get("TTLANG_EMULE_CI_CACHE"):
        base = Path(env["TTLANG_EMULE_CI_CACHE"])
        if not base.is_absolute() or base == Path("/"):
            raise ValueError(
                "explicit emulator cache must be an absolute non-root directory"
            )
        cache = base / args.output.name
    env["TT_METAL_CACHE"] = str(cache)
    env["TT_EMULE_JIT_CACHE_DIR"] = str(cache / "emule-jit")
    env["TTLANG_LIT_TEST_EXEC_ROOT"] = str(args.output / "lit-exec")
    # The observer marks the listed pytest cases; lit gets them as arguments.
    env["TTLANG_EMULE_CI_XFAIL"] = str(args.source / XFAIL_LIST)
    threads = lit_jobs(env) if "TTLANG_EMULE_CI_THREADS" in env else MAX_THREADS
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = str(threads)
    return env


def validate_source(source, revision):
    git = ["git", "-c", "safe.directory=" + str(source), "-C", str(source)]
    actual = subprocess.run(
        [*git, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if actual != revision:
        raise ValueError("source HEAD differs from the requested revision")
    clean = subprocess.run(
        [*git, "status", "--porcelain", "--untracked-files=normal"],
        capture_output=True,
        text=True,
        check=True,
    )
    if clean.stdout.strip():
        raise ValueError("benchmark source must be a clean committed checkout")


def collect(args, env):
    processes = {}
    device_events = args.output / "collection.events.jsonl"
    env["TTLANG_EMULE_CI_EVENTS"] = str(device_events)
    processes["pytest"] = run_command(
        pytest_command(args)
        + [
            "--collect-only",
            str(args.source / "test/python"),
            str(args.source / "test/me2e"),
        ],
        args.output / "pytest.log",
        600,
        env,
        args.source,
    )
    packaging_events = args.output / "packaging.events.jsonl"
    env["TTLANG_EMULE_CI_EVENTS"] = str(packaging_events)
    processes["packaging"] = run_command(
        packaging_command(args)
        + ["--collect-only", str(args.source / "test" / DIRECTORY_GROUPS["packaging"])],
        args.output / "packaging.log",
        600,
        env,
        args.source,
    )
    appended_events = args.output / "appended.events.jsonl"
    env["TTLANG_EMULE_CI_EVENTS"] = str(appended_events)
    processes["appended"] = run_command(
        pytest_command(args)
        + ["--collect-only"]
        + [str(args.source / "test" / path) for path in APPENDED_PYTEST_PATHS],
        args.output / "appended.log",
        600,
        env,
        args.source,
    )
    for kind in LIT_KINDS:
        suite = str(args.build / "test" / LIT_SUITES[kind].rstrip("/"))
        processes[kind] = run_command(
            [args.lit, "--show-tests", suite],
            args.output / f"{kind}.log",
            600,
            env,
            args.source,
        )
        # Evaluate every test's REQUIRES/UNSUPPORTED directives without running
        # anything: lit still reports UNSUPPORTED under --no-execute, so the
        # inventory records which cases may legitimately be unsupported.
        processes[kind + "_probe"] = run_command(
            [
                args.lit,
                f"-j{lit_jobs(env)}",
                "--no-execute",
                "-o",
                str(args.output / f"{kind}.probe.json"),
                suite,
            ],
            args.output / f"{kind}.probe.log",
            600,
            env,
            args.source,
        )
    processes["lit_features"] = run_command(
        [args.lit, "--show-suites", str(args.build / "test/python")],
        args.output / "lit-features.log",
        600,
        env,
        args.source,
    )
    write_json(args.output / "collection.json", processes)
    # The --no-execute probes may exit 1 (an XFAIL test counts as XPASS when
    # nothing runs); every other collection command must succeed.
    if any(
        result["exit_code"] not in ((0, 1) if name.endswith("_probe") else (0,))
        for name, result in processes.items()
    ):
        raise ValueError("collection failed; inspect collection.json and logs")
    features = lit_features((args.output / "lit-features.log").read_text())
    if not set(REQUIRED_LIT_FEATURES) <= features:
        raise ValueError("lit does not detect the emulator runtime features")
    collected = {
        "pytest": collection_inventory(device_events),
        "packaging": collection_inventory(packaging_events),
        "appended": collection_inventory(appended_events),
    }
    for kind in LIT_KINDS:
        probe = lit_report(args.output / f"{kind}.probe.json")
        collected[kind] = {
            "tests": (args.output / f"{kind}.log").read_text(),
            "expected_unsupported": [
                test["name"] for test in probe if test["code"] == "UNSUPPORTED"
            ],
        }
    manifest = make_manifest(collected, args.revision, args.runtime_id)
    planned = {case for group in manifest["groups"] for case in group["cases"]}
    unknown = set(xfail_list(args.source / XFAIL_LIST)) - planned
    if unknown:
        raise ValueError(
            f"the emulator xfail list names {len(unknown)} uncollected cases"
        )
    write_json(args.output / "manifest.json", manifest)


def group_command(args, group, prefix, env, listed):
    kind = group["kind"]
    if kind in PYTEST_KINDS:
        # Only the device suites need the observer to wrap ttnn.generic_op:
        # packaging and the binding file stay host-only, and the tutorial and
        # example scripts launch kernels in child processes it cannot see.
        observe = suite_of(group["id"]) in ("pytest", "me2e")
        env["TTLANG_EMULE_CI_OBSERVE_RUNTIME"] = "1" if observe else "0"
        # Those scripts bound each child process themselves.
        env["TUTORIAL_TIMEOUT_SECONDS"] = str(args.timeout)
        env["TTLANG_EXAMPLE_TIMEOUT_SECONDS"] = str(args.timeout)
        command = pytest_command(args) if kind == "pytest" else packaging_command(args)
        return command + [
            "--timeout=" + str(args.timeout),
            "--timeout-method=thread",
            "--reruns=0",
            "--junitxml=" + str(prefix.with_suffix(".xml")),
            str(args.source / "test" / group["path"]),
        ]
    jobs = 1 if kind == "lit" else lit_jobs(env)
    command = [args.lit, f"-j{jobs}", "--timeout=" + str(args.timeout), "-v"]
    expected = [case for case, run in listed.items() if run]
    if expected:
        command.append("--xfail=" + ";".join(expected))
    excluded = [re.escape(case) for case, run in listed.items() if not run]
    if excluded:
        # lit omits filtered-out cases from its report; a group of only such
        # cases then runs nothing and exits successfully.
        command += ["--allow-empty-runs", f"--filter-out=^(?:{'|'.join(excluded)})$"]
    return command + [
        "-o",
        str(prefix.with_suffix(".lit.json")),
        str(args.build / "test" / group["path"]),
    ]


def run(args, env):
    manifest = read_json(args.manifest)
    validate_manifest(manifest)
    if (args.revision, args.runtime_id) != (
        manifest["revision"],
        manifest["runtime_id"],
    ):
        raise ValueError("source/runtime identity differs from collected inventory")
    groups = manifest["groups"]
    xfail = xfail_list(args.source / XFAIL_LIST)
    planned = {suite: {"groups": 0, "cases": 0} for suite in SUITES}
    for group in groups:
        planned[suite_of(group["id"])]["groups"] += 1
        planned[suite_of(group["id"])]["cases"] += len(group["cases"])
    summary = {
        "planned_group_count": len(groups),
        "planned_case_count": sum(len(group["cases"]) for group in groups),
        "planned_suite_counts": planned,
        "groups": [],
    }
    write_json(args.output / "results.json", summary)
    write_json(args.output / "summary.json", public_summary(summary))
    for index, group in enumerate(groups):
        prefix = args.output / f"group-{index:04d}"
        events_path = prefix.with_suffix(".events.jsonl")
        env["TTLANG_EMULE_CI_EVENTS"] = str(events_path)
        listed = {case: xfail[case]["run"] for case in group["cases"] if case in xfail}
        result = run_command(
            group_command(args, group, prefix, env, listed),
            prefix.with_suffix(".log"),
            args.group_timeout,
            env,
            args.source,
        )
        result.update(id=group["id"], suite=suite_of(group["id"]), complete=False)
        failed = result["exit_code"] == 1
        try:
            if group["kind"] in PYTEST_KINDS:
                events = read_events(events_path)
                result["complete"] = pytest_coverage(group, events)
                result["launch_counts"] = dict(
                    Counter(
                        event["kind"]
                        for event in events
                        if event["kind"] in ("launch", "returned")
                    )
                )
                result["outcomes"] = pytest_outcomes(group, events)
            else:
                tests = lit_report(prefix.with_suffix(".lit.json"))
                result["complete"] = lit_coverage(group, tests, listed)
                result["lit_codes"] = dict(Counter(test["code"] for test in tests))
                result["outcomes"] = lit_outcomes(group, tests, listed)
                # lit exits 1 for the XPASS of a listed case, a non-strict pass.
                failed = failed and "failed" in result["outcomes"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["report_error"] = str(error)
            # An unreadable report must still account for every planned case.
            result["complete"] = False
            result.setdefault("outcomes", {"not_run": len(group["cases"])})
        result["status"] = (
            "infrastructure_error"
            if not result["complete"]
            or result["timed_out"]
            or result["exit_code"] not in (0, 1)
            else "failed" if failed else "passed"
        )
        summary["groups"].append(result)
        write_json(args.output / "results.json", summary)
        write_json(args.output / "summary.json", public_summary(summary))
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("id", "suite", "status", "complete", "wall_seconds")
                }
            ),
            flush=True,
        )
    if not public_summary(summary)["runtime_execution_observed"]:
        return 2
    return 0 if all(group["status"] == "passed" for group in summary["groups"]) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "run"):
        command = commands.add_parser(name)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--build", type=Path, required=True)
        command.add_argument("--revision", required=True)
        command.add_argument("--runtime-id", required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--lit", default="llvm-lit")
        if name == "run":
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--timeout", type=int, default=600)
            command.add_argument("--group-timeout", type=int, default=7200)
    args = parser.parse_args()
    try:
        args.source, args.build, args.output = (
            args.source.resolve(),
            args.build.resolve(),
            args.output.resolve(),
        )
        validate_source(args.source, args.revision)
        if args.command == "run" and min(args.timeout, args.group_timeout) <= 0:
            raise ValueError("timeouts must be positive")
        env = environment(args)
        args.output.mkdir(parents=True, exist_ok=False)
        if args.command == "collect":
            collect(args, env)
            return 0
        return run(args, env)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(2, f"emule-ci: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
