# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Collect, time, and shard the complete compiler test inventory on tt-emule.

The inventory mirrors ``check-ttlang-all``: the MLIR, Python-binding and
packaging suites run as compiler-only directory groups, every ``test/python``
and ``test/me2e`` pytest file runs through the emulator without marker
selection, and every Python lit test runs as its own group. Cases a test
declares unsupported on the single-device emulator profile (``multi_device``
pytest skips, lit ``UNSUPPORTED``) are reported explicitly instead of being
excluded, and cases that never ran are counted as ``not_run``.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import time

MANIFEST_VERSION = 2
SELECTION = "complete"
SUITES = ("mlir", "bindings", "packaging", "pytest", "me2e", "lit")
CLASSIFICATION_MARKERS = ("multi_device", "compile_only")
DIRECTORY_GROUPS = {
    "mlir": "ttlang",
    "bindings": "bindings/python",
    "packaging": "packaging",
}
LIT_SUITES = {"lit": "python/", "mlir": "ttlang/", "bindings": "bindings/python/"}
OUTCOMES = (
    "passed",
    "failed",
    "skipped",
    "xfail",
    "unsupported",
    "incomplete",
    "not_run",
)
GROUP_STATUSES = ("passed", "failed", "infrastructure_error")
LIT_OUTCOMES = {
    "PASS": "passed",
    "XFAIL": "xfail",
    "UNSUPPORTED": "unsupported",
    "FAIL": "failed",
    "XPASS": "failed",
    "NOEXE": "failed",
}
LIT_NAME = re.compile(r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:mlir|py)")
MAX_THREADS = 4


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def unique(values):
    if len(values) != len(set(values)):
        raise ValueError("duplicate group or test IDs")


def positive_seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("timings must be finite positive numbers")
    return value


def safe_relative_path(path):
    parsed = PurePosixPath(path)
    return (
        not parsed.is_absolute()
        and ".." not in parsed.parts
        and parsed.as_posix() == path
        and LIT_NAME.fullmatch(path) is not None
    )


def suite_of(group_id):
    kind, _, path = group_id.partition(":")
    if kind == "pytest":
        return "pytest" if path.startswith("python/") else "me2e"
    return kind


def validate_group(group):
    if "kind" not in group or group["kind"] not in (
        "pytest",
        "lit",
        *DIRECTORY_GROUPS,
    ):
        raise ValueError("invalid test group")
    kind = group["kind"]
    expected_keys = {"id", "kind", "path", "cases"}
    if kind == "pytest":
        expected_keys.add("markers")
    if set(group) != expected_keys:
        raise ValueError("unexpected test group fields")
    path, cases = group["path"], group["cases"]
    if (
        not isinstance(path, str)
        or group["id"] != f"{kind}:{path}"
        or not isinstance(cases, list)
        or not cases
        or any(not isinstance(case, str) or "\n" in case for case in cases)
    ):
        raise ValueError("invalid test group")
    if kind == "pytest":
        if not safe_relative_path(path) or not re.fullmatch(
            r"(?:python|me2e)/.+\.py", path
        ):
            raise ValueError("invalid test group")
        if any(not case.startswith(path + "::") for case in cases):
            raise ValueError("pytest group contains a test from another file")
        markers = group["markers"]
        if not isinstance(markers, dict) or set(markers) != set(CLASSIFICATION_MARKERS):
            raise ValueError("pytest group requires its classification markers")
        for marked in markers.values():
            if not isinstance(marked, list) or not set(marked) <= set(cases):
                raise ValueError("classified cases must belong to the group")
            unique(marked)
    elif kind == "lit":
        if not safe_relative_path(path) or not re.fullmatch(r"python/.+\.py", path):
            raise ValueError("invalid test group")
        if cases != ["TTLang :: " + path]:
            raise ValueError("invalid lit test ID")
    else:
        if path != DIRECTORY_GROUPS[kind]:
            raise ValueError("invalid test group")
        for case in cases:
            if kind == "packaging":
                valid = case.startswith(path + "/") and "::" in case
                valid = valid and safe_relative_path(case.split("::", 1)[0])
            else:
                valid = case.startswith("TTLang :: " + path + "/") and (
                    safe_relative_path(case[len("TTLang :: ") :])
                )
            if not valid:
                raise ValueError("invalid test group")


def validate_manifest(manifest):
    if set(manifest) != {"version", "revision", "runtime_id", "selection", "groups"}:
        raise ValueError("unexpected manifest fields")
    if (
        manifest.get("version") != MANIFEST_VERSION
        or manifest.get("selection") != SELECTION
    ):
        raise ValueError("unsupported manifest version or selection")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("revision", "")):
        raise ValueError("manifest requires a full source revision")
    if not re.fullmatch(
        r"(?:[^\s@]+@)?sha256:[0-9a-f]{64}", manifest.get("runtime_id", "")
    ):
        raise ValueError("manifest requires an immutable runtime image digest")
    if not manifest.get("groups"):
        raise ValueError("manifest requires nonempty groups")
    unique([group["id"] for group in manifest["groups"]])
    cases = []
    for group in manifest["groups"]:
        if not isinstance(group, dict):
            raise ValueError("invalid test group")
        validate_group(group)
        cases.extend(group["cases"])
    unique(cases)


def lit_names(output, prefix):
    """Parse ``llvm-lit --show-tests`` output for one suite directory."""
    names = []
    for line in output.splitlines():
        match = re.fullmatch(r"\s+TTLang :: (\S.*)", line)
        if " :: " in line and not match:
            raise ValueError("unexpected lit collection entry")
        if match:
            name = match.group(1)
            if not name.startswith(prefix) or not safe_relative_path(name):
                raise ValueError("unexpected lit collection entry")
            if prefix == LIT_SUITES["lit"] and not name.endswith(".py"):
                raise ValueError("unexpected lit collection entry")
            names.append(name)
    unique(names)
    return names


def pytest_groups(inventory):
    nodes = inventory["nodes"]
    marked = inventory.get("markers", {})
    if not isinstance(nodes, list) or not isinstance(marked, dict):
        raise ValueError("invalid pytest collection inventory")
    groups = {}
    for node in nodes:
        path = node.split("::", 1)[0]
        key = "pytest:" + path
        group = groups.setdefault(
            key,
            {
                "id": key,
                "kind": "pytest",
                "path": path,
                "cases": [],
                "markers": {marker: [] for marker in CLASSIFICATION_MARKERS},
            },
        )
        group["cases"].append(node)
        for marker in CLASSIFICATION_MARKERS:
            if marker in marked.get(node, []):
                group["markers"][marker].append(node)
    return list(groups.values())


def make_manifest(collected, revision, runtime_id):
    """Build the complete inventory from collection outputs.

    ``collected`` maps ``pytest`` and ``packaging`` to collection inventories
    (``nodes`` plus optional ``markers``), and ``lit``, ``mlir`` and
    ``bindings`` to their ``llvm-lit --show-tests`` output.
    """
    groups = []
    for kind in ("mlir", "bindings"):
        path = DIRECTORY_GROUPS[kind]
        names = lit_names(collected[kind], LIT_SUITES[kind])
        groups.append(
            {
                "id": f"{kind}:{path}",
                "kind": kind,
                "path": path,
                "cases": ["TTLang :: " + name for name in names],
            }
        )
    packaging = collected["packaging"]["nodes"]
    groups.append(
        {
            "id": "packaging:" + DIRECTORY_GROUPS["packaging"],
            "kind": "packaging",
            "path": DIRECTORY_GROUPS["packaging"],
            "cases": list(packaging),
        }
    )
    groups.extend(pytest_groups(collected["pytest"]))
    for name in lit_names(collected["lit"], LIT_SUITES["lit"]):
        groups.append(
            {
                "id": "lit:" + name,
                "kind": "lit",
                "path": name,
                "cases": ["TTLang :: " + name],
            }
        )
    manifest = {
        "version": MANIFEST_VERSION,
        "revision": revision,
        "runtime_id": runtime_id,
        "selection": SELECTION,
        "groups": groups,
    }
    validate_manifest(manifest)
    if {suite_of(group["id"]) for group in groups} != set(SUITES):
        raise ValueError("every compiler suite must collect tests")
    return manifest


def read_events(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


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


def lit_coverage(group, tests):
    return sorted(test["name"] for test in tests) == sorted(group["cases"]) and all(
        test["code"] not in ("UNRESOLVED", "TIMEOUT") for test in tests
    )


def lit_outcomes(group, tests):
    codes = {test["name"]: test["code"] for test in tests}
    counts = Counter()
    for case in group["cases"]:
        code = codes.get(case)
        counts["not_run" if code is None else LIT_OUTCOMES.get(code, "incomplete")] += 1
    return dict(counts)


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
        "manifest_digest": results["manifest_digest"],
        "selection": SELECTION,
        "shard": results["shard"],
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


def balance(manifest, results, count):
    validate_manifest(manifest)
    if not 1 <= count <= len(manifest["groups"]):
        raise ValueError("shard count must be between one and the group count")
    if results.get("manifest_digest") != digest(manifest):
        raise ValueError("timings belong to another manifest")
    measured = results["groups"]
    unique([group["id"] for group in measured])
    durations = {}
    for group in measured:
        if group["complete"] is not True or group["status"] not in ("passed", "failed"):
            raise ValueError("incomplete or infrastructure-failed timing baseline")
        durations[group["id"]] = positive_seconds(group["wall_seconds"])
    if not public_summary(results)["runtime_execution_observed"]:
        raise ValueError(
            "baseline did not observe compiler-generated runtime execution"
        )
    ids = [group["id"] for group in manifest["groups"]]
    if set(durations) != set(ids):
        raise ValueError("timings must cover every group exactly once")
    shards, totals = [[] for _ in range(count)], [0.0] * count
    for group in sorted(ids, key=lambda key: (-durations[key], key)):
        index = min(range(count), key=lambda index: (totals[index], index))
        shards[index].append(group)
        totals[index] += durations[group]
    order = {key: index for index, key in enumerate(ids)}
    return {
        "manifest_digest": digest(manifest),
        "shards": [sorted(shard, key=order.__getitem__) for shard in shards],
        "estimated_seconds": totals,
    }


def selected_groups(manifest, plan, shard):
    if plan is None:
        if shard != 1:
            raise ValueError("serial execution has only shard 1")
        return manifest["groups"]
    if plan.get("manifest_digest") != digest(manifest):
        raise ValueError("shard plan belongs to another manifest")
    shards = plan["shards"]
    flattened = [key for group in shards for key in group]
    unique(flattened)
    if (
        not 1 <= shard <= len(shards)
        or any(not group for group in shards)
        or set(flattened) != {group["id"] for group in manifest["groups"]}
    ):
        raise ValueError("shards must cover every group exactly once")
    wanted = set(shards[shard - 1])
    return [group for group in manifest["groups"] if group["id"] in wanted]


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


def validate_public_source(source, manifest):
    validate_source(source, manifest["revision"])
    tree = subprocess.run(
        [
            "git",
            "-c",
            "safe.directory=" + str(source),
            "-C",
            str(source),
            "ls-tree",
            "-r",
            "-z",
            manifest["revision"],
            "--",
            "test",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    regular_files = set()
    for entry in tree.stdout.split("\0"):
        if entry:
            metadata, path = entry.split("\t", 1)
            mode, kind, _ = metadata.split()
            if mode in ("100644", "100755") and kind == "blob":
                regular_files.add(path)
    for group in manifest["groups"]:
        path = "test/" + group["path"]
        if group["kind"] in DIRECTORY_GROUPS:
            tracked = any(name.startswith(path + "/") for name in regular_files)
        else:
            tracked = path in regular_files
        if not tracked:
            raise ValueError("public plan paths must be tracked regular source files")


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
    for kind, path in (("lit", "python"), *DIRECTORY_GROUPS.items()):
        if kind == "packaging":
            continue
        processes[kind] = run_command(
            [args.lit, "--show-tests", str(args.build / "test" / path)],
            args.output / f"{kind}.log",
            600,
            env,
            args.source,
        )
    write_json(args.output / "collection.json", processes)
    if any(result["exit_code"] != 0 for result in processes.values()):
        raise ValueError("collection failed; inspect collection.json and logs")
    collected = {
        "pytest": collection_inventory(device_events),
        "packaging": collection_inventory(packaging_events),
    }
    for kind in ("lit", "mlir", "bindings"):
        collected[kind] = (args.output / f"{kind}.log").read_text()
    manifest = make_manifest(collected, args.revision, args.runtime_id)
    write_json(args.output / "manifest.json", manifest)


def group_command(args, group, prefix, env):
    kind = group["kind"]
    if kind in ("pytest", "packaging"):
        command = pytest_command(args) if kind == "pytest" else packaging_command(args)
        return command + [
            "--timeout=" + str(args.timeout),
            "--timeout-method=thread",
            "--reruns=0",
            "--junitxml=" + str(prefix.with_suffix(".xml")),
            str(args.source / "test" / group["path"]),
        ]
    jobs = 1 if kind == "lit" else lit_jobs(env)
    return [
        args.lit,
        f"-j{jobs}",
        "--timeout=" + str(args.timeout),
        "-v",
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
    plan = read_json(args.plan) if args.plan else None
    groups = selected_groups(manifest, plan, args.shard)
    planned = {suite: {"groups": 0, "cases": 0} for suite in SUITES}
    for group in groups:
        planned[suite_of(group["id"])]["groups"] += 1
        planned[suite_of(group["id"])]["cases"] += len(group["cases"])
    summary = {
        "manifest_digest": digest(manifest),
        "shard": args.shard,
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
        result = run_command(
            group_command(args, group, prefix, env),
            prefix.with_suffix(".log"),
            args.group_timeout,
            env,
            args.source,
        )
        result.update(id=group["id"], suite=suite_of(group["id"]), complete=False)
        try:
            if group["kind"] in ("pytest", "packaging"):
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
                tests = read_json(prefix.with_suffix(".lit.json"))["tests"]
                result["complete"] = lit_coverage(group, tests)
                result["lit_codes"] = dict(Counter(test["code"] for test in tests))
                result["outcomes"] = lit_outcomes(group, tests)
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["report_error"] = str(error)
        result["status"] = (
            "infrastructure_error"
            if not result["complete"]
            or result["timed_out"]
            or result["exit_code"] not in (0, 1)
            else "passed" if result["exit_code"] == 0 else "failed"
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
            command.add_argument("--plan", type=Path)
            command.add_argument("--shard", type=int, default=1)
            command.add_argument("--timeout", type=int, default=600)
            command.add_argument("--group-timeout", type=int, default=7200)
    command = commands.add_parser("balance")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--results", type=Path, required=True)
    command.add_argument("--shards", type=int, default=3)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--public-source", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "balance":
            if args.output.exists():
                raise ValueError("choose a new output path")
            manifest = read_json(args.manifest)
            plan = balance(manifest, read_json(args.results), args.shards)
            if args.public_source:
                validate_public_source(args.public_source.resolve(), manifest)
            write_json(args.output, plan)
            return 0
        args.source, args.build, args.output = (
            args.source.resolve(),
            args.build.resolve(),
            args.output.resolve(),
        )
        validate_source(args.source, args.revision)
        if args.command == "run":
            positive_seconds(args.timeout)
            positive_seconds(args.group_timeout)
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
