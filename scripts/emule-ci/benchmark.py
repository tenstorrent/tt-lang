# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Collect, time, and shard the single-device compiler execution suites."""

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

MARKERS = "not multi_device and not compile_only"


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


def validate_manifest(manifest):
    if set(manifest) != {"version", "revision", "runtime_id", "markers", "groups"}:
        raise ValueError("unexpected manifest fields")
    if manifest.get("version") != 1 or manifest.get("markers") != MARKERS:
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
        if set(group) != {"id", "kind", "path", "cases"}:
            raise ValueError("unexpected test group fields")
        kind, path = group["kind"], group["path"]
        parsed = PurePosixPath(path)
        if (
            kind not in ("pytest", "lit")
            or parsed.is_absolute()
            or ".." in parsed.parts
            or parsed.as_posix() != path
            or not path.startswith(("python/", "me2e/"))
            or not path.endswith(".py")
            or "::" in path
            or (kind == "lit" and not path.startswith("python/"))
            or group["id"] != f"{kind}:{path}"
            or not group["cases"]
        ):
            raise ValueError("invalid test group")
        for case in group["cases"]:
            if kind == "pytest" and not case.startswith(path + "::"):
                raise ValueError("pytest group contains a test from another file")
            if kind == "lit" and case != "TTLang :: " + path:
                raise ValueError("invalid lit test ID")
        cases.extend(group["cases"])
    unique(cases)


def public_manifest(manifest):
    validate_manifest(manifest)
    return {
        "version": manifest["version"],
        "revision": manifest["revision"],
        "runtime_id": manifest["runtime_id"],
        "markers": manifest["markers"],
        "groups": [
            {
                "id": group["id"],
                "kind": group["kind"],
                "path": group["path"],
                "cases": list(group["cases"]),
            }
            for group in manifest["groups"]
        ],
    }


def make_manifest(nodes, lit_output, revision, runtime_id):
    groups = {}
    for node in nodes:
        path = node.split("::", 1)[0]
        key = "pytest:" + path
        groups.setdefault(
            key, {"id": key, "kind": "pytest", "path": path, "cases": []}
        )["cases"].append(node)
    for line in lit_output.splitlines():
        match = re.fullmatch(r"\s+TTLang :: (python/.+\.py)", line)
        if " :: " in line and not match:
            raise ValueError("unexpected lit collection entry")
        if match:
            path = match.group(1)
            key = "lit:" + path
            if key in groups:
                raise ValueError("duplicate lit collection entry")
            groups[key] = {
                "id": key,
                "kind": "lit",
                "path": path,
                "cases": ["TTLang :: " + path],
            }
    if not nodes or not any(group["kind"] == "lit" for group in groups.values()):
        raise ValueError("pytest and lit must both collect tests")
    manifest = {
        "version": 1,
        "revision": revision,
        "runtime_id": runtime_id,
        "markers": MARKERS,
        "groups": list(groups.values()),
    }
    validate_manifest(manifest)
    return manifest


def read_events(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def pytest_coverage(group, events):
    collections = [event["nodes"] for event in events if event["kind"] == "collection"]
    finished = [event["nodeid"] for event in events if event["kind"] == "finish"]
    return (
        collections == [group["cases"]]
        and len(finished) == len(set(finished))
        and set(finished) == set(group["cases"])
    )


def pytest_outcomes(events):
    reports = {}
    for event in events:
        if event["kind"] == "report":
            reports.setdefault(event["nodeid"], []).append(event)
    counts = Counter()
    for node_reports in reports.values():
        if any(report["outcome"] == "failed" for report in node_reports):
            outcome = "failed"
        elif any(report["outcome"] == "skipped" for report in node_reports):
            outcome = (
                "xfail"
                if any(report["expected_failure"] for report in node_reports)
                else "skipped"
            )
        elif any(report["phase"] == "call" for report in node_reports):
            outcome = "passed"
        else:
            outcome = "incomplete"
        counts[outcome] += 1
    return dict(counts)


def public_summary(results):
    groups = results["groups"]
    launches = sum(group.get("launch_counts", {}).get("launch", 0) for group in groups)
    returned = sum(
        group.get("launch_counts", {}).get("returned", 0) for group in groups
    )
    return {
        "manifest_digest": results["manifest_digest"],
        "shard": results["shard"],
        "planned_group_count": results.get("planned_group_count", len(groups)),
        "planned_case_count": results.get("planned_case_count", 0),
        "group_count": len(groups),
        "complete_group_count": sum(group["complete"] for group in groups),
        "group_status_counts": dict(Counter(group["status"] for group in groups)),
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
        "-m",
        MARKERS,
        "-o",
        "addopts=--order-scope=class",
        "-o",
        "cache_dir=" + str(args.output / "pytest-cache"),
    ]


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
    threads = int(env.get("TTLANG_EMULE_CI_THREADS", "4"))
    if not 1 <= threads <= 4:
        raise ValueError("emulator CI thread count must be between one and four")
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
    events_path = args.output / "collection.events.jsonl"
    env["TTLANG_EMULE_CI_EVENTS"] = str(events_path)
    command = pytest_command(args) + [
        "--collect-only",
        str(args.source / "test/python"),
        str(args.source / "test/me2e"),
    ]
    pytest_result = run_command(
        command, args.output / "pytest.log", 600, env, args.source
    )
    lit_result = run_command(
        [args.lit, "--show-tests", str(args.build / "test/python")],
        args.output / "lit.log",
        600,
        env,
        args.source,
    )
    write_json(
        args.output / "collection.json", {"pytest": pytest_result, "lit": lit_result}
    )
    if any(result["exit_code"] != 0 for result in (pytest_result, lit_result)):
        raise ValueError("collection failed; inspect collection.json and logs")
    collections = [
        event["nodes"]
        for event in read_events(events_path)
        if event["kind"] == "collection"
    ]
    if len(collections) != 1:
        raise ValueError("pytest collection did not report its complete inventory")
    manifest = make_manifest(
        collections[0],
        (args.output / "lit.log").read_text(),
        args.revision,
        args.runtime_id,
    )
    write_json(args.output / "manifest.json", manifest)


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
    summary = {
        "manifest_digest": digest(manifest),
        "shard": args.shard,
        "planned_group_count": len(groups),
        "planned_case_count": sum(len(group["cases"]) for group in groups),
        "groups": [],
    }
    write_json(args.output / "results.json", summary)
    write_json(args.output / "summary.json", public_summary(summary))
    for index, group in enumerate(groups):
        prefix = args.output / f"group-{index:04d}"
        events_path = prefix.with_suffix(".events.jsonl")
        env["TTLANG_EMULE_CI_EVENTS"] = str(events_path)
        if group["kind"] == "pytest":
            command = pytest_command(args) + [
                "--timeout=" + str(args.timeout),
                "--timeout-method=thread",
                "--reruns=0",
                "--junitxml=" + str(prefix.with_suffix(".xml")),
                str(args.source / "test" / group["path"]),
            ]
        else:
            command = [
                args.lit,
                "-j1",
                "--timeout=" + str(args.timeout),
                "-v",
                "-o",
                str(prefix.with_suffix(".lit.json")),
                str(args.build / "test" / group["path"]),
            ]
        result = run_command(
            command, prefix.with_suffix(".log"), args.group_timeout, env, args.source
        )
        result.update(id=group["id"], complete=False)
        try:
            if group["kind"] == "pytest":
                events = read_events(events_path)
                result["complete"] = pytest_coverage(group, events)
                result["launch_counts"] = dict(
                    Counter(
                        event["kind"]
                        for event in events
                        if event["kind"] in ("launch", "returned")
                    )
                )
                result["outcomes"] = pytest_outcomes(events)
            else:
                tests = read_json(prefix.with_suffix(".lit.json"))["tests"]
                result["complete"] = [test["name"] for test in tests] == group[
                    "cases"
                ] and all(
                    test["code"] not in ("UNRESOLVED", "TIMEOUT") for test in tests
                )
                result["outcomes"] = dict(Counter(test["code"] for test in tests))
        except (OSError, ValueError, KeyError) as error:
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
                    for key in ("id", "status", "complete", "wall_seconds")
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
    command.add_argument("--public-manifest", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "balance":
            if args.output.exists() or (
                args.public_manifest and args.public_manifest.exists()
            ):
                raise ValueError("choose a new output path")
            if (
                args.public_manifest
                and args.public_manifest.resolve() == args.output.resolve()
            ):
                raise ValueError(
                    "plan and public manifest require separate output paths"
                )
            manifest = read_json(args.manifest)
            plan = balance(manifest, read_json(args.results), args.shards)
            sanitized = public_manifest(manifest)
            write_json(
                args.output,
                plan,
            )
            if args.public_manifest:
                write_json(args.public_manifest, sanitized)
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
