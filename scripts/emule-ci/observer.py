# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Record pytest coverage and compiler-generated runtime launches."""

import functools
import json
import os

import pytest


def emit(kind, **fields):
    with open(os.environ["TTLANG_EMULE_CI_EVENTS"], "a", encoding="utf-8") as output:
        output.write(json.dumps({"kind": kind, **fields}) + "\n")


def pytest_collection_modifyitems(items):
    # The benchmark validates the emulator-only xfail list. Its marks come
    # first so they override a test's own xfail marks; non-strict, a listed
    # case that passes is reported passed.
    path = os.environ.get("TTLANG_EMULE_CI_XFAIL")
    if not path:
        return
    with open(path, encoding="utf-8") as listing:
        listed = json.load(listing)["tests"]
    for item in items:
        if item.nodeid in listed:
            entry = listed[item.nodeid]
            mark = pytest.mark.xfail(
                reason=entry["reason"], run=entry["run"], strict=False
            )
            item.add_marker(mark, append=False)


def pytest_collection_finish(session):
    # Record which cases declare themselves multi-device so the benchmark can
    # report single-device-profile skips as unsupported rather than silently
    # excluding or miscounting them.
    markers = {
        item.nodeid: ["multi_device"]
        for item in session.items
        if item.get_closest_marker("multi_device")
    }
    emit("collection", nodes=[item.nodeid for item in session.items], markers=markers)


def pytest_sessionstart(session):
    # Only the device suites ask for runtime observation; collection and the
    # host-only packaging suite never import the emulator runtime.
    if (
        session.config.option.collectonly
        or os.environ.get("TTLANG_EMULE_CI_OBSERVE_RUNTIME") != "1"
    ):
        return
    import ttnn

    original = ttnn.generic_op

    @functools.wraps(original)
    def observe(*args, **kwargs):
        emit("launch")
        result = original(*args, **kwargs)
        emit("returned")
        return result

    ttnn.generic_op = observe


def pytest_runtest_logreport(report):
    emit(
        "report",
        nodeid=report.nodeid,
        phase=report.when,
        outcome=report.outcome,
        seconds=report.duration,
        expected_failure=hasattr(report, "wasxfail"),
    )


def pytest_runtest_logfinish(nodeid, location):
    emit("finish", nodeid=nodeid)
