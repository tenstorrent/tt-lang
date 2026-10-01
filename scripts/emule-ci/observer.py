# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Record pytest coverage and compiler-generated runtime launches."""

import functools
import json
import os

current_node = None


def emit(kind, **fields):
    with open(os.environ["TTLANG_EMULE_CI_EVENTS"], "a", encoding="utf-8") as output:
        output.write(json.dumps({"kind": kind, **fields}) + "\n")


def pytest_collection_finish(session):
    emit("collection", nodes=[item.nodeid for item in session.items])


def pytest_sessionstart(session):
    if session.config.option.collectonly:
        return
    import ttnn

    original = ttnn.generic_op

    @functools.wraps(original)
    def observe(*args, **kwargs):
        emit("launch", nodeid=current_node)
        result = original(*args, **kwargs)
        emit("returned", nodeid=current_node)
        return result

    ttnn.generic_op = observe


def pytest_runtest_logstart(nodeid, location):
    global current_node
    current_node = nodeid


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
    global current_node
    current_node = None
