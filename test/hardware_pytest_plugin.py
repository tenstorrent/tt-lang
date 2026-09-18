# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest contracts for hardware and simulator suites."""

from __future__ import annotations

from typing import Any

import pytest

_worker_crashes: list[str] = []


def pytest_configure(config: pytest.Config) -> None:
    _worker_crashes.clear()
    config.addinivalue_line(
        "markers",
        "compile_only: does not execute on a device; executed serially on "
        "multi-chip hosts",
    )


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: Any, error: object | None) -> None:
    if error is not None:
        _worker_crashes.append(f"{node.gateway.id}: {error}")


def pytest_sessionfinish(session: pytest.Session, exitstatus: pytest.ExitCode) -> None:
    if not _worker_crashes:
        return

    terminal_reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminal_reporter is not None:
        terminal_reporter.write_sep("=", "xdist workers terminated abnormally")
        for worker_crash in _worker_crashes:
            terminal_reporter.write_line(worker_crash)

    # pytest-rerunfailures converts xdist crash reports to reruns. With worker
    # restarts disabled, xdist can then abandon pending tests and exit zero.
    if exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
