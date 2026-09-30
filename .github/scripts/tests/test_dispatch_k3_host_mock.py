# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verify trust restrictions, request correlation, and exact-SHA status publication."""

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "dispatch-k3-host-mock.py"
SPEC = importlib.util.spec_from_file_location("dispatch_host_mock", SCRIPT)
dispatch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dispatch)
COMPILER_SHA = "a" * 40
CORRELATION_ID = "unit-test-request"


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.environment = {
            "GITHUB_REPOSITORY": "tenstorrent/tt-lang",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_SHA": "b" * 40,
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_OUTPUT": str(Path(self.directory.name) / "output"),
            "GITHUB_STEP_SUMMARY": str(Path(self.directory.name) / "summary"),
            "K3_COMPILER_SHA": COMPILER_SHA,
            "K3_CORRELATION_ID": CORRELATION_ID,
            "GH_TOKEN": "public-token",
            "K3_DISPATCH_TOKEN": "private-token",
        }
        environment_patch = patch.dict(os.environ, self.environment, clear=True)
        environment_patch.start()
        self.addCleanup(environment_patch.stop)

    def test_reject_untrusted_events_without_api_calls(self):
        for setting, value in (
            ("GITHUB_REF", "refs/heads/feature"),
            ("GITHUB_EVENT_NAME", "pull_request"),
            ("GITHUB_EVENT_NAME", "pull_request_target"),
            ("GITHUB_REPOSITORY", "fork/tt-lang"),
        ):
            with self.subTest(setting=setting, value=value):
                with patch.dict(os.environ, {setting: value}):
                    with patch.object(dispatch, "request") as request:
                        with self.assertRaises(RuntimeError):
                            dispatch.prepare_request()
                        request.assert_not_called()

    def test_reject_non_main_ancestry(self):
        with patch.object(dispatch, "request", return_value={"status": "diverged"}):
            with self.assertRaises(RuntimeError):
                dispatch.prepare_request()
        self.assertFalse(Path(self.environment["GITHUB_OUTPUT"]).exists())

    def test_pending_status_uses_requested_sha(self):
        with patch.object(
            dispatch, "request", return_value={"status": "ahead"}
        ) as request:
            dispatch.prepare_request()
        self.assertEqual(request.call_args.args[0], "public-token")
        self.assertEqual(
            request.call_args.args[1], f"tenstorrent/tt-lang/statuses/{COMPILER_SHA}"
        )
        self.assertEqual(request.call_args.args[2]["state"], "pending")

    def test_dispatch_passes_exact_sha_and_correlation(self):
        with patch.object(dispatch, "request") as request:
            with patch.object(dispatch, "wait_for_run", return_value=True):
                self.assertTrue(dispatch.dispatch_request())
        self.assertEqual(
            request.call_args.args[2]["inputs"],
            {"compiler_sha": COMPILER_SHA, "correlation_id": CORRELATION_ID},
        )
        self.assertEqual(request.call_args.args[2]["ref"], "main")

    def test_terminal_status_handles_success_failure_and_cancellation(self):
        for outcome in ("success", "failure", "cancelled", "skipped"):
            with self.subTest(outcome=outcome):
                with patch.dict(os.environ, {"K3_RUN_OUTCOME": outcome}):
                    with patch.object(dispatch, "request") as request:
                        captured = io.StringIO()
                        with contextlib.redirect_stdout(captured):
                            dispatch.finish_request()
                state = "success" if outcome == "success" else "failure"
                self.assertEqual(request.call_args.args[2]["state"], state)
                self.assertEqual(
                    json.loads(captured.getvalue())["compiler_sha"], COMPILER_SHA
                )
                self.assertNotIn("private-token", captured.getvalue())

    def test_poll_requires_matching_correlated_run(self):
        run = {
            "id": 456,
            "display_title": f"K3 host mock | {COMPILER_SHA} | {CORRELATION_ID}",
            "head_branch": "main",
            "event": "workflow_dispatch",
            "status": "completed",
            "conclusion": "success",
        }
        for conclusion in ("success", "failure", "cancelled", "timed_out"):
            with self.subTest(conclusion=conclusion):
                response = {**run, "conclusion": conclusion}
                with patch.object(
                    dispatch,
                    "request",
                    side_effect=[{"workflow_runs": [run]}, response],
                ):
                    self.assertEqual(
                        dispatch.wait_for_run(
                            "token", COMPILER_SHA, CORRELATION_ID, float("inf")
                        ),
                        conclusion == "success",
                    )

    def test_missing_or_wrong_sha_run_times_out(self):
        run = {"display_title": f"K3 host mock | {'b' * 40} | {CORRELATION_ID}"}
        with patch.object(dispatch.time, "monotonic", side_effect=[0, 0, 2]):
            with patch.object(dispatch.time, "sleep"):
                with patch.object(
                    dispatch, "request", return_value={"workflow_runs": [run]}
                ):
                    with self.assertRaises(RuntimeError):
                        dispatch.wait_for_run("token", COMPILER_SHA, CORRELATION_ID, 1)


if __name__ == "__main__":
    unittest.main()
