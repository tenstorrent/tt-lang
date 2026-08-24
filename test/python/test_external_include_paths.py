# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for external-header include-directory collection."""

from types import SimpleNamespace

import ttl.ttl_api as ttl_api


def test_external_include_paths_are_unique_and_preserve_first_occurrence_order():
    compiled_threads = [
        SimpleNamespace(_opaque_include_paths=["/project/first", "/project/shared"]),
        SimpleNamespace(
            _opaque_include_paths=[
                "/project/shared",
                "/project/second",
                "/project/first",
            ]
        ),
        SimpleNamespace(),
    ]

    assert ttl_api._collect_opaque_include_paths(compiled_threads) == [
        "/project/first",
        "/project/shared",
        "/project/second",
    ]
