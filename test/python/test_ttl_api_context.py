# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttl import ttl_api


class _ContextSpy:
    def __init__(self):
        self.multithreading = []

    def enable_multithreading(self, enabled):
        self.multithreading.append(enabled)


def test_mlir_context_keeps_parallelism_by_default(monkeypatch):
    monkeypatch.delenv("TTLANG_DISABLE_MLIR_THREADING", raising=False)
    monkeypatch.setattr(ttl_api, "Context", _ContextSpy)

    ctx = ttl_api._make_mlir_context()

    assert ctx.multithreading == []


def test_mlir_context_can_disable_parallel_passes(monkeypatch):
    monkeypatch.setenv("TTLANG_DISABLE_MLIR_THREADING", "1")
    monkeypatch.setattr(ttl_api, "Context", _ContextSpy)

    ctx = ttl_api._make_mlir_context()

    assert ctx.multithreading == [False]
