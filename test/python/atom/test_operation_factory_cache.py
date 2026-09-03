# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests factory compilation-cache propagation through unified operations."""

import ttl
import ttl.atom as atom_module


def test_unified_operation_propagates_factory_cache(monkeypatch):
    """Unified operations forward shared compilation-cache arguments."""
    observed = {}
    factory_cache = {}
    factory_cache_key = ("layer", 7)

    def fake_make_operation_wrapper(*_args, **kwargs):
        observed.update(kwargs)
        return lambda *_args, **_kwargs: None

    monkeypatch.setattr(
        atom_module, "_make_operation_wrapper", fake_make_operation_wrapper
    )

    cached_decorator = ttl.operation(
        grid=(1, 1),
        factory_cache=factory_cache,
        factory_cache_key=factory_cache_key,
    )

    @cached_decorator
    def cached_operation():
        pass

    assert cached_operation is not None
    assert observed["factory_cache"] is factory_cache
    assert observed["factory_cache_key"] == factory_cache_key
