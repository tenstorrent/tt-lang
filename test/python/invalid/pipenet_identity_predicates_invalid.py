# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_CASE=chained not %python %s 2>&1 | FileCheck %s --check-prefix=CHAINED
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_CASE=property not %python %s 2>&1 | FileCheck %s --check-prefix=PROPERTY
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_CASE=types not %python %s 2>&1 | FileCheck %s --check-prefix=TYPES

"""Invalid graph callback identity predicates report source diagnostics."""

import os

import pytest
import ttl

pytest.importorskip("ttnn", exc_type=ImportError)

DEVICE_DOMAIN = ttl.DeviceDomain((1, 2))
EXCHANGE_NET = ttl.PipeNet(graph=ttl.TransferGraph.all_to_all(DEVICE_DOMAIN))


@ttl.operation(grid=(1, 1), device_domain=DEVICE_DOMAIN)
def chained_identity_predicate():
    @ttl.datamovement()
    def data_movement():
        def callback(pipe):
            if 0 < pipe.source_device_index < 2:
                pass

        EXCHANGE_NET.if_dst(callback)


@ttl.operation(grid=(1, 1), device_domain=DEVICE_DOMAIN)
def unknown_identity_property():
    @ttl.datamovement()
    def data_movement():
        def callback(pipe):
            if pipe.unknown_device_index == 0:
                pass

        EXCHANGE_NET.if_dst(callback)


@ttl.operation(grid=(1, 1), device_domain=DEVICE_DOMAIN)
def incompatible_identity_types():
    @ttl.datamovement()
    def data_movement():
        def callback(pipe):
            if pipe.source_device_index < "zero":
                pass

        EXCHANGE_NET.if_dst(callback)


if __name__ == "__main__":
    test_case = os.environ["TTLANG_CASE"]
    if test_case == "chained":
        chained_identity_predicate()
    elif test_case == "property":
        unknown_identity_property()
    else:
        incompatible_identity_types()


# CHAINED: error: chained comparisons are not supported
# CHAINED: if 0 < pipe.source_device_index < 2:
# PROPERTY: error: pipe callback identity has no property 'unknown_device_index'
# PROPERTY: if pipe.unknown_device_index == 0:
# TYPES: error: comparison operands must be compiler values, got OpResult and str
# TYPES: if pipe.source_device_index < "zero":
