# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_CASE=positional not %python %s 2>&1 | FileCheck %s --check-prefix=POSITIONAL
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_CASE=keyword not %python %s 2>&1 | FileCheck %s --check-prefix=KEYWORD

"""Invalid PipeNet destination-count calls report source diagnostics."""

import os

import pytest
import ttl

pytest.importorskip("ttnn", exc_type=ImportError)

GATHER_NET = ttl.PipeNet(
    [
        ttl.Pipe(src=(0, 0), dst=(2, 0)),
        ttl.Pipe(src=(1, 0), dst=(2, 0)),
    ]
)


@ttl.operation(grid=(3, 1))
def positional_argument():
    @ttl.datamovement()
    def data_movement():
        GATHER_NET.destination_count(0)


@ttl.operation(grid=(3, 1))
def keyword_argument():
    @ttl.datamovement()
    def data_movement():
        GATHER_NET.destination_count(unexpected=0)


if __name__ == "__main__":
    if os.environ["TTLANG_CASE"] == "positional":
        positional_argument()
    else:
        keyword_argument()


# POSITIONAL: error: PipeNet.destination_count() takes no arguments
# POSITIONAL: GATHER_NET.destination_count(0)
# KEYWORD: error: PipeNet.destination_count() takes no arguments
# KEYWORD: GATHER_NET.destination_count(unexpected=0)
