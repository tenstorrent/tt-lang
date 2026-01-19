# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2>&1 | FileCheck %s

"""Test that TTL compiler passes are registered and accessible from Python."""

from ttmlir.ir import Context
from ttmlir.passmanager import PassManager
from ttl.dialects import ttl


def test_ttl_passes_registered():
    """Test that TTL passes can be parsed and executed."""
    ctx = Context()
    ttl.ensure_dialects_registered(ctx)

    # Function-level passes.
    func_passes = [
        "convert-ttl-to-compute",
        "ttl-assign-dst",
        "ttl-insert-tile-regs-sync",
        "ttl-lower-to-loops",
        "ttl-annotate-cb-associations",
    ]

    for pass_name in func_passes:
        PassManager.parse(f"builtin.module(func.func({pass_name}))", context=ctx)
        print(f"{pass_name} pass registered")
        # CHECK: convert-ttl-to-compute pass registered
        # CHECK: ttl-assign-dst pass registered
        # CHECK: ttl-insert-tile-regs-sync pass registered
        # CHECK: ttl-lower-to-loops pass registered
        # CHECK: ttl-annotate-cb-associations pass registered

    # Module-level pass.
    PassManager.parse("builtin.module(convert-ttl-to-ttkernel)", context=ctx)
    print("convert-ttl-to-ttkernel pass registered")
    # CHECK: convert-ttl-to-ttkernel pass registered


if __name__ == "__main__":
    test_ttl_passes_registered()
    print("All TTL passes successfully registered!")
    # CHECK: All TTL passes successfully registered!
