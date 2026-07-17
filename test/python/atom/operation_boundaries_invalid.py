# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: not %python %s default 2>&1 | FileCheck %s --check-prefix=DEFAULT
# RUN: not %python %s return 2>&1 | FileCheck %s --check-prefix=RETURN
# RUN: not %python %s non-tensor 2>&1 | FileCheck %s --check-prefix=NON-TENSOR
# RUN: not %python %s expand-only 2>&1 | FileCheck %s --check-prefix=EXPAND-ONLY
# RUN: not %python %s external-dfb 2>&1 | FileCheck %s --check-prefix=EXTERNAL-DFB
# RUN: not %python %s nested-resource 2>&1 | FileCheck %s --check-prefix=NESTED
# RUN: not %python %s shadow 2>&1 | FileCheck %s --check-prefix=SHADOW

"""Public diagnostics for unsupported unified-operation boundaries."""

import sys

import ttl


# Operation parameters cannot define runtime defaults.
def default_parameter():
    # DEFAULT: ValueError: @ttl.operation parameters cannot have default values (parameter 'inp')
    @ttl.operation(grid=(1, 1))
    def invalid(inp=None):
        pass


# Operation bodies cannot return values.
def return_value():
    # RETURN: ValueError: @ttl.operation functions cannot return a value or use return statements
    @ttl.operation(grid=(1, 1))
    def invalid(inp):
        return inp


# Direct runtime arguments must be TT-NN tensors.
def non_tensor_argument():
    @ttl.operation(grid=(1, 1))
    def invalid(inp):
        pass

    # NON-TENSOR: TypeError: @ttl.operation runtime argument 'inp' must be a TT-NN tensor, got int
    invalid(7)


# Resource parameters make an operation expand-only.
def direct_expand_only_call():
    @ttl.operation(grid=(1, 1))
    def invalid(buf: ttl.DFB):
        buf.wait()

    # EXPAND-ONLY: ValueError: @ttl.operation 'invalid' is expand-only because it has DFB or PipeNet parameter(s): 'buf'; it cannot be called directly
    invalid(None)


# DFBs must be declared by the operation that owns them.
def external_dfb_capture():
    external_dfb = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)

    # EXTERNAL-DFB: ValueError: @ttl.operation 'invalid': external DFB 'external_dfb' is not supported
    @ttl.operation(grid=(1, 1))
    def invalid():
        external_dfb.wait()


# Resource declarations must remain at operation top level.
def nested_resource_declaration():
    # NESTED: ValueError: @ttl.operation 'invalid': resource declaration 'make_dfb' must be a simple top-level assignment
    @ttl.operation(grid=(1, 1))
    def invalid():
        if True:
            scratch = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)


# A nested binding cannot capture a call-site argument after expansion.
def nested_binding_shadow():
    @ttl.operation()
    def stage(block):
        def callback(pipe):
            pass

    # SHADOW: ValueError: @ttl.operation: composing 'stage' into 'invalid' would capture or rebind ['pipe']; rename the nested binding
    @ttl.operation(grid=(1, 1))
    def invalid(pipe):
        stage(pipe)


CASES = {
    "default": default_parameter,
    "return": return_value,
    "non-tensor": non_tensor_argument,
    "expand-only": direct_expand_only_call,
    "external-dfb": external_dfb_capture,
    "nested-resource": nested_resource_declaration,
    "shadow": nested_binding_shadow,
}

CASES[sys.argv[1]]()
