# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.alias.mlir STATIC_BINDING_MODE=alias %python %s > %t.alias.cpp 2>&1
# RUN: FileCheck %s --check-prefix=ALIAS < %t.alias.mlir
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.capture.mlir STATIC_BINDING_MODE=capture %python %s > %t.capture.cpp 2>&1
# RUN: FileCheck %s --check-prefix=CAPTURE < %t.capture.mlir

"""Verify composed ScalarType references use compiler-owned bindings."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn

from Inputs.composed_scalar_result_helpers import (
    caller_without_ttl_binding,
    helper_with_scalar_type_alias,
)


@ttl.operation(grid=(1, 1))
def caller_without_scalar_type_alias(inp):
    helper_with_scalar_type_alias()


# ALIAS: ttl.opaque_call "alias_result" () {{.*}} : () -> i64
# ALIAS: ttl.opaque_call "alias_consume" ({{.*}}) {{.*}} : (i64) -> ()
# CAPTURE: ttl.opaque_call "captured_result" () {{.*}} : () -> i64
# CAPTURE: ttl.opaque_call "captured_consume" ({{.*}}) {{.*}} : (i64) -> ()


if __name__ == "__main__":
    host_input = torch.zeros((32, 32), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        host_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    operation = {
        "alias": caller_without_scalar_type_alias,
        "capture": caller_without_ttl_binding,
    }[os.environ["STATIC_BINDING_MODE"]]
    operation(input_tensor)
