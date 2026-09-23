# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.mlir %python %s > %t.cpp 2>&1
# RUN: FileCheck %s --check-prefix=IR < %t.mlir
# RUN: FileCheck %s --check-prefix=CPP < %t.cpp

"""Verify the external-add example lowers its DFB arguments and C++ call."""

import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from compiler_only_external_call import compiler_only_external_call


# IR-LABEL: func.func @compute
# IR-DAG: %[[A:.*]] = ttl.bind_cb{cb_index = 0
# IR-DAG: %[[B:.*]] = ttl.bind_cb{cb_index = 1
# IR-DAG: %[[OUT:.*]] = ttl.bind_cb{cb_index = 2
# IR: ttl.opaque_call "ckernel::compiler_only_add" (%[[A]], %[[B]], %[[OUT]])

# CPP: === compute kernel written to {{.*}} ===
# CPP: #include "{{.*}}compiler_only_add.hpp"
# CPP: ckernel::compiler_only_add(get_compile_time_arg_val(0), get_compile_time_arg_val(1), get_compile_time_arg_val(2));

if __name__ == "__main__":
    tensors = [
        ttnn.from_torch(
            torch.zeros((64, 64), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        for _ in range(3)
    ]
    compiler_only_external_call(*tensors)
