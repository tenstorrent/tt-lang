# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=INITIAL < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CPP < %t.output

"""Verify canonical-kernel DFB reconfiguration emission and lowering."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn
from ttl import ttl_api


def _blackhole_compile_target(_runtime_args):
    return "blackhole"


ttl_api._device_target_arch = _blackhole_compile_target


def make_reconfiguration_operation():
    boundary = ttl.DFBReconfiguration(
        participants=(
            ttl.KernelKind.COMPUTE,
            ttl.KernelKind.DATA_MOVEMENT,
            ttl.PIPE_SOURCE_KERNEL,
        )
    )

    @ttl.operation(grid=(1, 1))
    def canonical_reconfiguration_operation(input_tensor):
        ttl.reconfigure_dfbs(boundary)

    return canonical_reconfiguration_operation


canonical_reconfiguration_operation = make_reconfiguration_operation()


# INITIAL-LABEL: func.func @canonical_reconfiguration_operation__trisc
# INITIAL-SAME: ttl.logical_kernel = #ttl.logical_kernel<kind = compute>
# INITIAL: ttl.dfb_reconfiguration <{{[0-9]+}}, participants[<kind = compute>, <kind = data_movement>, <kind = data_movement, identity = "<pipe_source>", role = "pipe_source">]>
# INITIAL-LABEL: func.func @canonical_reconfiguration_operation__ncrisc
# INITIAL-SAME: ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>
# INITIAL: ttl.dfb_reconfiguration <{{[0-9]+}}, participants[<kind = compute>, <kind = data_movement>, <kind = data_movement, identity = "<pipe_source>", role = "pipe_source">]>
# INITIAL-LABEL: func.func @canonical_reconfiguration_operation__brisc
# INITIAL-SAME: ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "<pipe_source>", role = "pipe_source">
# INITIAL: ttl.dfb_reconfiguration <{{[0-9]+}}, participants[<kind = compute>, <kind = data_movement>, <kind = data_movement, identity = "<pipe_source>", role = "pipe_source">]>

# CPP-COUNT-3: experimental::reconfigure_dfb_interfaces(


if __name__ == "__main__":
    input_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    canonical_reconfiguration_operation(input_tensor)
