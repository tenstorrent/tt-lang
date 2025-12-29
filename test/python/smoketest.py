# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: env TTLANG_INITIAL_MLIR=%t.initial.mlir TTLANG_FINAL_MLIR=%t.final.mlir %python %s | FileCheck %s

from ttmlir.ir import *
from ttmlir.dialects import ttcore, ttir

with Context() as ctx:

    module = Module.parse(
        """
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = ttir.empty() : tensor<64x128xf32>
    %2 = "ttir.multiply"(%0, %1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    """
    )
    # CHECK: = ttir.empty() : tensor<64x128xf32>
    # CHECK: = "ttir.multiply"
    # CHECK-SAME: (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    print(str(module))
