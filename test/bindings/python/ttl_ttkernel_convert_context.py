# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# RUN: %python %s | FileCheck %s
#
# Verifies a single MLIR Context can load both tt-lang TTL and tt-mlir TTKernel
# dialects and successfully run convert-ttl-to-ttkernel from Python.

from ttmlir.ir import Context, Location, Module
from ttmlir.passmanager import PassManager
from ttlang.dialects import ttl


MLIR_INPUT = r"""
#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module {
  func.func @dma_single_tile_single_copy(%arg0: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %arg0, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}
"""


def main():
    with Context() as ctx, Location.unknown(ctx):
        # ttmlir dialects/passes are registered via site initialization when
        # Context() is created. We only need to register ttlang dialects.
        from ttmlir._mlir_libs import get_dialect_registry
        import ttlang._mlir_libs._ttlang as _ttlang

        registry = get_dialect_registry()
        _ttlang.register_dialects(registry)

        ctx.append_dialect_registry(registry)
        ctx.load_all_available_dialects()

        # Ensure TTL is loaded (attrs/types usable).
        ttl.ensure_dialects_registered(ctx)

        # Ensure required upstream dialects are loaded in the same context.
        _ = ctx.dialects["ttkernel"]
        _ = ctx.dialects["ttnn"]
        _ = ctx.dialects["ttcore"]
        _ = ctx.dialects["func"]
        _ = ctx.dialects["arith"]

        module = Module.parse(MLIR_INPUT, ctx)

        pm = PassManager.parse("builtin.module(convert-ttl-to-ttkernel)", context=ctx)
        pm.enable_verifier(True)
        pm.run(module.operation)

        print(module)


if __name__ == "__main__":
    main()

# CHECK-LABEL: func.func @dma_single_tile_single_copy
# CHECK: ttkernel.get_common_arg_val
# CHECK: ttkernel.TensorAccessorArgs
# CHECK: ttkernel.TensorAccessor
