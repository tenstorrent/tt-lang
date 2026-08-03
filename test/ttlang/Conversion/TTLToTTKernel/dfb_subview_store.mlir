// Verifies that tile stores through DFB subviews fold the subview offset into
// the TTKernel DFB tile index.

// RUN: ttlang-opt %s --convert-ttl-to-ttkernel --canonicalize -cse | FileCheck %s

// CHECK-LABEL: func.func @store_to_dfb_subview
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: ttkernel.pack_tile(%{{.*}}, %[[CB]], %[[C2]]
func.func @store_to_dfb_subview(%tile: !ttcore.tile<32x32, bf16>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
  %view = ttl.cb_reserve %cb
      : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
      -> tensor<1x4x!ttcore.tile<32x32, bf16>>
  %subview = tensor.extract_slice %view[0, 2] [1, 1] [1, 1]
      : tensor<1x4x!ttcore.tile<32x32, bf16>>
      to tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  ttl.tile_store %tile, %subview[%c0, %c0] from dst[%c0]
      : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}
