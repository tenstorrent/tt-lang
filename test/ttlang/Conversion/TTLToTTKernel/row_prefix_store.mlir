// Summary: Verifies BF16 and FP32 row-prefix stores use normal tile packing.

// RUN: ttlang-opt %s --convert-ttl-to-ttkernel --canonicalize -cse | FileCheck %s

// A BF16 row-prefix store uses the destination's short-height tile format.
// CHECK-LABEL: func.func @row_prefix_bf16
// CHECK: %[[BF16_CB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-NEXT: ttkernel.cb_reserve_back(%[[BF16_CB]], %{{.*}})
// CHECK-NEXT: ttkernel.pack_tile(%{{.*}}, %[[BF16_CB]], %{{.*}}, true)
func.func @row_prefix_bf16(%tile: !ttcore.tile<32x32, bf16>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, bf16>, 1>
  %view = ttl.cb_reserve %cb
      : <[1, 1], !ttcore.tile<16x32, bf16>, 1>
      -> tensor<1x1x!ttcore.tile<16x32, bf16>>
  %c0 = arith.constant 0 : index
  ttl.tile_store %tile, %view[%c0, %c0] from dst[%c0] {row_prefix}
      : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<16x32, bf16>>
  func.return
}

// An FP32 row-prefix store uses the FP32 short-height destination format.
// CHECK-LABEL: func.func @row_prefix_f32
// CHECK: %[[F32_CB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK-NEXT: ttkernel.cb_reserve_back(%[[F32_CB]], %{{.*}})
// CHECK-NEXT: ttkernel.pack_tile(%{{.*}}, %[[F32_CB]], %{{.*}}, true)
func.func @row_prefix_f32(%tile: !ttcore.tile<32x32, f32>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, f32>, 1>
  %view = ttl.cb_reserve %cb
      : <[1, 1], !ttcore.tile<16x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<16x32, f32>>
  %c0 = arith.constant 0 : index
  ttl.tile_store %tile, %view[%c0, %c0] from dst[%c0] {row_prefix}
      : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<16x32, f32>>
  func.return
}
