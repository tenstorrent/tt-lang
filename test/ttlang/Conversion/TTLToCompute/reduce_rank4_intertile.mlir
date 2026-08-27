// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute),cse,canonicalize)' --split-input-file | FileCheck %s
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute),ttl-set-compute-kernel-config{enable-fpu-binary-ops=0 matmul-full-fp32=0 reduce-full-fp32=0},func.func(ttl-assign-dst,ttl-subblock-compute-for-dst{subblock-sync=true},ttl-lower-to-loops{dst-accumulation=true},ttl-schedule-operations,ttl-annotate-cb-associations),convert-ttl-to-ttkernel,ttkernel-insert-inits,canonicalize,cse)' --split-input-file | FileCheck %s --check-prefix=TTKERNEL

// Rank-4 sum over a leading dimension uses an elementwise DST accumulator.
// CHECK: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: affine_map<(d0, d1, d2, d3) -> (0, 0)>
// CHECK: affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
// CHECK-LABEL: func.func @reduce_sum_leading_dim
// TTKERNEL-LABEL: func.func @reduce_sum_leading_dim
// TTKERNEL: ttkernel.fill_tile
// TTKERNEL: scf.for
// TTKERNEL: ttkernel.add_binary_tile
func.func @reduce_sum_leading_dim() attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 3, 4, 5], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 1, 4, 5], !ttcore.tile<32x32, bf16>, 2>
  %inp = ttl.cb_wait %cb0 : <[2, 3, 4, 5], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x3x4x5x!ttcore.tile<32x32, bf16>>
  %inp_cb = ttl.attach_cb %inp, %cb0 : (tensor<2x3x4x5x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 3, 4, 5], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x3x4x5x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler_cb = ttl.attach_cb %scaler, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[2, 1, 4, 5], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x1x4x5x!ttcore.tile<32x32, bf16>>

  // CHECK: ttl.compute
  // CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel", "parallel"]
  // CHECK: ttl.tile_reduction_init 0 : i32
  // CHECK: ttl.tile_mul
  // CHECK: ttl.tile_add_in_place
  // CHECK: ttl.yield
  %result = ttl.reduce %inp_cb, %scaler_cb 0 : i32 [1] : (tensor<2x3x4x5x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<2x1x4x5x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %reserve : tensor<2x1x4x5x!ttcore.tile<32x32, bf16>>, tensor<2x1x4x5x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb2 : !ttl.cb<[2, 1, 4, 5], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb1 : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : !ttl.cb<[2, 3, 4, 5], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Rank-4 max over a leading dimension uses an in-place maximum accumulator.
// CHECK-LABEL: func.func @reduce_max_leading_dim
// TTKERNEL-LABEL: func.func @reduce_max_leading_dim
// TTKERNEL: ttkernel.fill_tile
// TTKERNEL: scf.for
// TTKERNEL: ttkernel.binary_max_tile
func.func @reduce_max_leading_dim() attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 3, 4, 5], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 1, 4, 5], !ttcore.tile<32x32, bf16>, 2>
  %inp = ttl.cb_wait %cb0 : <[2, 3, 4, 5], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x3x4x5x!ttcore.tile<32x32, bf16>>
  %inp_cb = ttl.attach_cb %inp, %cb0 : (tensor<2x3x4x5x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 3, 4, 5], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x3x4x5x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler_cb = ttl.attach_cb %scaler, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[2, 1, 4, 5], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x1x4x5x!ttcore.tile<32x32, bf16>>

  // CHECK: ttl.compute
  // CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel", "parallel"]
  // CHECK: ttl.tile_reduction_init 1 : i32
  // CHECK: ttl.tile_mul
  // CHECK: ttl.tile_max
  // CHECK: ttl.yield
  %result = ttl.reduce %inp_cb, %scaler_cb 1 : i32 [1] : (tensor<2x3x4x5x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<2x1x4x5x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %reserve : tensor<2x1x4x5x!ttcore.tile<32x32, bf16>>, tensor<2x1x4x5x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb2 : !ttl.cb<[2, 1, 4, 5], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb1 : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : !ttl.cb<[2, 3, 4, 5], !ttcore.tile<32x32, bf16>, 2>
  func.return
}
