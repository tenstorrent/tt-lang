// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute),cse,canonicalize)' | FileCheck %s

// Summary: Tests for ttl.bcast lowering to ttl.compute with tile_bcast.
// Bcast reads from CB and writes to DST, unlike normal elementwise ops.
// BcastType enum values: Col=1, Row=2, Scalar=3

// Row broadcast: (1,N) -> (M,N). Broadcasts first row to all rows.
// CHECK-LABEL: func.func @bcast_row
func.func @bcast_row(%arg0: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: tile_bcast{{.*}}2 : i32
  // CHECK: ttl.tile_store
  // CHECK: ttl.yield
  %reserve = ttl.cb_reserve %cb1 : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.bcast %arg0_cb, %init_cb 2 : i32 : (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %result, %reserve : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Column broadcast: (M,1) -> (M,N). Broadcasts first column to all columns.
// CHECK-LABEL: func.func @bcast_col
func.func @bcast_col(%arg0: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: tile_bcast{{.*}}1 : i32
  // CHECK: ttl.tile_store
  // CHECK: ttl.yield
  %reserve = ttl.cb_reserve %cb1 : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.bcast %arg0_cb, %init_cb 1 : i32 : (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %result, %reserve : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Scalar broadcast: (1,1) -> (M,N). Broadcasts single value to all elements.
// CHECK-LABEL: func.func @bcast_scalar
func.func @bcast_scalar(%arg0: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: tile_bcast{{.*}}3 : i32
  // CHECK: ttl.tile_store
  // CHECK: ttl.yield
  %reserve = ttl.cb_reserve %cb1 : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.bcast %arg0_cb, %init_cb 3 : i32 : (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %result, %reserve : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Scalar reduce -> SCALAR broadcast with correct dims=[0,1].
// Verifies reduce-dim tracing accepts matching bcast type.
// CHECK-LABEL: func.func @bcast_scalar_after_scalar_reduce
func.func @bcast_scalar_after_scalar_reduce() {
  %inp_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %sc_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %red_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %out_cb = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>

  %inp_wait = ttl.cb_wait %inp_cb : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %inp_a = ttl.attach_cb %inp_wait, %inp_cb : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %sc_wait = ttl.cb_wait %sc_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sc_a = ttl.attach_cb %sc_wait, %sc_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %red_res = ttl.cb_reserve %red_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reduced = ttl.reduce %inp_a, %sc_a 0 : i32 [0, 1] : (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %reduced, %red_res : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %red_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %red_wait = ttl.cb_wait %red_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %red_a = ttl.attach_cb %red_wait, %red_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_res = ttl.cb_reserve %out_cb : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %out_init = tensor.empty() : tensor<2x1x!ttcore.tile<32x32, bf16>>
  %out_a = ttl.attach_cb %out_init, %out_cb : (tensor<2x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>

  // BcastType::Scalar = 3
  // CHECK: ttl.compute
  // CHECK: tile_bcast{{.*}}3 : i32
  %bcast = ttl.bcast %red_a, %out_a 3 : i32 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  ttl.store %bcast, %out_res : tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %out_cb : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %red_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %sc_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %inp_cb : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Scalar reduce -> SCALAR broadcast with reduce+store inside a nested region
// (ttl.dst_section). Verifies getInputReduceDim traces across nested blocks.
// CHECK-LABEL: func.func @bcast_scalar_after_scalar_reduce_nested
func.func @bcast_scalar_after_scalar_reduce_nested() {
  %inp_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %sc_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %red_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %out_cb = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>

  %inp_wait = ttl.cb_wait %inp_cb : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %inp_a = ttl.attach_cb %inp_wait, %inp_cb : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %sc_wait = ttl.cb_wait %sc_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sc_a = ttl.attach_cb %sc_wait, %sc_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %red_res = ttl.cb_reserve %red_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.dst_section {
    %reduced = ttl.reduce %inp_a, %sc_a 0 : i32 [0, 1] : (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %reduced, %red_res : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  }
  ttl.cb_push %red_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %red_wait = ttl.cb_wait %red_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %red_a = ttl.attach_cb %red_wait, %red_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_res = ttl.cb_reserve %out_cb : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %out_init = tensor.empty() : tensor<2x1x!ttcore.tile<32x32, bf16>>
  %out_a = ttl.attach_cb %out_init, %out_cb : (tensor<2x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>

  // BcastType::Scalar = 3
  // CHECK: ttl.compute
  // CHECK: tile_bcast{{.*}}3 : i32
  %bcast = ttl.bcast %red_a, %out_a 3 : i32 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  ttl.store %bcast, %out_res : tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %out_cb : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %red_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %sc_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %inp_cb : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  func.return
}
