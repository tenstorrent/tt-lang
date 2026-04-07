// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute),cse,canonicalize)' --split-input-file | FileCheck %s

// Summary: Tests for ttl.reduce lowering to ttl.compute with tile_reduce.
// Verifies indexing maps, iterator types, body data flow, and reduce_dim
// for each reduction dimension combination (dim 0, dim 1, both).

// Reduce sum along dim 0 (REDUCE_COL): (2,2) input -> (1,2) output.
// Iteration domain is [2, 2] with reduction on dim 0.
// CHECK: #[[$MAP_ID:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP_OUT:.*]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-LABEL: func.func @reduce_sum_dim0
func.func @reduce_sum_dim0() attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %inp = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  // CHECK: %[[INP_CB:.*]] = ttl.attach_cb
  %inp_cb = ttl.attach_cb %inp, %cb0 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // CHECK: %[[SC_CB:.*]] = ttl.attach_cb
  %scaler_cb = ttl.attach_cb %scaler, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // CHECK: %[[RESERVE:.*]] = ttl.cb_reserve
  %reserve = ttl.cb_reserve %cb2 : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>

  // Verify compute op structure: 2 inputs (data + scaler), 1 output.
  // CHECK: %[[OUT_CB:.*]] = ttl.attach_cb
  // CHECK: ttl.compute
  // CHECK-SAME: ins(%[[INP_CB]], %[[SC_CB]] :
  // CHECK-SAME: outs(%[[OUT_CB]] :
  // CHECK-SAME: indexing_maps = [#[[$MAP_ID]], #[[$MAP_OUT]], #[[$MAP_OUT]]]
  // CHECK-SAME: iterator_types = ["reduction", "parallel"]
  //
  // Verify body data flow: block args -> tile_reduce -> tile_store.
  // CHECK: ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, bf16>, %[[SC:.*]]: !ttcore.tile<32x32, bf16>, %[[OUT:.*]]: !ttcore.tile<32x32, bf16>):
  // CHECK-NEXT: %[[ITER:.*]] = ttl.iter_index 1
  // CHECK-NEXT: %[[RED:.*]] = ttl.tile_reduce %[[IN]], %[[SC]], %[[OUT]] 0 : i32 <reduce_dim_col>
  // CHECK-NEXT: ttl.tile_store %[[RED]], %[[RESERVE]][%{{.*}}, %[[ITER]]]
  // CHECK-NEXT: ttl.yield
  %result = ttl.reduce %inp_cb, %scaler_cb 0 : i32 [0] : (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %reserve : tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb2 : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb1 : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Reduce sum along dim 1 (REDUCE_ROW): (2,2) input -> (2,1) output.
// CHECK: #[[$MAP_ID:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP_OUT:.*]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK-LABEL: func.func @reduce_sum_dim1
func.func @reduce_sum_dim1() attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>
  %inp = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %inp_cb = ttl.attach_cb %inp, %cb0 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler_cb = ttl.attach_cb %scaler, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[2, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x1x!ttcore.tile<32x32, bf16>>

  // CHECK: ttl.compute
  // CHECK-SAME: indexing_maps = [#[[$MAP_ID]], #[[$MAP_OUT]], #[[$MAP_OUT]]]
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]
  // CHECK: ttl.tile_reduce {{.*}} 0 : i32 <reduce_dim_row>
  // CHECK: ttl.yield
  %result = ttl.reduce %inp_cb, %scaler_cb 0 : i32 [1] : (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %reserve : tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb2 : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb1 : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Reduce max along both dims (REDUCE_SCALAR): (2,2) input -> (1,1) output.
// CHECK: #[[$MAP_ID:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP_C00:.*]] = affine_map<(d0, d1) -> (0, 0)>
// CHECK-LABEL: func.func @reduce_max_both
func.func @reduce_max_both() attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %inp = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %inp_cb = ttl.attach_cb %inp, %cb0 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler_cb = ttl.attach_cb %scaler, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  // Both dims reduced: output map is (0, 0), both iterators are reduction.
  // reduce_type = 1 (Max), reduce_dim = reduce_dim_scalar.
  // CHECK: ttl.compute
  // CHECK-SAME: indexing_maps = [#[[$MAP_ID]], #[[$MAP_C00]], #[[$MAP_C00]]]
  // CHECK-SAME: iterator_types = ["reduction", "reduction"]
  // CHECK: ttl.tile_reduce {{.*}} 1 : i32 <reduce_dim_scalar>
  // CHECK: ttl.yield
  %result = ttl.reduce %inp_cb, %scaler_cb 1 : i32 [0, 1] : (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb2 : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb1 : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  func.return
}
