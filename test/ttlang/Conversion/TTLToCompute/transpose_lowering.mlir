// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute),cse,canonicalize)' --split-input-file | FileCheck %s

// Summary: Tests for ttl.transpose lowering to ttl.compute with tile_transpose.
// Verifies swapped input indexing map, parallel iterators, and body data flow.

// Transpose 2x3 -> 3x2: input map swaps dims.
// CHECK: #[[$MAP_SWAP:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK: #[[$MAP_ID:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @transpose_2x3
func.func @transpose_2x3() attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[3, 2], !ttcore.tile<32x32, bf16>, 2>
  %inp = ttl.cb_wait %cb0 : <[2, 3], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  // CHECK: %[[INP_CB:.*]] = ttl.attach_cb
  %inp_cb = ttl.attach_cb %inp, %cb0 : (tensor<2x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  // CHECK: %[[RESERVE:.*]] = ttl.cb_reserve
  %reserve = ttl.cb_reserve %cb1 : <[3, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<3x2x!ttcore.tile<32x32, bf16>>

  // Verify compute op: 1 input with swapped map, 1 output with identity.
  // CHECK: %[[OUT_CB:.*]] = ttl.attach_cb
  // CHECK: ttl.compute
  // CHECK-SAME: ins(%[[INP_CB]] :
  // CHECK-SAME: outs(%[[OUT_CB]] :
  // CHECK-SAME: indexing_maps = [#[[$MAP_SWAP]], #[[$MAP_ID]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel"]
  //
  // Verify body: tile_transpose consumes input block arg, produces result
  // that flows to tile_store with iter_index coordinates.
  // CHECK: ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, bf16>, %[[OUT:.*]]: !ttcore.tile<32x32, bf16>):
  // CHECK-NEXT: %[[I:.*]] = ttl.iter_index 0
  // CHECK-NEXT: %[[J:.*]] = ttl.iter_index 1
  // CHECK-NEXT: %[[TR:.*]] = ttl.tile_transpose %[[IN]], %[[OUT]]
  // CHECK-NEXT: ttl.tile_store %[[TR]], %[[RESERVE]][%[[I]], %[[J]]]
  // CHECK-NEXT: ttl.yield
  %result = ttl.transpose %inp_cb : tensor<2x3x!ttcore.tile<32x32, bf16>> -> tensor<3x2x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %reserve : tensor<3x2x!ttcore.tile<32x32, bf16>>, tensor<3x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb1 : <[3, 2], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : <[2, 3], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Transpose 1x1 (trivial): swapped map on 1x1 is still identity behavior.
// CHECK-LABEL: func.func @transpose_1x1
func.func @transpose_1x1() attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %inp = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %inp_cb = ttl.attach_cb %inp, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  // CHECK: ttl.compute
  // CHECK-SAME: iterator_types = ["parallel", "parallel"]
  // CHECK: ttl.tile_transpose
  // CHECK: ttl.yield
  %result = ttl.transpose %inp_cb : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}
