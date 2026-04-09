// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute))' --split-input-file | FileCheck %s

// Matmul+add fold: add is eliminated, producing 3-operand tile_matmul_block.
// Post-matmul unary: applied in-place in the same fused compute body.

// CHECK-DAG: #[[$LHS:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$RHS:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$PAR:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func.func @matmul_add
// CHECK:         %[[A:.*]] = ttl.attach_cb
// CHECK:         %[[B:.*]] = ttl.attach_cb
// CHECK:         %[[C:.*]] = ttl.attach_cb
// CHECK:         ttl.compute ins(%[[A]], %[[B]], %[[C]] :
// CHECK-SAME:      indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$PAR]], #[[$PAR]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-NEXT:    ^bb0(%[[AT:.*]]: !ttcore.tile{{.*}}, %[[BT:.*]]: !ttcore.tile{{.*}}, %[[CT:.*]]: !ttcore.tile{{.*}}, %[[OUT:.*]]: !ttcore.tile{{.*}}):
// CHECK:           %[[MM:.*]] = ttl.tile_matmul_block %[[AT]], %[[BT]], %[[CT]]
// CHECK-NOT:       ttl.tile_add
// CHECK:      ttl.tile_store %[[MM]],{{.*}} from dst[%c-1]
// CHECK-NEXT:      ttl.yield
func.func @matmul_add() attributes {ttl.base_cta_index = 4 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w0 = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a = ttl.attach_cb %w0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %w1 = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %w1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %w2 = ttl.cb_wait %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c = ttl.attach_cb %w2, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sum = ttl.add %mm, %c : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Commuted add: c + matmul(a,b) produces the same 3-operand fold.
// The accumulator (c) traces first, so it appears as the first block arg.

// CHECK-DAG: #[[$C_PAR:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG: #[[$C_LHS:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$C_RHS:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>

// CHECK-LABEL: func.func @matmul_add_commuted
// CHECK:         ttl.compute
// CHECK-SAME:      indexing_maps = [#[[$C_PAR]], #[[$C_LHS]], #[[$C_RHS]], #[[$C_PAR]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-NEXT:    ^bb0(%[[CT:.*]]: !ttcore.tile{{.*}}, %[[AT:.*]]: !ttcore.tile{{.*}}, %[[BT:.*]]: !ttcore.tile{{.*}}, %{{.*}}: !ttcore.tile{{.*}}):
// CHECK-NEXT:      ttl.iter_index 0
// CHECK-NEXT:      ttl.iter_index 1
// CHECK:      %[[MM:.*]] = ttl.tile_matmul_block %[[AT]], %[[BT]], %[[CT]]
// CHECK-NOT:       ttl.tile_add
// CHECK:      ttl.tile_store %[[MM]],{{.*}}from dst[%c-1]
func.func @matmul_add_commuted() attributes {ttl.base_cta_index = 4 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w0 = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a = ttl.attach_cb %w0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %w1 = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %w1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %w2 = ttl.cb_wait %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c = ttl.attach_cb %w2, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sum = ttl.add %c, %mm : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Post-matmul unary: relu(a @ b). 2-operand matmul (no accumulator) + tile_relu.

// CHECK-LABEL: func.func @matmul_relu
// CHECK:         ttl.compute
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-NEXT:    ^bb0(%[[AT:.*]]: !ttcore.tile{{.*}}, %[[BT:.*]]: !ttcore.tile{{.*}}, %{{.*}}: !ttcore.tile{{.*}}):
// CHECK-NEXT:      ttl.iter_index 0
// CHECK-NEXT:      ttl.iter_index 1
// CHECK:      %[[MM:.*]] = ttl.tile_matmul_block %[[AT]], %[[BT]]{{.*}}into dst[%c-1] {ttl.dst_placeholder} :
// CHECK:      %[[R:.*]] = ttl.tile_relu %[[MM]]{{.*}}into dst[%c-1] {ttl.dst_placeholder}
// CHECK:      ttl.tile_store %[[R]],{{.*}}from dst[%c-1]
// CHECK-NEXT:      ttl.yield
func.func @matmul_relu() attributes {ttl.base_cta_index = 4 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w0 = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a = ttl.attach_cb %w0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %w1 = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %w1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %activated = ttl.relu %mm : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %activated, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Combined: relu(matmul(a,b) + c). Add folded into 3-operand matmul, relu in-place.

// CHECK-LABEL: func.func @matmul_add_relu
// CHECK:         ttl.compute
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-NEXT:    ^bb0(%[[AT:.*]]: !ttcore.tile{{.*}}, %[[BT:.*]]: !ttcore.tile{{.*}}, %[[CT:.*]]: !ttcore.tile{{.*}}, %{{.*}}: !ttcore.tile{{.*}}):
// CHECK-NEXT:      ttl.iter_index 0
// CHECK-NEXT:      ttl.iter_index 1
// CHECK:      %[[MM:.*]] = ttl.tile_matmul_block %[[AT]], %[[BT]], %[[CT]]
// CHECK-NOT:       ttl.tile_add
// CHECK:      %[[R:.*]] = ttl.tile_relu %[[MM]]{{.*}}into dst[%c-1] {ttl.dst_placeholder}
// CHECK:      ttl.tile_store %[[R]],{{.*}}from dst[%c-1]
// CHECK-NEXT:      ttl.yield
func.func @matmul_add_relu() attributes {ttl.base_cta_index = 4 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w0 = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a = ttl.attach_cb %w0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %w1 = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %w1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %w2 = ttl.cb_wait %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c = ttl.attach_cb %w2, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sum = ttl.add %mm, %c : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %activated = ttl.relu %sum : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %activated, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Standalone matmul (no fusion): 2-operand tile_matmul_block, no accumulator.

// CHECK-LABEL: func.func @matmul_standalone
// CHECK:         ttl.compute
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-NEXT:    ^bb0(%[[AT:.*]]: !ttcore.tile{{.*}}, %[[BT:.*]]: !ttcore.tile{{.*}}, %{{.*}}: !ttcore.tile{{.*}}):
// CHECK-NEXT:      ttl.iter_index 0
// CHECK-NEXT:      ttl.iter_index 1
// CHECK:      %[[MM:.*]] = ttl.tile_matmul_block %[[AT]], %[[BT]]{{.*}}into dst[%c-1] {ttl.dst_placeholder} :
// CHECK:      ttl.tile_store %[[MM]],{{.*}}from dst[%c-1]
// CHECK-NEXT:      ttl.yield
func.func @matmul_standalone() attributes {ttl.base_cta_index = 4 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w0 = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a = ttl.attach_cb %w0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %w1 = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %w1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %mm, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Multi-tile matmul+add with K-sliced inputs (broadcast-compatible).
// LHS [2,1] and RHS [1,2] are broadcast-compatible with [2,2] output.
// This is the standard pattern from Python K-accumulation loops.

// CHECK-LABEL: func.func @matmul_add_broadcast_compatible
// CHECK:         ttl.compute
// CHECK-SAME:      indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$PAR]], #[[$PAR]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
// CHECK:           ttl.tile_matmul_block
// CHECK-NOT:       ttl.tile_add
func.func @matmul_add_broadcast_compatible() attributes {ttl.base_cta_index = 4 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %w0 = ttl.cb_wait %cb0 : <[2, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %a = ttl.attach_cb %w0, %cb0 : (tensor<2x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %w1 = ttl.cb_wait %cb1 : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %w1, %cb1 : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %w2 = ttl.cb_wait %cb2 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %c = ttl.attach_cb %w2, %cb2 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb3 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b : tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %sum = ttl.add %mm, %c : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %reserve : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb3 : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : <[2, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb1 : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb2 : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Matmul with K > 1 inputs staged separately from add: LHS [2,4] x RHS [4,2]
// has K=4, so matmul and add are written as separate compute regions with an
// intermediate CB. Matmul gets a 3D [M,N,K] iteration space with reduction,
// add gets a 2D parallel compute.

// CHECK-LABEL: func.func @matmul_add_incompatible_shapes
// CHECK:         ttl.compute
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
// CHECK:           ttl.tile_matmul_block
// CHECK:         ttl.compute
// CHECK-SAME:      iterator_types = ["parallel", "parallel"]
// CHECK:           ttl.tile_add
func.func @matmul_add_incompatible_shapes() attributes {ttl.base_cta_index = 4 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb_mm = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb_out = ttl.bind_cb {cb_index = 4, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  // Matmul inputs
  %w0 = ttl.cb_wait %cb0 : <[2, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  %a = ttl.attach_cb %w0, %cb0 : (tensor<2x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  %w1 = ttl.cb_wait %cb1 : <[4, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x2x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %w1, %cb1 : (tensor<4x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x2x!ttcore.tile<32x32, bf16>>
  // Matmul -> intermediate CB
  %mm_reserve = ttl.cb_reserve %cb_mm : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b : tensor<2x4x!ttcore.tile<32x32, bf16>>, tensor<4x2x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.store %mm, %mm_reserve : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb_mm : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
  // Add reads from intermediate CB + bias CB
  %w_mm = ttl.cb_wait %cb_mm : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %mm_val = ttl.attach_cb %w_mm, %cb_mm : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %w2 = ttl.cb_wait %cb2 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %c = ttl.attach_cb %w2, %cb2 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %out_reserve = ttl.cb_reserve %cb_out : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %sum = ttl.add %mm_val, %c : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %out_reserve : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb_out : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : <[2, 4], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb1 : <[4, 2], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb_mm : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb2 : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Matmul+sub: the fold only applies to add. Sub produces a 2-operand matmul
// followed by an explicit tile_sub.

// CHECK-LABEL: func.func @matmul_sub_no_fold
// CHECK:         ttl.compute
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-NEXT:    ^bb0(%[[AT:.*]]: !ttcore.tile{{.*}}, %[[BT:.*]]: !ttcore.tile{{.*}}, %[[CT:.*]]: !ttcore.tile{{.*}}, %{{.*}}: !ttcore.tile{{.*}}):
// CHECK-NEXT:      ttl.iter_index 0
// CHECK-NEXT:      ttl.iter_index 1
// CHECK:      %[[MM:.*]] = ttl.tile_matmul_block %[[AT]], %[[BT]]{{.*}}into dst[%c-1] {ttl.dst_placeholder} :
// CHECK-NOT:       ttl.tile_add
// CHECK-NEXT:      %[[S:.*]] = ttl.tile_sub %[[MM]], %[[CT]]
// CHECK:      ttl.tile_store %[[S]],{{.*}}from dst[%c-1]
// CHECK-NEXT:      ttl.yield
func.func @matmul_sub_no_fold() attributes {ttl.base_cta_index = 4 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w0 = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a = ttl.attach_cb %w0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %w1 = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %w1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %w2 = ttl.cb_wait %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c = ttl.attach_cb %w2, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %diff = ttl.sub %mm, %c : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %diff, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}


// -----

// Regression test: non-square fused matmul+add. A=[2,4], B=[4,3], acc=[2,3].
// With incorrect 2D identity maps, B would be sliced along M during
// subblocking, producing wrong tile indices. The 3D maps ensure B gets
// (d0,d1,d2)->(d2,d1) and is indexed by [K,N], not [M,N].

// CHECK-LABEL: func.func @matmul_add_non_square
// CHECK:         ttl.compute
// CHECK-SAME:      indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$PAR]], #[[$PAR]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
// CHECK:           ttl.tile_matmul_block
// CHECK-NOT:       ttl.tile_add
func.func @matmul_add_non_square() attributes {ttl.base_cta_index = 4 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 3], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 2>
  %w0 = ttl.cb_wait %cb0 : <[2, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  %a = ttl.attach_cb %w0, %cb0 : (tensor<2x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  %w1 = ttl.cb_wait %cb1 : <[4, 3], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x3x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %w1, %cb1 : (tensor<4x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 3], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x3x!ttcore.tile<32x32, bf16>>
  %w2 = ttl.cb_wait %cb2 : <[2, 3], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  %c = ttl.attach_cb %w2, %cb2 : (tensor<2x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb3 : <[2, 3], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b : tensor<2x4x!ttcore.tile<32x32, bf16>>, tensor<4x3x!ttcore.tile<32x32, bf16>> -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  %sum = ttl.add %mm, %c : tensor<2x3x!ttcore.tile<32x32, bf16>>, tensor<2x3x!ttcore.tile<32x32, bf16>> -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %reserve : tensor<2x3x!ttcore.tile<32x32, bf16>>, tensor<2x3x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb3 : <[2, 3], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : <[2, 4], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb1 : <[4, 3], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb2 : <[2, 3], !ttcore.tile<32x32, bf16>, 2>
  func.return
}
