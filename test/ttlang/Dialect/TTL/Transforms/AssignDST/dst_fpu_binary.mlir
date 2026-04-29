// Summary: Tests for FPU binary detection in DST allocation. Binary tile ops
// (add, sub, mul) with BOTH operands as block arguments use the FPU execution
// engine which reads from CB, consuming 0 DST input slots. This doubles the
// achievable unroll_factor for simple binary patterns.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8}), canonicalize, cse)' --split-input-file | FileCheck %s
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8 separate-output-region=1}), canonicalize, cse)' --split-input-file | FileCheck %s --check-prefix=SEPARATE
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8 enable-fpu-binary-ops=0}), canonicalize, cse)' --split-input-file | FileCheck %s --check-prefix=SFPU

// Verify no placeholder copies remain in final IR
// CHECK-NOT: placeholder
// SEPARATE-NOT: placeholder
// SFPU-NOT: placeholder

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 1: Simple FPU binary add
// =============================================================================
// Purpose: tile_add with both block args -> FPU binary, dstPerIteration=1.
// No copy_tile needed. With capacity=8 and 2x2 tensor (4 tiles), unroll_factor=4.

// CHECK-LABEL: func.func @simple_fpu_add
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:           ttl.compute
// CHECK-SAME:      ttl.unroll_factor = 4
// CHECK-NEXT:      ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>, %[[B:[^:]*]]: !ttcore.tile<32x32, f32>, %[[OUT:[^:]*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:        %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:        %[[I1:.*]] = ttl.iter_index 1 : index
// No copy_tile - FPU reads from CB
// CHECK-NOT:       ttl.copy_tile
// CHECK:           %[[ADD:.*]] = ttl.tile_add %[[A]], %[[B]] into dst[%[[C0]]] {ttl.fpu_binary} : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:           ttl.tile_store %[[ADD]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:      ttl.yield
// SEPARATE-LABEL:  func.func @simple_fpu_add
// SEPARATE-DAG:       %[[C0:.*]] = arith.constant 0 : index
// SEPARATE:        %[[ADDS:.*]] = ttl.tile_add {{.*}} into dst[%[[C0]]] {ttl.fpu_binary}
// SEPARATE:        ttl.tile_store
// SEPARATE-NEXT:   ttl.yield
//
// SFPU path: both operands need copy_tile, no fpu_binary attribute
// SFPU-LABEL: func.func @simple_fpu_add
// SFPU-DAG:       %[[C0:.*]] = arith.constant 0 : index
// SFPU:           ttl.compute
// SFPU-SAME:      ttl.unroll_factor = 4
// SFPU-NEXT:      ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>, %[[B:[^:]*]]: !ttcore.tile<32x32, f32>, %[[OUT:[^:]*]]: !ttcore.tile<32x32, f32>):
// SFPU-NEXT:        %[[SI0:.*]] = ttl.iter_index 0 : index
// SFPU-NEXT:        %[[SI1:.*]] = ttl.iter_index 1 : index
// SFPU-NOT:       fpu_binary
// SFPU:           %{{.*}}, %[[ATILE:.*]] = ttl.copy_tile %[[A]][%[[SI0]], %[[SI1]]] into dst[%[[C0]]]
// SFPU:           %{{.*}}, %[[BTILE:.*]] = ttl.copy_tile %[[B]][%[[SI0]], %[[SI1]]] into dst[%c1]
// SFPU:      %[[ADD:.*]] = ttl.tile_add %[[ATILE]], %[[BTILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// SFPU:           ttl.tile_store %[[ADD]], %{{.*}}[%[[SI0]], %[[SI1]]]
// SFPU-NEXT:      ttl.yield

func.func @simple_fpu_add(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                          %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 16, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %add = ttl.tile_add %a_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %add, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 2: Mixed FPU + SFPU chain
// =============================================================================
// Purpose: First add is FPU (both block args), then mul is SFPU (one operand
// from DST). C needs copy_tile for SFPU mul. dstPerIteration=2 (1 FPU output +
// 1 copy for C). unroll_factor=4.

// CHECK-LABEL: func.func @mixed_fpu_sfpu
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:           ttl.compute
// CHECK-SAME:      ttl.unroll_factor = 4
// CHECK-NEXT:      ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>, %[[B:[^:]*]]: !ttcore.tile<32x32, f32>, %[[C:[^:]*]]: !ttcore.tile<32x32, f32>, %[[OUT:[^:]*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:        %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:        %[[I1:.*]] = ttl.iter_index 1 : index
// FPU binary add: both block args, reads from CB
// CHECK:           %[[ADD:.*]] = ttl.tile_add %[[A]], %[[B]] into dst[%[[C0]]] {ttl.fpu_binary} : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// C copied for SFPU mul
// CHECK:           %{{.*}}, %[[CTILE:.*]] = ttl.copy_tile %[[C]][%[[I0]], %[[I1]]] into dst[%c1]
// SFPU mul: one operand from DST (add result), one from copy_tile
// CHECK:      %[[MUL:.*]] = ttl.tile_mul %[[ADD]], %[[CTILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:           ttl.tile_store %[[MUL]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:      ttl.yield
// SEPARATE-LABEL:  func.func @mixed_fpu_sfpu
// SEPARATE-DAG:       %[[C0:.*]] = arith.constant 0 : index
// SEPARATE-DAG:       %[[C2:.*]] = arith.constant 2 : index
// SEPARATE:        %[[ADDS:.*]] = ttl.tile_add {{.*}} into dst[%[[C0]]] {ttl.fpu_binary}
// SEPARATE:        ttl.tile_mul {{.*}} into dst[%[[C2]]]
//
// SFPU path: all binary ops use copy_tile for both operands, no fpu_binary
// SFPU-LABEL: func.func @mixed_fpu_sfpu
// SFPU-DAG:       %[[C0:.*]] = arith.constant 0 : index
// SFPU:           ttl.compute
// SFPU-SAME:      ttl.unroll_factor = 4
// SFPU-NEXT:      ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>, %[[B:[^:]*]]: !ttcore.tile<32x32, f32>, %[[C:[^:]*]]: !ttcore.tile<32x32, f32>, %[[OUT:[^:]*]]: !ttcore.tile<32x32, f32>):
// SFPU-NEXT:        %[[SI0:.*]] = ttl.iter_index 0 : index
// SFPU-NEXT:        %[[SI1:.*]] = ttl.iter_index 1 : index
// SFPU-NOT:       fpu_binary
// copy A and B for SFPU add
// SFPU:           %{{.*}}, %[[ATILE:.*]] = ttl.copy_tile %[[A]][%[[SI0]], %[[SI1]]] into dst[%[[C0]]]
// SFPU:           %{{.*}}, %[[BTILE:.*]] = ttl.copy_tile %[[B]][%[[SI0]], %[[SI1]]] into dst[%c1]
// SFPU:      %[[ADD:.*]] = ttl.tile_add %[[ATILE]], %[[BTILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// copy C for SFPU mul
// SFPU:           %{{.*}}, %[[CTILE:.*]] = ttl.copy_tile %[[C]][%[[SI0]], %[[SI1]]] into dst[%c1]
// SFPU:      %[[MUL:.*]] = ttl.tile_mul %[[ADD]], %[[CTILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// SFPU:           ttl.tile_store %[[MUL]], %{{.*}}[%[[SI0]], %[[SI1]]]
// SFPU-NEXT:      ttl.yield

func.func @mixed_fpu_sfpu(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                          %b: tensor<2x2x!ttcore.tile<32x32, f32>>,
                          %c: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb3 = ttl.bind_cb {cb_index = 16, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view = ttl.cb_reserve %cb3 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%a_cb, %b_cb, %c_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>, %c_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    // FPU: both block args
    %add = ttl.tile_add %a_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    // One operand is computed (add result), so this stays SFPU
    %mul = ttl.tile_mul %add, %c_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %mul, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 3: NOT FPU - computed operand prevents FPU
// =============================================================================
// Purpose: tile_add where one operand is computed (exp result) must stay SFPU.
// No fpu_binary attribute. Both inputs need copy_tile. dstPerIteration=2.

// CHECK-LABEL: func.func @not_fpu_computed_operand
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:           ttl.compute
// CHECK-NEXT:      ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>, %[[B:[^:]*]]: !ttcore.tile<32x32, f32>, %[[OUT:[^:]*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:        %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:        %[[I1:.*]] = ttl.iter_index 1 : index
// No FPU binary in this compute body - one operand is always computed
// CHECK-NOT:       fpu_binary
// A needs copy_tile for exp (SFPU unary)
// CHECK:           %{{.*}}, %[[ATILE:.*]] = ttl.copy_tile %[[A]][%[[I0]], %[[I1]]] into dst[%[[C0]]]
// CHECK:      %[[EXP:.*]] = ttl.tile_exp %[[ATILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// B needs copy_tile for SFPU add; no fpu_binary between exp and copy_tile
// CHECK-NOT:       fpu_binary
// CHECK:           %{{.*}}, %[[BTILE:.*]] = ttl.copy_tile %[[B]][%[[I0]], %[[I1]]] into dst[%c1]
// tile_add is SFPU (one operand computed) - attribute dict closes after dst_idx
// CHECK:      %[[ADD:.*]] = ttl.tile_add %[[EXP]], %[[BTILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:           ttl.tile_store %[[ADD]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:      ttl.yield
//
// SFPU path: same as CHECK - one operand is always computed, so no FPU possible
// SFPU-LABEL: func.func @not_fpu_computed_operand
// SFPU:           ttl.compute
// SFPU-NOT:       fpu_binary
// SFPU:           ttl.copy_tile
// SFPU:           ttl.tile_exp
// SFPU:           ttl.copy_tile
// SFPU:           %[[ADDS:.*]] = ttl.tile_add {{.*}} : !ttcore.tile<32x32, f32>
// SFPU:            ttl.tile_store
// SFPU-NEXT:       ttl.yield

func.func @not_fpu_computed_operand(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                    %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 16, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    // exp makes lhs a computed value
    %c0 = arith.constant 0 : index
    %exp = ttl.tile_exp %a_tile into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    // add has one computed operand -> stays SFPU
    %add = ttl.tile_add %exp, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %add, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 4: All three FPU op types (add, sub, mul)
// =============================================================================
// Purpose: All binary ops with both block args are FPU. Each gets its own
// DST register. No copy_tile needed. dstPerIteration=3, unroll_factor=2.

// CHECK-LABEL: func.func @all_fpu_ops
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : index
// CHECK:           ttl.compute
// CHECK-SAME:      ttl.unroll_factor = 2
// CHECK-NEXT:      ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>, %[[B:[^:]*]]: !ttcore.tile<32x32, f32>,
// CHECK:           %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:      %[[I1:.*]] = ttl.iter_index 1 : index
// No copy_tile for any op - all FPU
// CHECK-NOT:       ttl.copy_tile
// CHECK:           %[[ADD:.*]] = ttl.tile_add %[[A]], %[[B]] into dst[%[[C0]]] {ttl.fpu_binary} : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:      %[[SUB:.*]] = ttl.tile_sub %[[A]], %[[B]] into dst[%[[C1]]] {ttl.fpu_binary} : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:      %[[MUL:.*]] = ttl.tile_mul %[[A]], %[[B]] into dst[%[[C2]]] {ttl.fpu_binary} : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:           ttl.tile_store %[[ADD]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK:           ttl.tile_store %[[SUB]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK:           ttl.tile_store %[[MUL]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:      ttl.yield
//
// SFPU path: all ops need copy_tile for both operands, no fpu_binary
// SFPU-LABEL: func.func @all_fpu_ops
// SFPU-DAG:       %[[C0:.*]] = arith.constant 0 : index
// SFPU-DAG:       %[[C2:.*]] = arith.constant 2 : index
// SFPU-DAG:       %[[C3:.*]] = arith.constant 3 : index
// SFPU:           ttl.compute
// SFPU-SAME:      ttl.unroll_factor = 2
// SFPU-NOT:       fpu_binary
// copy A and B for SFPU ops
// SFPU:           %{{.*}}, %[[ATILE:.*]] = ttl.copy_tile {{.*}}
// SFPU:           %{{.*}}, %[[BTILE:.*]] = ttl.copy_tile {{.*}}
// SFPU:           %[[ADD:.*]] = ttl.tile_add %[[ATILE]], %[[BTILE]] into dst[%[[C2]]] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// SFPU:      %[[SUB:.*]] = ttl.tile_sub %[[ATILE]], %[[BTILE]] into dst[%[[C3]]] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// SFPU:      %[[MUL:.*]] = ttl.tile_mul %[[ATILE]], %[[BTILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// SFPU:            ttl.tile_store
// SFPU:            ttl.tile_store
// SFPU:            ttl.tile_store
// SFPU-NEXT:       ttl.yield

func.func @all_fpu_ops(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                       %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) {
  %c0 = arith.constant 0 : index
  %init0 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init2 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 16, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb3 = ttl.bind_cb {cb_index = 17, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb4 = ttl.bind_cb {cb_index = 18, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init0_cb = ttl.attach_cb %init0, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1_cb = ttl.attach_cb %init1, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init2_cb = ttl.attach_cb %init2, %cb4 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view_0 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view_1 = ttl.cb_reserve %cb3 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view_2 = ttl.cb_reserve %cb4 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result:3 = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init0_cb, %init1_cb, %init2_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>,
       %out0: !ttcore.tile<32x32, f32>, %out1: !ttcore.tile<32x32, f32>, %out2: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %add = ttl.tile_add %a_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %sub = ttl.tile_sub %a_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %mul = ttl.tile_mul %a_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %add, %out_view_0[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.tile_store %sub, %out_view_1[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.tile_store %mul, %out_view_2[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)
  func.return %result#0, %result#1, %result#2 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 5: Block arg used by both FPU and SFPU ops
// =============================================================================
// Purpose: Same block arg (A) is used by an FPU binary (tile_add) and an
// SFPU unary (tile_exp). The FPU binary reads A from CB (no copy needed), but
// the SFPU exp still needs copy_tile for A. dstPerIteration=2, unroll_factor=4.

// CHECK-LABEL: func.func @block_arg_fpu_and_sfpu
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:           ttl.compute
// CHECK-SAME:      ttl.unroll_factor = 4
// CHECK-NEXT:      ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>, %[[B:[^:]*]]: !ttcore.tile<32x32, f32>,
// CHECK:           %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:      %[[I1:.*]] = ttl.iter_index 1 : index
// FPU binary: reads A and B from CB
// CHECK:           %[[ADD:.*]] = ttl.tile_add %[[A]], %[[B]] into dst[%[[C0]]] {ttl.fpu_binary} : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// copy_tile A for SFPU exp
// CHECK:           %{{.*}}, %[[ATILE:.*]] = ttl.copy_tile %[[A]][%[[I0]], %[[I1]]] into dst[%[[C1]]]
// CHECK:      %[[EXP:.*]] = ttl.tile_exp %[[ATILE]] into dst[%[[C1]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:           ttl.tile_store %[[ADD]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK:           ttl.tile_store %[[EXP]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:      ttl.yield
//
// SFPU path: add also needs copy_tile for both operands
// SFPU-LABEL: func.func @block_arg_fpu_and_sfpu
// SFPU-DAG:       %[[C0:.*]] = arith.constant 0 : index
// SFPU-DAG:       %[[C1:.*]] = arith.constant 1 : index
// SFPU:           ttl.compute
// SFPU-SAME:      ttl.unroll_factor = 4
// SFPU-NOT:       fpu_binary
// copy A and B for SFPU add
// SFPU:           %{{.*}}, %[[ATILE1:.*]] = ttl.copy_tile %{{.*}}[%{{.*}}, %{{.*}}] into dst[%[[C0]]]
// SFPU:           %{{.*}}, %[[BTILE:.*]] = ttl.copy_tile %{{.*}}[%{{.*}}, %{{.*}}] into dst[%[[C1]]]
// SFPU:      %[[ADD:.*]] = ttl.tile_add %[[ATILE1]], %[[BTILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// copy A again for exp
// SFPU:           %{{.*}}, %[[ATILE2:.*]] = ttl.copy_tile %{{.*}}[%{{.*}}, %{{.*}}] into dst[%[[C1]]]
// SFPU:      %[[EXP:.*]] = ttl.tile_exp %[[ATILE2]] into dst[%[[C1]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// SFPU:            ttl.tile_store
// SFPU:            ttl.tile_store
// SFPU-NEXT:       ttl.yield

func.func @block_arg_fpu_and_sfpu(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                   %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) {
  %c0 = arith.constant 0 : index
  %init0 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 16, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb3 = ttl.bind_cb {cb_index = 17, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init0_cb = ttl.attach_cb %init0, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1_cb = ttl.attach_cb %init1, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view_0 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view_1 = ttl.cb_reserve %cb3 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result:2 = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init0_cb, %init1_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>,
       %out0: !ttcore.tile<32x32, f32>, %out1: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    // FPU binary: both block args
    %add = ttl.tile_add %a_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    // SFPU unary: block arg A still needs copy_tile for DST
    %exp = ttl.tile_exp %a_tile into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %add, %out_view_0[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.tile_store %exp, %out_view_1[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)
  func.return %result#0, %result#1 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 6: Multiple FPU binary ops consumed mid-body (no DST register reuse)
// =============================================================================
// Purpose: Two FPU binary ops (add, mul) whose results are consumed mid-body
// by SFPU binary ops. Without the interval extension fix, both FPU ops would
// get the same DST index, causing the second to accumulate on the first's
// residual. Pattern: abs(a) + (a+b) + (a*b).
// dstPerIteration=3: DST[0] for abs/SFPU chain, DST[1] for FPU add,
// DST[2] for FPU mul. unroll_factor=2.

// CHECK-LABEL: func.func @fpu_binary_no_dst_reuse
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : index
// CHECK:           ttl.compute
// CHECK-SAME:      ttl.unroll_factor = 2
// CHECK-NEXT:      ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>, %[[B:[^:]*]]: !ttcore.tile<32x32, f32>, %[[OUT:[^:]*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:        %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:        %[[I1:.*]] = ttl.iter_index 1 : index
// copy_tile A for abs (SFPU unary)
// CHECK:           %{{.*}}, %[[ATILE:.*]] = ttl.copy_tile %[[A]][%[[I0]], %[[I1]]] into dst[%[[C0]]]
// CHECK:      %[[ABS:.*]] = ttl.tile_abs %[[ATILE]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// FPU add: reads from CB, gets DST[1]
// CHECK:      %[[ADD:.*]] = ttl.tile_add %[[A]], %[[B]] into dst[%[[C1]]] {ttl.fpu_binary} : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// SFPU binary: abs + add -> DST[0]
// CHECK:      %[[SUM1:.*]] = ttl.tile_add %[[ABS]], %[[ADD]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// FPU mul: reads from CB, gets DST[2] (NOT DST[1] - no reuse!)
// CHECK:      %[[MUL:.*]] = ttl.tile_mul %[[A]], %[[B]] into dst[%[[C2]]] {ttl.fpu_binary} : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// SFPU binary: sum1 + mul -> DST[0]
// CHECK:      %[[SUM2:.*]] = ttl.tile_add %[[SUM1]], %[[MUL]] into dst[%[[C0]]] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK:           ttl.tile_store %[[SUM2]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:      ttl.yield
//
// SFPU path: all binary ops use copy_tile, no fpu_binary. Pattern:
// copy A->0, abs->0, copy A->1, copy B->2, add(1,2)->1, add(0,1)->0,
// copy A->1, mul(1,2)->1, add(0,1)->0
// SFPU-LABEL: func.func @fpu_binary_no_dst_reuse
// SFPU-DAG:       %[[C0:.*]] = arith.constant 0 : index
// SFPU-DAG:       %[[C1:.*]] = arith.constant 1 : index
// SFPU:           ttl.compute
// SFPU-SAME:      ttl.unroll_factor = 2
// SFPU-NOT:       fpu_binary
// SFPU:           ttl.copy_tile {{.*}}
// SFPU:           ttl.tile_abs {{.*}} into dst[%[[C0]]]
// SFPU:           ttl.copy_tile {{.*}}
// SFPU:           ttl.copy_tile {{.*}}
// SFPU:           ttl.tile_add {{.*}} into dst[%[[C1]]]
// SFPU:           ttl.tile_add {{.*}} into dst[%[[C0]]]
// SFPU:           ttl.copy_tile {{.*}}
// SFPU:           ttl.tile_mul {{.*}} into dst[%[[C1]]]
// SFPU:           %[[SUM2S:.*]] = ttl.tile_add {{.*}} into dst[%[[C0]]]
// SFPU:            ttl.tile_store
// SFPU-NEXT:       ttl.yield

func.func @fpu_binary_no_dst_reuse(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                    %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 16, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    // SFPU unary: abs(a) -> needs copy_tile
    %c0 = arith.constant 0 : index
    %abs = ttl.tile_abs %a_tile into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    // FPU binary: a + b (both block args, reads from CB)
    %add = ttl.tile_add %a_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    // SFPU binary: abs(a) + (a+b) - consumes add result
    %sum1 = ttl.tile_add %abs, %add into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    // FPU binary: a * b (both block args) - must NOT reuse add's DST register
    %mul = ttl.tile_mul %a_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    // SFPU binary: result + (a*b)
    %sum2 = ttl.tile_add %sum1, %mul into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %sum2, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
