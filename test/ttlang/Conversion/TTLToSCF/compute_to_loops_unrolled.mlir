// Summary: Verify loop unrolling in ttl-lower-to-loops pass.
//
// Tests that ttl.compute ops with ttl.unroll_factor attributes generate
// correctly unrolled scf.for loops with:
// - Loop step equals unroll_factor
// - Multiple unrolled iterations in loop body with adjusted IVs
// - Correct CB tile index computation for each unrolled iteration
// - Separate remainder loop when num_tiles % unroll_factor != 0
//
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-lower-to-loops))' --split-input-file | FileCheck %s

#map = affine_map<(d0) -> (d0)>

// Purpose: 1D loop with 4 tiles, unroll_factor=2 (evenly divisible).
// Expected: Main loop step=2, two unrolled iterations at indices i and i+1.
// Remainder loop: from 4 to 4, step=1 (zero iterations for evenly divisible).
// CHECK-LABEL: func.func @unroll_1d_binary_even
// CHECK-SAME: (%[[A:.*]]: tensor<4x!ttcore.tile<32x32, f32>>, %[[B:.*]]: tensor<4x!ttcore.tile<32x32, f32>>)
func.func @unroll_1d_binary_even(%a: tensor<4x!ttcore.tile<32x32, f32>>,
                                  %b: tensor<4x!ttcore.tile<32x32, f32>>)
    -> tensor<4x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[INIT:.*]] = tensor.empty() : tensor<4x!ttcore.tile<32x32, f32>>
  %init = tensor.empty() : tensor<4x!ttcore.tile<32x32, f32>>
  %cba = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
  %cbb = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>
  %b_att = ttl.attach_cb %b, %cbb : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>

  // Main loop: step=2, bound=floor(4/2)*2=4
  // CHECK: %[[MAIN:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C4]] step %[[C2]] iter_args(%[[ACC:.*]] = %{{.*}}) -> (tensor<4x!ttcore.tile<32x32, f32>>)

  // First unrolled iteration: process tile at index i
  // CHECK-NEXT: %[[EXT_A_0:.*]] = tensor.extract %{{.*}}[%[[I]]] : tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[EXT_B_0:.*]] = tensor.extract %{{.*}}[%[[I]]] : tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[SUM_0:.*]] = ttl.tile_add %[[EXT_A_0]], %[[EXT_B_0]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[INS_0:.*]] = tensor.insert %[[SUM_0]] into %[[ACC]][%[[I]]] : tensor<4x!ttcore.tile<32x32, f32>>

  // Second unrolled iteration: process tile at index i+1
  // CHECK-NEXT: %[[I_PLUS_1:.*]] = arith.addi %[[I]], %[[C1]] : index
  // CHECK-NEXT: %[[EXT_A_1:.*]] = tensor.extract %{{.*}}[%[[I_PLUS_1]]] : tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[EXT_B_1:.*]] = tensor.extract %{{.*}}[%[[I_PLUS_1]]] : tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[SUM_1:.*]] = ttl.tile_add %[[EXT_A_1]], %[[EXT_B_1]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[INS_1:.*]] = tensor.insert %[[SUM_1]] into %[[INS_0]][%[[I_PLUS_1]]] : tensor<4x!ttcore.tile<32x32, f32>>

  // CHECK-NEXT: scf.yield %[[INS_1]] : tensor<4x!ttcore.tile<32x32, f32>>

  // Remainder loop: from 4 to 4, step=1 (zero iterations for evenly divisible)
  // CHECK: %[[REM:.*]] = scf.for %[[I_REM:.*]] = %[[C4]] to %[[C4]] step %[[C1]] iter_args(%[[REM_ACC:.*]] = %[[MAIN]]) -> (tensor<4x!ttcore.tile<32x32, f32>>)
  // CHECK: scf.yield

  // CHECK: return %[[REM]] : tensor<4x!ttcore.tile<32x32, f32>>

  %0 = ttl.compute ins(%a_att, %b_att : tensor<4x!ttcore.tile<32x32, f32>>, tensor<4x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<4x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"], ttl.unroll_factor = 2 : i32} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x!ttcore.tile<32x32, f32>>
}

// -----

#map1d = affine_map<(d0) -> (d0)>

// Purpose: 1D loop with 5 tiles, unroll_factor=2 (remainder: 1 iteration).
// Expected: Main loop step=2, bound=floor(5/2)*2=4 (processes indices 0,1 and 2,3).
// Remainder loop: from 4 to 5, step=1 (processes index 4 safely without OOB access).
// CHECK-LABEL: func.func @unroll_1d_binary_remainder
// CHECK-SAME: (%[[A:.*]]: tensor<5x!ttcore.tile<32x32, f32>>, %[[B:.*]]: tensor<5x!ttcore.tile<32x32, f32>>)
func.func @unroll_1d_binary_remainder(%a: tensor<5x!ttcore.tile<32x32, f32>>,
                                       %b: tensor<5x!ttcore.tile<32x32, f32>>)
    -> tensor<5x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[C5:.*]] = arith.constant 5 : index
  %init = tensor.empty() : tensor<5x!ttcore.tile<32x32, f32>>
  %cba = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[5], !ttcore.tile<32x32, f32>, 2>
  %cbb = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[5], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[5], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<5x!ttcore.tile<32x32, f32>>, !ttl.cb<[5], !ttcore.tile<32x32, f32>, 2>) -> tensor<5x!ttcore.tile<32x32, f32>>
  %b_att = ttl.attach_cb %b, %cbb : (tensor<5x!ttcore.tile<32x32, f32>>, !ttl.cb<[5], !ttcore.tile<32x32, f32>, 2>) -> tensor<5x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<5x!ttcore.tile<32x32, f32>>, !ttl.cb<[5], !ttcore.tile<32x32, f32>, 2>) -> tensor<5x!ttcore.tile<32x32, f32>>

  // Main loop: step=2, bound=floor(5/2)*2=4
  // CHECK: %[[MAIN:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C4]] step %[[C2]] iter_args(%[[ACC:.*]] = %{{.*}}) -> (tensor<5x!ttcore.tile<32x32, f32>>)

  // First unrolled iteration at index i
  // CHECK-NEXT: %[[EXT_A_0:.*]] = tensor.extract %{{.*}}[%[[I]]] : tensor<5x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[EXT_B_0:.*]] = tensor.extract %{{.*}}[%[[I]]] : tensor<5x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[SUM_0:.*]] = ttl.tile_add %[[EXT_A_0]], %[[EXT_B_0]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[INS_0:.*]] = tensor.insert %[[SUM_0]] into %[[ACC]][%[[I]]] : tensor<5x!ttcore.tile<32x32, f32>>

  // Second unrolled iteration at index i+1 (computed)
  // CHECK-NEXT: %[[I_PLUS_1:.*]] = arith.addi %[[I]], %[[C1]] : index
  // CHECK-NEXT: %[[EXT_A_1:.*]] = tensor.extract %{{.*}}[%[[I_PLUS_1]]] : tensor<5x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[EXT_B_1:.*]] = tensor.extract %{{.*}}[%[[I_PLUS_1]]] : tensor<5x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[SUM_1:.*]] = ttl.tile_add %[[EXT_A_1]], %[[EXT_B_1]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[INS_1:.*]] = tensor.insert %[[SUM_1]] into %[[INS_0]][%[[I_PLUS_1]]] : tensor<5x!ttcore.tile<32x32, f32>>

  // CHECK-NEXT: scf.yield %[[INS_1]] : tensor<5x!ttcore.tile<32x32, f32>>

  // Remainder loop: from 4 to 5, step=1
  // CHECK: %[[REM:.*]] = scf.for %[[I_REM:.*]] = %[[C4]] to %[[C5]] step %[[C1]] iter_args(%[[REM_ACC:.*]] = %[[MAIN]]) -> (tensor<5x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: %[[REM_EXT_A:.*]] = tensor.extract %{{.*}}[%[[I_REM]]] : tensor<5x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[REM_EXT_B:.*]] = tensor.extract %{{.*}}[%[[I_REM]]] : tensor<5x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[REM_SUM:.*]] = ttl.tile_add %[[REM_EXT_A]], %[[REM_EXT_B]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[REM_INS:.*]] = tensor.insert %[[REM_SUM]] into %[[REM_ACC]][%[[I_REM]]] : tensor<5x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: scf.yield %[[REM_INS]] : tensor<5x!ttcore.tile<32x32, f32>>

  // CHECK: return %[[REM]] : tensor<5x!ttcore.tile<32x32, f32>>

  %0 = ttl.compute ins(%a_att, %b_att : tensor<5x!ttcore.tile<32x32, f32>>, tensor<5x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<5x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map1d, #map1d, #map1d], iterator_types = ["parallel"], ttl.unroll_factor = 2 : i32} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<5x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<5x!ttcore.tile<32x32, f32>>
}

// -----

#map2d = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: 2D loop with 2x2 tiles, unroll_factor=2 on innermost dimension.
// Expected: inner loop has step=2, outer loop has step=1.
// Remainder loop nest with inner loop: 2 to 2, step=1 (zero iterations).
// CB tile index must be linearized for StoreOp.
// CHECK-LABEL: func.func @unroll_2d_binary
// CHECK-SAME: (%[[A:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[B:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>)
func.func @unroll_2d_binary(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                             %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %cba = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cbb = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_att = ttl.attach_cb %b, %cbb : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Main loop nest: outer loop step=1, inner loop step=2 (unrolled)
  // CHECK: %[[OUTER:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[OUTER_ACC:.*]] = %{{.*}}) -> (tensor<2x2x!ttcore.tile<32x32, f32>>)

  // Inner main loop: step=2 (unrolled), bound=floor(2/2)*2=2
  // CHECK-NEXT: %[[INNER:.*]] = scf.for %[[J:.*]] = %[[C0]] to %[[C2]] step %[[C2]] iter_args(%[[INNER_ACC:.*]] = %[[OUTER_ACC]]) -> (tensor<2x2x!ttcore.tile<32x32, f32>>)

  // First iteration: [i, j]
  // CHECK-NEXT: %[[EXT_A_0:.*]] = tensor.extract %{{.*}}[%[[I]], %[[J]]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[EXT_B_0:.*]] = tensor.extract %{{.*}}[%[[I]], %[[J]]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[SUM_0:.*]] = ttl.tile_add %[[EXT_A_0]], %[[EXT_B_0]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[INS_0:.*]] = tensor.insert %[[SUM_0]] into %[[INNER_ACC]][%[[I]], %[[J]]] : tensor<2x2x!ttcore.tile<32x32, f32>>

  // Second iteration: [i, j+1]
  // CHECK-NEXT: %[[J_PLUS_1:.*]] = arith.addi %[[J]], %[[C1]] : index
  // CHECK-NEXT: %[[EXT_A_1:.*]] = tensor.extract %{{.*}}[%[[I]], %[[J_PLUS_1]]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[EXT_B_1:.*]] = tensor.extract %{{.*}}[%[[I]], %[[J_PLUS_1]]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[SUM_1:.*]] = ttl.tile_add %[[EXT_A_1]], %[[EXT_B_1]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[INS_1:.*]] = tensor.insert %[[SUM_1]] into %[[INS_0]][%[[I]], %[[J_PLUS_1]]] : tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK-NEXT: scf.yield %[[INS_1]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK: scf.yield %[[INNER]] : tensor<2x2x!ttcore.tile<32x32, f32>>

  // Remainder loop nest: outer loop processes all rows again, inner loop 2 to 2 (zero iterations)
  // CHECK: %[[REM_OUTER:.*]] = scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%{{.*}} = %[[OUTER]]) -> (tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: %[[REM_INNER:.*]] = scf.for %{{.*}} = %[[C2]] to %[[C2]] step %[[C1]]
  // CHECK: scf.yield
  // CHECK: scf.yield %[[REM_INNER]]

  // CHECK: return %[[REM_OUTER]] : tensor<2x2x!ttcore.tile<32x32, f32>>

  %0 = ttl.compute ins(%a_att, %b_att : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map2d, #map2d, #map2d], iterator_types = ["parallel", "parallel"], ttl.unroll_factor = 2 : i32} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map_unary = affine_map<(d0) -> (d0)>

// Purpose: Unary op with 4 tiles, unroll_factor=4 (high unroll, evenly divisible).
// Expected: Main loop step=4, four unrolled iterations in loop body.
// Remainder loop: from 4 to 4, step=1 (zero iterations).
// CHECK-LABEL: func.func @unroll_1d_unary_high
// CHECK-SAME: (%[[A:.*]]: tensor<4x!ttcore.tile<32x32, f32>>)
func.func @unroll_1d_unary_high(%a: tensor<4x!ttcore.tile<32x32, f32>>)
    -> tensor<4x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  %init = tensor.empty() : tensor<4x!ttcore.tile<32x32, f32>>
  %cba = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>

  // Main loop: step=4, bound=floor(4/4)*4=4 (processes all 4 tiles in single iteration)
  // CHECK: %[[MAIN:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C4]] step %[[C4]] iter_args(%[[ACC:.*]] = %{{.*}}) -> (tensor<4x!ttcore.tile<32x32, f32>>)

  // First iteration: i
  // CHECK-NEXT: %[[EXT_0:.*]] = tensor.extract %{{.*}}[%[[I]]] : tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[EXP_0:.*]] = ttl.tile_exp %[[EXT_0]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[INS_0:.*]] = tensor.insert %[[EXP_0]] into %[[ACC]][%[[I]]] : tensor<4x!ttcore.tile<32x32, f32>>

  // Second iteration: i+1
  // CHECK-NEXT: %[[I_PLUS_1:.*]] = arith.addi %[[I]], %[[C1]] : index
  // CHECK-NEXT: %[[EXT_1:.*]] = tensor.extract %{{.*}}[%[[I_PLUS_1]]] : tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[EXP_1:.*]] = ttl.tile_exp %[[EXT_1]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[INS_1:.*]] = tensor.insert %[[EXP_1]] into %[[INS_0]][%[[I_PLUS_1]]] : tensor<4x!ttcore.tile<32x32, f32>>

  // Third iteration: i+2
  // CHECK-NEXT: %[[I_PLUS_2:.*]] = arith.addi %[[I]], %[[C2]] : index
  // CHECK-NEXT: %[[EXT_2:.*]] = tensor.extract %{{.*}}[%[[I_PLUS_2]]] : tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[EXP_2:.*]] = ttl.tile_exp %[[EXT_2]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[INS_2:.*]] = tensor.insert %[[EXP_2]] into %[[INS_1]][%[[I_PLUS_2]]] : tensor<4x!ttcore.tile<32x32, f32>>

  // Fourth iteration: i+3
  // CHECK-NEXT: %[[I_PLUS_3:.*]] = arith.addi %[[I]], %[[C3]] : index
  // CHECK-NEXT: %[[EXT_3:.*]] = tensor.extract %{{.*}}[%[[I_PLUS_3]]] : tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[EXP_3:.*]] = ttl.tile_exp %[[EXT_3]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[INS_3:.*]] = tensor.insert %[[EXP_3]] into %[[INS_2]][%[[I_PLUS_3]]] : tensor<4x!ttcore.tile<32x32, f32>>

  // CHECK-NEXT: scf.yield %[[INS_3]] : tensor<4x!ttcore.tile<32x32, f32>>

  // Remainder loop: from 4 to 4, step=1 (zero iterations)
  // CHECK: %[[REM:.*]] = scf.for %[[I_REM:.*]] = %[[C4]] to %[[C4]] step %[[C1]] iter_args(%[[REM_ACC:.*]] = %[[MAIN]]) -> (tensor<4x!ttcore.tile<32x32, f32>>)
  // CHECK: scf.yield

  // CHECK: return %[[REM]] : tensor<4x!ttcore.tile<32x32, f32>>

  %0 = ttl.compute ins(%a_att : tensor<4x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<4x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map_unary, #map_unary], iterator_types = ["parallel"], ttl.unroll_factor = 4 : i32} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %exp = ttl.tile_exp %arg0 : !ttcore.tile<32x32, f32>
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x!ttcore.tile<32x32, f32>>
}

// -----

#map_store = affine_map<(d0) -> (d0)>

// Purpose: Verify CB tile index computation with StoreOp in unrolled loop.
// For each unrolled iteration, the CB index must use the adjusted IV.
// Remainder loop: from 4 to 4, step=1 (zero iterations for evenly divisible).
// CHECK-LABEL: func.func @unroll_with_store
// CHECK-SAME: (%[[A:.*]]: tensor<4x!ttcore.tile<32x32, f32>>, %[[B:.*]]: tensor<4x!ttcore.tile<32x32, f32>>)
func.func @unroll_with_store(%a: tensor<4x!ttcore.tile<32x32, f32>>,
                              %b: tensor<4x!ttcore.tile<32x32, f32>>)
    -> tensor<4x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<4x!ttcore.tile<32x32, f32>>
  %cba = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
  %cbb = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>
  %b_att = ttl.attach_cb %b, %cbb : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>

  // Main loop: step=2, bound=floor(4/2)*2=4
  // CHECK: %[[MAIN:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C4]] step %[[C2]] iter_args(%[[ACC:.*]] = %{{.*}}) -> (tensor<4x!ttcore.tile<32x32, f32>>)

  // First iteration at index i: CB index must be i
  // CHECK-NEXT: %[[EXT_A_0:.*]] = tensor.extract %{{.*}}[%[[I]]] : tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[EXT_B_0:.*]] = tensor.extract %{{.*}}[%[[I]]] : tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[SUM_0:.*]] = ttl.tile_add %[[EXT_A_0]], %[[EXT_B_0]] : !ttcore.tile<32x32, f32>
  // Verify CB index for first iteration is i (not a constant)
  // CHECK: ttl.store %[[SUM_0]], %{{.*}}[%[[I]]] : !ttcore.tile<32x32, f32>, tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[INS_0:.*]] = tensor.insert %[[SUM_0]] into %[[ACC]][%[[I]]] : tensor<4x!ttcore.tile<32x32, f32>>

  // Second iteration at index i+1: CB index must be i+1
  // CHECK-NEXT: %[[I_PLUS_1:.*]] = arith.addi %[[I]], %[[C1]] : index
  // CHECK-NEXT: %[[EXT_A_1:.*]] = tensor.extract %{{.*}}[%[[I_PLUS_1]]] : tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[EXT_B_1:.*]] = tensor.extract %{{.*}}[%[[I_PLUS_1]]] : tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[SUM_1:.*]] = ttl.tile_add %[[EXT_A_1]], %[[EXT_B_1]] : !ttcore.tile<32x32, f32>
  // CHECK: ttl.store %[[SUM_1]], %{{.*}}[%[[I_PLUS_1]]] : !ttcore.tile<32x32, f32>, tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[INS_1:.*]] = tensor.insert %[[SUM_1]] into %[[INS_0]][%[[I_PLUS_1]]] : tensor<4x!ttcore.tile<32x32, f32>>

  // CHECK-NEXT: scf.yield %[[INS_1]] : tensor<4x!ttcore.tile<32x32, f32>>

  // Remainder loop: from 4 to 4, step=1 (zero iterations)
  // CHECK: %[[REM:.*]] = scf.for %[[I_REM:.*]] = %[[C4]] to %[[C4]] step %[[C1]] iter_args(%[[REM_ACC:.*]] = %[[MAIN]]) -> (tensor<4x!ttcore.tile<32x32, f32>>)
  // CHECK: scf.yield

  // CHECK: return %[[REM]] : tensor<4x!ttcore.tile<32x32, f32>>

  %0 = ttl.compute ins(%a_att, %b_att : tensor<4x!ttcore.tile<32x32, f32>>, tensor<4x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<4x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map_store, #map_store, #map_store], iterator_types = ["parallel"], ttl.unroll_factor = 2 : i32} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    %view = ttl.cb_reserve %cbout : <[4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x!ttcore.tile<32x32, f32>>
    ttl.store %sum, %view[%c0] : !ttcore.tile<32x32, f32>, tensor<4x!ttcore.tile<32x32, f32>>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x!ttcore.tile<32x32, f32>>
}

// -----

#map_no_unroll = affine_map<(d0) -> (d0)>

// Purpose: ttl.compute without unroll_factor attribute should not unroll (default behavior when attribute absent).
// Expected: scf.for step=1 (normal loop, no unrolling).
// CHECK-LABEL: func.func @no_unroll_attr
// CHECK-SAME: (%[[A:.*]]: tensor<4x!ttcore.tile<32x32, f32>>)
func.func @no_unroll_attr(%a: tensor<4x!ttcore.tile<32x32, f32>>)
    -> tensor<4x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  %init = tensor.empty() : tensor<4x!ttcore.tile<32x32, f32>>
  %cba = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>

  // Verify NO unrolling: step=1, single iteration in loop body
  // CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ACC:.*]] = %{{.*}}) -> (tensor<4x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: %[[EXT:.*]] = tensor.extract %{{.*}}[%[[I]]] : tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK: %[[EXP:.*]] = ttl.tile_exp %[[EXT]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[INS:.*]] = tensor.insert %[[EXP]] into %[[ACC]][%[[I]]] : tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: scf.yield %[[INS]] : tensor<4x!ttcore.tile<32x32, f32>>
  // Verify no adjusted IVs are computed (no arith.addi for i+1, i+2, etc.)
  // CHECK-NOT: arith.addi

  %0 = ttl.compute ins(%a_att : tensor<4x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<4x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map_no_unroll, #map_no_unroll], iterator_types = ["parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %exp = ttl.tile_exp %arg0 : !ttcore.tile<32x32, f32>
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x!ttcore.tile<32x32, f32>>
}
