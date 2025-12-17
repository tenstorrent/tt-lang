// RUN: ttlang-opt %s -ttl-lower-to-loops | FileCheck %s

// Test: ttl.compute with tile_batch_size batching attribute

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @compute_batched_2x1
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>, %[[ARG1:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>)
func.func @compute_batched_2x1(%a: tensor<4x4x!ttcore.tile<32x32, f32>>, %b: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
  %init = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>

  // Outer loops: step by batch size (2 in dim0, 1 in dim1)
  // CHECK: %[[OUTER_I:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C4]] step %[[C2]] iter_args(%[[OUTER_ARG_I:.*]] = %[[INIT]])
  // CHECK: %[[OUTER_J:.*]] = scf.for %[[J:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[OUTER_ARG_J:.*]] = %[[OUTER_ARG_I]])

  // Inner loops: iterate over tiles within batch (0..2 for dim0, 0..1 for dim1)
  // CHECK: %[[INNER_I:.*]] = scf.for %[[II:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[INNER_ARG_I:.*]] = %[[OUTER_ARG_J]])
  // CHECK: %[[INNER_J:.*]] = scf.for %[[JJ:.*]] = %[[C0]] to %[[C1]] step %[[C1]] iter_args(%[[INNER_ARG_J:.*]] = %[[INNER_ARG_I]])

  // Compute actual indices: outer + inner
  // CHECK: %[[IDX_I:.*]] = arith.addi %[[I]], %[[II]]
  // CHECK: %[[IDX_J:.*]] = arith.addi %[[J]], %[[JJ]]

  // Extract, compute, insert using computed indices
  // CHECK: %[[EXT_A:.*]] = tensor.extract %[[ARG0]][%[[IDX_I]], %[[IDX_J]]]
  // CHECK: %[[EXT_B:.*]] = tensor.extract %[[ARG1]][%[[IDX_I]], %[[IDX_J]]]
  // CHECK: %[[SUM:.*]] = ttl.tile_add %[[EXT_A]], %[[EXT_B]]
  // CHECK: %[[INS:.*]] = tensor.insert %[[SUM]] into %[[INNER_ARG_J]][%[[IDX_I]], %[[IDX_J]]]
  // CHECK: scf.yield %[[INS]]

  %0 = ttl.compute ins(%a, %b : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>>)
      outs(%init : tensor<4x4x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"], tile_batch_size = array<i64: 2, 1>} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Batching with 4 tiles per DST cycle (2x2 batch)

// CHECK-LABEL: func.func @compute_batched_2x2
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8x!ttcore.tile<32x32, f32>>)
func.func @compute_batched_2x2(%a: tensor<8x8x!ttcore.tile<32x32, f32>>) -> tensor<8x8x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  %init = tensor.empty() : tensor<8x8x!ttcore.tile<32x32, f32>>

  // Outer loops with step=2 for both dimensions
  // CHECK: scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C2]]
  // CHECK: scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C2]]

  // Inner 2x2 loops
  // CHECK: scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
  // CHECK: scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]

  %0 = ttl.compute ins(%a : tensor<8x8x!ttcore.tile<32x32, f32>>)
      outs(%init : tensor<8x8x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"], tile_batch_size = array<i64: 2, 2>} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %exp = ttl.tile_exp %arg0 : !ttcore.tile<32x32, f32>
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<8x8x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<8x8x!ttcore.tile<32x32, f32>>
}
