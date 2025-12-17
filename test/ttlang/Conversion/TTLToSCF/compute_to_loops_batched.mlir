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
  // CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C4]] step %[[C2]] iter_args(%[[ARGI:.*]] = %[[INIT]])
  // CHECK: scf.for %[[J:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ARGJ:.*]] = %[[ARGI]])

  // Inner loops: iterate over tiles within batch (clamped at tail)
  // CHECK: %[[REM_I:.*]] = arith.subi %[[C4]], %[[I]]
  // CHECK: %[[UB_I:.*]] = arith.select {{.*}} %[[REM_I]], %[[C2]]
  // CHECK: %[[REM_J:.*]] = arith.subi %[[C4]], %[[J]]
  // CHECK: %[[UB_J:.*]] = arith.select {{.*}} %[[REM_J]], %[[C1]]
  // CHECK: scf.for %[[II:.*]] = %[[C0]] to %[[UB_I]] step %[[C1]] iter_args(%[[ARGII:.*]] = %[[ARGJ]])
  // CHECK: scf.for %[[JJ:.*]] = %[[C0]] to %[[UB_J]] step %[[C1]] iter_args(%[[ARGJJ:.*]] = %[[ARGII]])

  // Compute actual indices: outer + inner
  // CHECK: %[[IDX_I:.*]] = arith.addi %[[I]], %[[II]]
  // CHECK: %[[IDX_J:.*]] = arith.addi %[[J]], %[[JJ]]

  // Extract, compute, insert using computed indices
  // CHECK: %[[EXT_A:.*]] = tensor.extract %[[ARG0]][%[[IDX_I]], %[[IDX_J]]]
  // CHECK: %[[EXT_B:.*]] = tensor.extract %[[ARG1]][%[[IDX_I]], %[[IDX_J]]]
  // CHECK: %[[SUM:.*]] = ttl.tile_add %[[EXT_A]], %[[EXT_B]]
  // CHECK: tensor.insert %[[SUM]] into %{{.*}}[%[[IDX_I]], %[[IDX_J]]]
  // CHECK: scf.yield %{{.*}} : tensor<4x4x!ttcore.tile<32x32, f32>>

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
  // CHECK: scf.for %[[OI:.*]] = %[[C0]] to %[[C8]] step %[[C2]]
  // CHECK: scf.for %[[OJ:.*]] = %[[C0]] to %[[C8]] step %[[C2]]

  // Tail-clamped inner 2x2 loops
  // CHECK: %[[REM_OI:.*]] = arith.subi %[[C8]], %[[OI]]
  // CHECK: %[[UB_OI:.*]] = arith.select {{.*}} %[[REM_OI]], %[[C2]]
  // CHECK: %[[REM_OJ:.*]] = arith.subi %[[C8]], %[[OJ]]
  // CHECK: %[[UB_OJ:.*]] = arith.select {{.*}} %[[REM_OJ]], %[[C2]]
  // CHECK: scf.for %{{.*}} = %[[C0]] to %[[UB_OI]] step %[[C1]]
  // CHECK: scf.for %{{.*}} = %[[C0]] to %[[UB_OJ]] step %[[C1]]

  %0 = ttl.compute ins(%a : tensor<8x8x!ttcore.tile<32x32, f32>>)
      outs(%init : tensor<8x8x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"], tile_batch_size = array<i64: 2, 2>} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %exp = ttl.tile_exp %arg0 : !ttcore.tile<32x32, f32>
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<8x8x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<8x8x!ttcore.tile<32x32, f32>>
}
