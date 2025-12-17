// RUN: ttlang-opt %s -ttl-lower-to-loops | FileCheck %s

// Test: Binary compute op with tile_add lowered to nested scf.for loops.
// Verifies extraction of tiles from inputs, computation, and insertion back.

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @compute_add_2x2
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[ARG1:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>)
func.func @compute_add_2x2(%a: tensor<2x2x!ttcore.tile<32x32, f32>>, %b: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK: %[[OUTER:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[OUTER_ARG:.*]] = %[[INIT]]) -> (tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: %[[INNER:.*]] = scf.for %[[J:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[INNER_ARG:.*]] = %[[OUTER_ARG]]) -> (tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: %[[EXT_A:.*]] = tensor.extract %[[ARG0]][%[[I]], %[[J]]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[EXT_B:.*]] = tensor.extract %[[ARG1]][%[[I]], %[[J]]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[SUM:.*]] = ttl.tile_add %[[EXT_A]], %[[EXT_B]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[INS:.*]] = tensor.insert %[[SUM]] into %[[INNER_ARG]][%[[I]], %[[J]]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: scf.yield %[[INS]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK: scf.yield %[[INNER]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK: return %[[OUTER]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute ins(%a, %b : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%init : tensor<2x2x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Unary compute op with tile_exp lowered to scf.for loops.

#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @compute_exp_3x3
// CHECK-SAME: (%[[ARG0:.*]]: tensor<3x3x!ttcore.tile<32x32, f32>>)
func.func @compute_exp_3x3(%a: tensor<3x3x!ttcore.tile<32x32, f32>>) -> tensor<3x3x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<3x3x!ttcore.tile<32x32, f32>>
  %init = tensor.empty() : tensor<3x3x!ttcore.tile<32x32, f32>>
  // CHECK: %[[OUTER:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[OUTER_ARG:.*]] = %[[INIT]]) -> (tensor<3x3x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: %[[INNER:.*]] = scf.for %[[J:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[INNER_ARG:.*]] = %[[OUTER_ARG]]) -> (tensor<3x3x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: %[[EXT:.*]] = tensor.extract %[[ARG0]][%[[I]], %[[J]]] : tensor<3x3x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[EXP:.*]] = ttl.tile_exp %[[EXT]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[INS:.*]] = tensor.insert %[[EXP]] into %[[INNER_ARG]][%[[I]], %[[J]]] : tensor<3x3x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: scf.yield %[[INS]] : tensor<3x3x!ttcore.tile<32x32, f32>>
  // CHECK: scf.yield %[[INNER]] : tensor<3x3x!ttcore.tile<32x32, f32>>
  // CHECK: return %[[OUTER]] : tensor<3x3x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute ins(%a : tensor<3x3x!ttcore.tile<32x32, f32>>) outs(%init : tensor<3x3x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %exp = ttl.tile_exp %arg0 : !ttcore.tile<32x32, f32>
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<3x3x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<3x3x!ttcore.tile<32x32, f32>>
}

// -----

// Test: 1D tensor produces a single loop.

#map2 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @compute_relu_1d
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x!ttcore.tile<32x32, f32>>)
func.func @compute_relu_1d(%a: tensor<4x!ttcore.tile<32x32, f32>>) -> tensor<4x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<4x!ttcore.tile<32x32, f32>>
  %init = tensor.empty() : tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK: %[[LOOP:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ARG:.*]] = %[[INIT]]) -> (tensor<4x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: %[[EXT:.*]] = tensor.extract %[[ARG0]][%[[I]]] : tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[RELU:.*]] = ttl.tile_relu %[[EXT]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[INS:.*]] = tensor.insert %[[RELU]] into %[[ARG]][%[[I]]] : tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: scf.yield %[[INS]] : tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK: return %[[LOOP]] : tensor<4x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute ins(%a : tensor<4x!ttcore.tile<32x32, f32>>) outs(%init : tensor<4x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %relu = ttl.tile_relu %arg0 : !ttcore.tile<32x32, f32>
    ttl.yield %relu : !ttcore.tile<32x32, f32>
  } -> tensor<4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Chained operations in compute body are all cloned.

#map3 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @compute_chain
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[ARG1:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>)
func.func @compute_chain(%a: tensor<2x2x!ttcore.tile<32x32, f32>>, %b: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: %[[EXT_A:.*]] = tensor.extract
  // CHECK: %[[EXT_B:.*]] = tensor.extract
  // CHECK: %[[ADD:.*]] = ttl.tile_add %[[EXT_A]], %[[EXT_B]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[RELU:.*]] = ttl.tile_relu %[[ADD]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[INS:.*]] = tensor.insert %[[RELU]]
  // CHECK: scf.yield %[[INS]]
  %0 = ttl.compute ins(%a, %b : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%init : tensor<2x2x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
    %add = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    %relu = ttl.tile_relu %add : !ttcore.tile<32x32, f32>
    ttl.yield %relu : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}
