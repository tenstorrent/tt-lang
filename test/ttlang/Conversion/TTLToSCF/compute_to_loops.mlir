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
  // CHECK-DAG: %[[INIT:.*]] = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %cba = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbb = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_att = ttl.attach_cb %b, %cbb : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-DAG: %[[A_CB:.*]] = ttl.attach_cb %[[ARG0]]
  // CHECK-DAG: %[[B_CB:.*]] = ttl.attach_cb %[[ARG1]]
  // CHECK-DAG: %[[INIT_CB:.*]] = ttl.attach_cb %[[INIT]]
  // CHECK: %[[OUTER:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[OUTER_ARG:.*]] = %[[INIT_CB]]) -> (tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: %[[INNER:.*]] = scf.for %[[J:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[INNER_ARG:.*]] = %[[OUTER_ARG]]) -> (tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: %[[EXT_A:.*]] = tensor.extract %[[A_CB]][%[[I]], %[[J]]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[EXT_B:.*]] = tensor.extract %[[B_CB]][%[[I]], %[[J]]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[SUM:.*]] = ttl.tile_add %[[EXT_A]], %[[EXT_B]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[INS:.*]] = tensor.insert %[[SUM]] into %[[INNER_ARG]][%[[I]], %[[J]]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: scf.yield %[[INS]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK: scf.yield %[[INNER]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK: return %[[OUTER]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute ins(%a_att, %b_att : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} {
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
  // CHECK-DAG: %[[INIT:.*]] = tensor.empty() : tensor<3x3x!ttcore.tile<32x32, f32>>
  %init = tensor.empty() : tensor<3x3x!ttcore.tile<32x32, f32>>
  %cba = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<3x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<3x3x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<3x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<3x3x!ttcore.tile<32x32, f32>>
  // CHECK-DAG: %[[A_CB:.*]] = ttl.attach_cb %[[ARG0]]
  // CHECK-DAG: %[[INIT_CB:.*]] = ttl.attach_cb %[[INIT]]
  // CHECK: %[[OUTER:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[OUTER_ARG:.*]] = %[[INIT_CB]]) -> (tensor<3x3x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: %[[INNER:.*]] = scf.for %[[J:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[INNER_ARG:.*]] = %[[OUTER_ARG]]) -> (tensor<3x3x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: %[[EXT:.*]] = tensor.extract %[[A_CB]][%[[I]], %[[J]]] : tensor<3x3x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[EXP:.*]] = ttl.tile_exp %[[EXT]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[INS:.*]] = tensor.insert %[[EXP]] into %[[INNER_ARG]][%[[I]], %[[J]]] : tensor<3x3x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: scf.yield %[[INS]] : tensor<3x3x!ttcore.tile<32x32, f32>>
  // CHECK: scf.yield %[[INNER]] : tensor<3x3x!ttcore.tile<32x32, f32>>
  // CHECK: return %[[OUTER]] : tensor<3x3x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute ins(%a_att : tensor<3x3x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<3x3x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} {
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
  // CHECK-DAG: %[[INIT:.*]] = tensor.empty() : tensor<4x!ttcore.tile<32x32, f32>>
  %init = tensor.empty() : tensor<4x!ttcore.tile<32x32, f32>>
  %cba = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK-DAG: %[[A_CB:.*]] = ttl.attach_cb %[[ARG0]]
  // CHECK-DAG: %[[INIT_CB:.*]] = ttl.attach_cb %[[INIT]]
  // CHECK: %[[LOOP:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ARG:.*]] = %[[INIT_CB]]) -> (tensor<4x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: %[[EXT:.*]] = tensor.extract %[[A_CB]][%[[I]]] : tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[RELU:.*]] = ttl.tile_relu %[[EXT]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[INS:.*]] = tensor.insert %[[RELU]] into %[[ARG]][%[[I]]] : tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: scf.yield %[[INS]] : tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK: return %[[LOOP]] : tensor<4x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute ins(%a_att : tensor<4x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<4x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} {
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
  %cba = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbb = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_att = ttl.attach_cb %b, %cbb : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-DAG: %[[A_CB:.*]] = ttl.attach_cb %[[ARG0]]
  // CHECK-DAG: %[[B_CB:.*]] = ttl.attach_cb %[[ARG1]]
  // CHECK-DAG: %[[INIT_CB:.*]] = ttl.attach_cb %[[INIT:.*]]
  // CHECK: %[[OUTER:.*]] = scf.for %[[I:.*]] = %[[C0:.*]] to %[[C2:.*]] step %[[C1:.*]] iter_args(%[[ARG_OUTER:.*]] = %[[INIT_CB]]) -> (tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: %[[INNER:.*]] = scf.for %[[J:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG_INNER:.*]] = %[[ARG_OUTER]]) -> (tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK: %[[EXT_A:.*]] = tensor.extract %[[A_CB]][%[[I]], %[[J]]]
  // CHECK: %[[EXT_B:.*]] = tensor.extract %[[B_CB]][%[[I]], %[[J]]]
  // CHECK: %[[ADD:.*]] = ttl.tile_add %[[EXT_A]], %[[EXT_B]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[RELU:.*]] = ttl.tile_relu %[[ADD]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[INS:.*]] = tensor.insert %[[RELU]]
  // CHECK: scf.yield %[[INS]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute ins(%a_att, %b_att : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
    %add = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    %relu = ttl.tile_relu %add : !ttcore.tile<32x32, f32>
    ttl.yield %relu : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Input indexing map permutation is applied when extracting tiles.

#map_perm = affine_map<(d0, d1) -> (d1, d0)>
#map_id2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @compute_permuted_input
// CHECK: tensor.extract %[[ARG0:.*]][%[[J:.*]], %[[I:.*]]]
// CHECK: tensor.extract %[[ARG1:.*]][%[[I]], %[[J]]]
func.func @compute_permuted_input(%a: tensor<3x2x!ttcore.tile<32x32, f32>>, %b: tensor<2x3x!ttcore.tile<32x32, f32>>) -> tensor<2x3x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x3x!ttcore.tile<32x32, f32>>
  %cba = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbb = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<3x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<3x2x!ttcore.tile<32x32, f32>>
  %b_att = ttl.attach_cb %b, %cbb : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute ins(%a_att, %b_att : tensor<3x2x!ttcore.tile<32x32, f32>>, tensor<2x3x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<2x3x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map_perm, #map_id2, #map_id2], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
    %add = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    ttl.yield %add : !ttcore.tile<32x32, f32>
  } -> tensor<2x3x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x3x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Broadcast map drops a dimension for the input tensor.

#map_broadcast = affine_map<(d0, d1) -> (d1)>
#map_id3 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @compute_broadcast_input
// CHECK: tensor.extract %[[ARG0:.*]][%[[J:.*]]]
// CHECK: tensor.insert
func.func @compute_broadcast_input(%a: tensor<3x!ttcore.tile<32x32, f32>>) -> tensor<2x3x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x3x!ttcore.tile<32x32, f32>>
  %cba = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<3x!ttcore.tile<32x32, f32>>, !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>) -> tensor<3x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute ins(%a_att : tensor<3x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<2x3x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map_broadcast, #map_id3], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %relu = ttl.tile_relu %arg0 : !ttcore.tile<32x32, f32>
    ttl.yield %relu : !ttcore.tile<32x32, f32>
  } -> tensor<2x3x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x3x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Reduction iterator accumulates across reduction dimension.

#map_red_in = affine_map<(d0, d1) -> (d0, d1)>
#map_red_out = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @compute_reduction
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x3x!ttcore.tile<32x32, f32>>)
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[INIT:.*]] = tensor.empty() : tensor<2x!ttcore.tile<32x32, f32>>
// CHECK-DAG: %[[A_CB:.*]] = ttl.attach_cb %[[ARG0]]
// CHECK-DAG: %[[INIT_CB:.*]] = ttl.attach_cb %[[INIT]]
// CHECK: %[[OUTER:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG_OUT:.*]] = %[[INIT_CB]]) -> (tensor<2x!ttcore.tile<32x32, f32>>)
// CHECK-NEXT: %[[INNER:.*]] = scf.for %[[J:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG_IN:.*]] = %[[ARG_OUT]]) -> (tensor<2x!ttcore.tile<32x32, f32>>)
// CHECK: %[[EXT_A:.*]] = tensor.extract %[[A_CB]][%[[I]], %[[J]]]
// CHECK: %[[EXT_ACC:.*]] = tensor.extract %[[ARG_IN]][%[[I]]]
// CHECK: %[[ADD:.*]] = ttl.tile_add %[[EXT_A]], %[[EXT_ACC]] : !ttcore.tile<32x32, f32>
// CHECK: %[[INS:.*]] = tensor.insert %[[ADD]] into %[[ARG_IN]][%[[I]]]
// CHECK: scf.yield %[[INS]] : tensor<2x!ttcore.tile<32x32, f32>>
func.func @compute_reduction(%a: tensor<2x3x!ttcore.tile<32x32, f32>>) -> tensor<2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x!ttcore.tile<32x32, f32>>
  %cba = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute ins(%a_att : tensor<2x3x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<2x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map_red_in, #map_red_out], iterator_types = ["parallel", "reduction"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %add = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    ttl.yield %add : !ttcore.tile<32x32, f32>
  } -> tensor<2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Multiple results are inserted with their own indexing maps.

#map_id4 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @compute_multiple_results
// CHECK: tensor.insert
// CHECK: tensor.insert
func.func @compute_multiple_results(%a: tensor<2x2x!ttcore.tile<32x32, f32>>) -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) {
  %init0 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %cba = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbout0 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbout1 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init0_att = ttl.attach_cb %init0, %cbout0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1_att = ttl.attach_cb %init1, %cbout1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %0, %1 = ttl.compute ins(%a_att : tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%init0_att, %init1_att : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map_id4, #map_id4, #map_id4], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
    %relu = ttl.tile_relu %arg0 : !ttcore.tile<32x32, f32>
    ttl.yield %relu, %relu : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>
  } -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)
  func.return %0, %1 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Two compute ops in sequence are both lowered to loops.
// Pseudocode:
//   add_result = compute(a + b)
//   relu_result = compute(relu(add_result))
// Both compute ops should be lowered to nested scf.for loops.

#map_seq = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @compute_two_ops
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[ARG1:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>)
func.func @compute_two_ops(%a: tensor<2x2x!ttcore.tile<32x32, f32>>, %b: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[INIT0:.*]] = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-DAG: %[[INIT1:.*]] = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init0 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %cba = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbb = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbout0 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbout1 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a_cb = ttl.attach_cb %a, %cba : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cbb : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init0_cb = ttl.attach_cb %init0, %cbout0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1_cb = ttl.attach_cb %init1, %cbout1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-DAG: %[[A_CB:.*]] = ttl.attach_cb %[[ARG0]]
  // CHECK-DAG: %[[B_CB:.*]] = ttl.attach_cb %[[ARG1]]
  // CHECK-DAG: %[[INIT0_CB:.*]] = ttl.attach_cb %[[INIT0]]
  // CHECK-DAG: %[[INIT1_CB:.*]] = ttl.attach_cb %[[INIT1]]
  // First compute: add
  // CHECK: %[[ADD_OUTER:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ADD_OUTER_ARG:.*]] = %[[INIT0_CB]]) -> (tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: %[[ADD_INNER:.*]] = scf.for %[[J:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ADD_INNER_ARG:.*]] = %[[ADD_OUTER_ARG]]) -> (tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: %[[EXT_A:.*]] = tensor.extract %[[A_CB]][%[[I]], %[[J]]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[EXT_B:.*]] = tensor.extract %[[B_CB]][%[[I]], %[[J]]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[SUM:.*]] = ttl.tile_add %[[EXT_A]], %[[EXT_B]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[INS_ADD:.*]] = tensor.insert %[[SUM]] into %[[ADD_INNER_ARG]][%[[I]], %[[J]]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: scf.yield %[[INS_ADD]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK: scf.yield %[[ADD_INNER]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  %add_result = ttl.compute ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%init0_cb : tensor<2x2x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map_seq, #map_seq, #map_seq], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // Second compute: relu on the result of first compute
  %add_result_cb = ttl.attach_cb %add_result, %cba : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-DAG: %[[ADD_RESULT_CB:.*]] = ttl.attach_cb %[[ADD_OUTER]]
  // CHECK: %[[RELU_OUTER:.*]] = scf.for %[[I2:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[RELU_OUTER_ARG:.*]] = %[[INIT1_CB]]) -> (tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: %[[RELU_INNER:.*]] = scf.for %[[J2:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[RELU_INNER_ARG:.*]] = %[[RELU_OUTER_ARG]]) -> (tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: %[[EXT_ADD:.*]] = tensor.extract %[[ADD_RESULT_CB]][%[[I2]], %[[J2]]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[RELU:.*]] = ttl.tile_relu %[[EXT_ADD]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[INS_RELU:.*]] = tensor.insert %[[RELU]] into %[[RELU_INNER_ARG]][%[[I2]], %[[J2]]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: scf.yield %[[INS_RELU]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK: scf.yield %[[RELU_INNER]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK: return %[[RELU_OUTER]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  %relu_result = ttl.compute ins(%add_result_cb : tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%init1_cb : tensor<2x2x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map_seq, #map_seq], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %relu = ttl.tile_relu %arg0 : !ttcore.tile<32x32, f32>
    ttl.yield %relu : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %relu_result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
