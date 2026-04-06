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
  %cba = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_att = ttl.attach_cb %b, %cbb : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-DAG: %[[A_CB:.*]] = ttl.attach_cb %[[ARG0]]
  // CHECK-DAG: %[[B_CB:.*]] = ttl.attach_cb %[[ARG1]]
  // CHECK-DAG: %[[INIT_CB:.*]] = ttl.attach_cb %[[INIT]]
  // CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
  // CHECK-NEXT: scf.for %[[J:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
  // CHECK-NEXT: ttl.dst_section {
  // CHECK-NEXT: %[[EXT_A:.*]] = tensor.extract %[[A_CB]][%[[I]], %[[J]]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[EXT_B:.*]] = tensor.extract %[[B_CB]][%[[I]], %[[J]]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[SUM:.*]] = ttl.tile_add %[[EXT_A]], %[[EXT_B]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: ttl.tile_store %[[SUM]], %{{.*}}[%[[I]], %[[J]]]
  // CHECK-NEXT: }
        // CHECK: return
  %out_view = ttl.cb_reserve %cbout : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute ins(%a_att, %b_att : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
    %i0 = ttl.iter_index 0 : index
    %j0 = ttl.iter_index 1 : index
    %sum = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    ttl.tile_store %sum, %out_view[%i0, %j0] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
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
  %cba = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<3x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<3x3x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<3x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<3x3x!ttcore.tile<32x32, f32>>
  // CHECK-DAG: %[[A_CB:.*]] = ttl.attach_cb %[[ARG0]]
  // CHECK-DAG: %[[INIT_CB:.*]] = ttl.attach_cb %[[INIT]]
  // CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C3]] step %[[C1]]
  // CHECK-NEXT: scf.for %[[J:.*]] = %[[C0]] to %[[C3]] step %[[C1]]
  // CHECK-NEXT: ttl.dst_section {
  // CHECK-NEXT: %[[EXT:.*]] = tensor.extract %[[A_CB]][%[[I]], %[[J]]] : tensor<3x3x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[EXP:.*]] = ttl.tile_exp %[[EXT]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: ttl.tile_store %[[EXP]], %{{.*}}[%[[I]], %[[J]]]
  // CHECK-NEXT: }
        // CHECK: return
  %out_view_0 = ttl.cb_reserve %cbout : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute ins(%a_att : tensor<3x3x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<3x3x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %i1 = ttl.iter_index 0 : index
    %j1 = ttl.iter_index 1 : index
    %exp = ttl.tile_exp %arg0 : !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %out_view_0[%i1, %j1] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
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
  %cba = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK-DAG: %[[A_CB:.*]] = ttl.attach_cb %[[ARG0]]
  // CHECK-DAG: %[[INIT_CB:.*]] = ttl.attach_cb %[[INIT]]
  // CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C4]] step %[[C1]]
  // CHECK-NEXT: ttl.dst_section {
  // CHECK-NEXT: %[[EXT:.*]] = tensor.extract %[[A_CB]][%[[I]]] : tensor<4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[RELU:.*]] = ttl.tile_relu %[[EXT]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: ttl.tile_store %[[RELU]], %{{.*}}[%[[I]]]
  // CHECK-NEXT: }
      // CHECK: return
  %out_view_1 = ttl.cb_reserve %cbout : <[1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute ins(%a_att : tensor<4x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<4x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %i2 = ttl.iter_index 0 : index
    %relu = ttl.tile_relu %arg0 : !ttcore.tile<32x32, f32>
    ttl.tile_store %relu, %out_view_1[%i2] : !ttcore.tile<32x32, f32>, tensor<1x!ttcore.tile<32x32, f32>>
    ttl.yield
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
  %cba = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_att = ttl.attach_cb %b, %cbb : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-DAG: %[[A_CB:.*]] = ttl.attach_cb %[[ARG0]]
  // CHECK-DAG: %[[B_CB:.*]] = ttl.attach_cb %[[ARG1]]
  // CHECK-DAG: %[[INIT_CB:.*]] = ttl.attach_cb %[[INIT:.*]]
  // CHECK: scf.for %[[I:.*]] = %[[C0:.*]] to %[[C2:.*]] step %[[C1:.*]]
  // CHECK-NEXT: scf.for %[[J:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
  // CHECK-NEXT: ttl.dst_section {
  // CHECK: %[[EXT_A:.*]] = tensor.extract %[[A_CB]][%[[I]], %[[J]]]
  // CHECK: %[[EXT_B:.*]] = tensor.extract %[[B_CB]][%[[I]], %[[J]]]
  // CHECK: %[[ADD:.*]] = ttl.tile_add %[[EXT_A]], %[[EXT_B]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[RELU:.*]] = ttl.tile_relu %[[ADD]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: ttl.tile_store %[[RELU]], %{{.*}}[%[[I]], %[[J]]]
  // CHECK-NEXT: }
      %out_view_2 = ttl.cb_reserve %cbout : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %0 = ttl.compute ins(%a_att, %b_att : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
    %i3 = ttl.iter_index 0 : index
    %j3 = ttl.iter_index 1 : index
    %add = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    %relu = ttl.tile_relu %add : !ttcore.tile<32x32, f32>
    ttl.tile_store %relu, %out_view_2[%i3, %j3] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Input indexing map permutation is applied when extracting tiles.

#map_perm = affine_map<(d0, d1) -> (d1, d0)>
#map_id2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @compute_permuted_input
// CHECK: ttl.dst_section {
// CHECK: tensor.extract %[[ARG0:.*]][%[[J:.*]], %[[I:.*]]]
// CHECK: tensor.extract %[[ARG1:.*]][%[[I]], %[[J]]]
func.func @compute_permuted_input(%a: tensor<3x2x!ttcore.tile<32x32, f32>>, %b: tensor<2x3x!ttcore.tile<32x32, f32>>) -> tensor<2x3x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x3x!ttcore.tile<32x32, f32>>
  %cba = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<3x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<3x2x!ttcore.tile<32x32, f32>>
  %b_att = ttl.attach_cb %b, %cbb : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %out_view_3 = ttl.cb_reserve %cbout : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute ins(%a_att, %b_att : tensor<3x2x!ttcore.tile<32x32, f32>>, tensor<2x3x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<2x3x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map_perm, #map_id2, #map_id2], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
    %i4 = ttl.iter_index 0 : index
    %j4 = ttl.iter_index 1 : index
    %add = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    ttl.tile_store %add, %out_view_3[%i4, %j4] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x3x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x3x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Broadcast map drops a dimension for the input tensor.

#map_broadcast = affine_map<(d0, d1) -> (d1)>
#map_id3 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @compute_broadcast_input
// CHECK: ttl.dst_section {
// CHECK: tensor.extract %[[ARG0:.*]][%[[J:.*]]]
func.func @compute_broadcast_input(%a: tensor<3x!ttcore.tile<32x32, f32>>) -> tensor<2x3x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x3x!ttcore.tile<32x32, f32>>
  %cba = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<3x!ttcore.tile<32x32, f32>>, !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>) -> tensor<3x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %out_view_4 = ttl.cb_reserve %cbout : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute ins(%a_att : tensor<3x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<2x3x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map_broadcast, #map_id3], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %i5 = ttl.iter_index 0 : index
    %j5 = ttl.iter_index 1 : index
    %relu = ttl.tile_relu %arg0 : !ttcore.tile<32x32, f32>
    ttl.tile_store %relu, %out_view_4[%i5, %j5] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
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
// CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-NEXT: scf.for %[[J:.*]] = %[[C0]] to %[[C3]] step %[[C1]]
// CHECK-NEXT: ttl.dst_section {
// CHECK: %[[EXT_A:.*]] = tensor.extract %[[A_CB]][%[[I]], %[[J]]]
// CHECK: %[[EXT_ACC:.*]] = tensor.extract %[[INIT_CB]][%[[I]]]
// CHECK: %[[ADD:.*]] = ttl.tile_add %[[EXT_A]], %[[EXT_ACC]] : !ttcore.tile<32x32, f32>
// Reduction output map (d0,d1)->(d0): tile_store gets only the parallel dim.
// CHECK: ttl.tile_store %[[ADD]], %{{.*}}[%[[I]]]
// CHECK-NEXT: }
func.func @compute_reduction(%a: tensor<2x3x!ttcore.tile<32x32, f32>>) -> tensor<2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x!ttcore.tile<32x32, f32>>
  %cba = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x!ttcore.tile<32x32, f32>>
  %out_view_5 = ttl.cb_reserve %cbout : <[1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute ins(%a_att : tensor<2x3x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<2x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map_red_in, #map_red_out], iterator_types = ["parallel", "reduction"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %i6 = ttl.iter_index 0 : index
    %j6 = ttl.iter_index 1 : index
    %add = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    ttl.tile_store %add, %out_view_5[%i6] : !ttcore.tile<32x32, f32>, tensor<1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Multiple results are inserted with their own indexing maps.

#map_id4 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @compute_multiple_results
func.func @compute_multiple_results(%a: tensor<2x2x!ttcore.tile<32x32, f32>>) -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) {
  %init0 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %cba = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbout0 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbout1 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init0_att = ttl.attach_cb %init0, %cbout0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1_att = ttl.attach_cb %init1, %cbout1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view_6 = ttl.cb_reserve %cbout0 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %out_view_7 = ttl.cb_reserve %cbout1 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %0, %1 = ttl.compute ins(%a_att : tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%init0_att, %init1_att : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map_id4, #map_id4, #map_id4], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
    %i7 = ttl.iter_index 0 : index
    %j7 = ttl.iter_index 1 : index
    %relu = ttl.tile_relu %arg0 : !ttcore.tile<32x32, f32>
    ttl.tile_store %relu, %out_view_6[%i7, %j7] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.tile_store %relu, %out_view_7[%i7, %j7] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)
  func.return %0, %1 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Two compute ops in sequence are both lowered to loops.
// Pseudocode:
//   add_result = compute(a + b)
//   relu_result = compute(relu(add_result))
// Both compute ops should be lowered to nested scf.for loops.
// The intermediate result uses a fresh CB (cb_index=4), not reusing an input CB.

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
  %cba = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbout0 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbout1 = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  // Fresh CB for intermediate result (not reusing input CBs)
  %cb_intermediate = ttl.bind_cb {cb_index = 4, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a_cb = ttl.attach_cb %a, %cba : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cbb : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init0_cb = ttl.attach_cb %init0, %cbout0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1_cb = ttl.attach_cb %init1, %cbout1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-DAG: %[[A_CB:.*]] = ttl.attach_cb %[[ARG0]]
  // CHECK-DAG: %[[B_CB:.*]] = ttl.attach_cb %[[ARG1]]
  // CHECK-DAG: %[[INIT0_CB:.*]] = ttl.attach_cb %[[INIT0]]
  // CHECK-DAG: %[[INIT1_CB:.*]] = ttl.attach_cb %[[INIT1]]
  // First compute: add
  // CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
  // CHECK-NEXT: scf.for %[[J:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
  // CHECK-NEXT: ttl.dst_section {
  // CHECK-NEXT: %[[EXT_A:.*]] = tensor.extract %[[A_CB]][%[[I]], %[[J]]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[EXT_B:.*]] = tensor.extract %[[B_CB]][%[[I]], %[[J]]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[SUM:.*]] = ttl.tile_add %[[EXT_A]], %[[EXT_B]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: ttl.tile_store %[[SUM]], %{{.*}}[%[[I]], %[[J]]]
  // CHECK-NEXT: }
        %out_view_8 = ttl.cb_reserve %cbout0 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %add_result = ttl.compute ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%init0_cb : tensor<2x2x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map_seq, #map_seq, #map_seq], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
    %i8 = ttl.iter_index 0 : index
    %j8 = ttl.iter_index 1 : index
    %sum = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    ttl.tile_store %sum, %out_view_8[%i8, %j8] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // Second compute: relu on the result of first compute (using fresh CB, not input CB)
  %add_result_cb = ttl.attach_cb %add_result, %cb_intermediate : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-DAG: %[[ADD_RESULT_CB:.*]] = ttl.attach_cb %[[INIT0_CB]]
  // CHECK: scf.for %[[I2:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
  // CHECK-NEXT: scf.for %[[J2:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
  // CHECK-NEXT: ttl.dst_section {
  // CHECK-NEXT: %[[EXT_ADD:.*]] = tensor.extract %[[ADD_RESULT_CB]][%[[I2]], %[[J2]]] : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[RELU:.*]] = ttl.tile_relu %[[EXT_ADD]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: ttl.tile_store %[[RELU]], %{{.*}}[%[[I2]], %[[J2]]]
  // CHECK-NEXT: }
        // CHECK: return
  %out_view_9 = ttl.cb_reserve %cbout1 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %relu_result = ttl.compute ins(%add_result_cb : tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%init1_cb : tensor<2x2x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map_seq, #map_seq], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %i9 = ttl.iter_index 0 : index
    %j9 = ttl.iter_index 1 : index
    %relu = ttl.tile_relu %arg0 : !ttcore.tile<32x32, f32>
    ttl.tile_store %relu, %out_view_9[%i9, %j9] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %relu_result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
