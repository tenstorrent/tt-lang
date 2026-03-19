// Summary: Tests DST allocation for binary in-place ops (max, min). These ops
// use 2-arg in-place form: DST[dst0] = op(DST[dst0], DST[dst1]), so operand 0
// is clobbered. When both max and min share operands, copies must be inserted
// to prevent one op from destroying the other's inputs.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8}))' --split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 1: max(a,b) + min(a,b) -- both ops share the same operands
// =============================================================================
// Purpose: max and min both read from %a (operand 0) and %b (operand 1). Since
// both are in-place (clobber operand 0), copies must be inserted so that min
// gets unclobbered inputs after max runs.

// CHECK-LABEL: func.func @max_plus_min
// CHECK:           ttl.compute
// CHECK-NEXT:      ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, bf16>, %[[B:[^:]*]]: !ttcore.tile<32x32, bf16>, %[[OUT:[^:]*]]: !ttcore.tile<32x32, bf16>):

// First pair: copy_tile for max's operands
// CHECK:           %{{.*}}, %[[A_COPY:.*]] = ttl.copy_tile %[[A]]
// CHECK:           %{{.*}}, %[[B_COPY:.*]] = ttl.copy_tile %[[B]]

// Max operates on copies (in-place, clobbers A_COPY)
// CHECK:           %[[MX:.*]] = ttl.tile_max %[[A_COPY]], %[[B_COPY]]

// Second pair: copy_tile for min's operands (fresh copies of a and b)
// CHECK:           %{{.*}}, %[[A2:.*]] = ttl.copy_tile %[[A]]
// CHECK:           %{{.*}}, %[[B2:.*]] = ttl.copy_tile %[[B]]

// Min operates on fresh copies (in-place, clobbers A2)
// CHECK:           %[[MN:.*]] = ttl.tile_min %[[A2]], %[[B2]]

// Add uses the results (max from DST and min from DST)
// CHECK:           %[[SUM:.*]] = ttl.tile_add %[[MX]], %[[MN]]
// CHECK:           ttl.tile_store %[[SUM]]
// CHECK-NEXT:      ttl.yield

func.func @max_plus_min(%a: tensor<1x1x!ttcore.tile<32x32, bf16>>,
                        %b: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_view = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, bf16>, %b_tile: !ttcore.tile<32x32, bf16>, %out_tile: !ttcore.tile<32x32, bf16>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %mx = ttl.tile_max %a_tile, %b_tile : !ttcore.tile<32x32, bf16>
    %mn = ttl.tile_min %a_tile, %b_tile : !ttcore.tile<32x32, bf16>
    %sum = ttl.tile_add %mx, %mn : !ttcore.tile<32x32, bf16>
    ttl.tile_store %sum, %out_view[%i, %j] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 2: Single max -- no copy needed when operand 0 has one in-place consumer
// =============================================================================
// Purpose: With only one consumer of %a (max), no copy insertion is needed.
// The in-place merge assigns max's result to the same DST as operand 0.

// CHECK-LABEL: func.func @single_max
// CHECK:           ttl.compute
// CHECK-NEXT:      ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, bf16>, %[[B:[^:]*]]: !ttcore.tile<32x32, bf16>, %[[OUT:[^:]*]]: !ttcore.tile<32x32, bf16>):
// CHECK:           %{{.*}}, %[[AT:.*]] = ttl.copy_tile %[[A]]
// CHECK:           %{{.*}}, %[[BT:.*]] = ttl.copy_tile %[[B]]
// CHECK:           %[[MX:.*]] = ttl.tile_max %[[AT]], %[[BT]]
// CHECK:           ttl.tile_store %[[MX]]
// CHECK-NEXT:      ttl.yield

func.func @single_max(%a: tensor<1x1x!ttcore.tile<32x32, bf16>>,
                      %b: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_view = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, bf16>, %b_tile: !ttcore.tile<32x32, bf16>, %out_tile: !ttcore.tile<32x32, bf16>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %mx = ttl.tile_max %a_tile, %b_tile : !ttcore.tile<32x32, bf16>
    ttl.tile_store %mx, %out_view[%i, %j] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}
