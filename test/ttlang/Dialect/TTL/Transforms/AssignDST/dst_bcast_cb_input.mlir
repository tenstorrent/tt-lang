// Summary: tile_bcast reads from CB (not DST), so no copy_tile for its input.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8}),canonicalize,cse)' --split-input-file | FileCheck %s

// Verify no placeholder copies remain in final IR
// CHECK-NOT: placeholder

#map = affine_map<(d0, d1) -> (d0, d1)>

// tile_bcast has TTLCBInputTileOpTrait: reads from CB directly, writes to DST.
// Unlike normal tile ops, bcast input does not require copy_tile since it reads
// from CB instead of DST. The bcast result should get dst_idx assigned.
// BcastType enum values: Col=1, Row=2, Scalar=3
// CHECK-LABEL: func.func @bcast_standalone
func.func @bcast_standalone(%a: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

// CHECK: %[[RESULT:.*]] = ttl.compute
// CHECK: ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// No copy_tile for bcast input - it reads from CB directly
// CHECK-NOT: ttl.copy_tile %[[A]]
// CHECK: %[[BCAST:.*]] = ttl.tile_bcast %[[A]], %[[OUT]] 2 : i32 {dst_idx = 0 : i32}
// CHECK-NEXT: ttl.yield %[[BCAST]] : !ttcore.tile<32x32, f32>
  %result = ttl.compute
      ins(%a_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %bcast = ttl.tile_bcast %a_tile, %out_tile 2 : i32 : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    ttl.yield %bcast : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Bcast followed by unary op. The bcast result is in DST, so subsequent unary
// ops consume the DST value. No copy_tile for bcast input.
// CHECK-LABEL: func.func @bcast_then_exp
func.func @bcast_then_exp(%a: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

// CHECK: %[[RESULT:.*]] = ttl.compute
// CHECK: ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// CHECK-NOT: ttl.copy_tile %[[A]]
// CHECK: %[[BCAST:.*]] = ttl.tile_bcast %[[A]], %[[OUT]] 2 : i32 {dst_idx = 0 : i32}
// CHECK-NEXT: %[[EXP:.*]] = ttl.tile_exp %[[BCAST]] {dst_idx = 0 : i32}
// CHECK-NEXT: ttl.yield %[[EXP]] : !ttcore.tile<32x32, f32>
  %result = ttl.compute
      ins(%a_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %bcast = ttl.tile_bcast %a_tile, %out_tile 2 : i32 : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %bcast : !ttcore.tile<32x32, f32>
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Bcast followed by binary op with another CB input. Bcast reads from CB (no
// copy_tile), but the second binary operand needs copy_tile since tile_add
// reads from DST. This tests mixing CB-reading and DST-reading ops.
// CHECK-LABEL: func.func @bcast_then_add
func.func @bcast_then_add(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                          %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

// CHECK: %[[RESULT:.*]] = ttl.compute
// CHECK: ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[B:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// Bcast comes first - no copy_tile needed for its input (reads from CB)
// CHECK-NEXT: %[[BCAST:.*]] = ttl.tile_bcast %[[A]], %[[OUT]] 2 : i32 {dst_idx = 0 : i32}
// B needs copy_tile because tile_add reads from DST
// CHECK-NEXT: %[[LINIDX:.*]] = ttl.linearized_index
// CHECK-NEXT: %[[DTOK:.*]], %[[DTILE:.*]] = ttl.copy_tile %[[B]]
// CHECK-NEXT: %[[ADD:.*]] = ttl.tile_add %[[BCAST]], %[[DTILE]] {dst_idx = 0 : i32}
// CHECK-NEXT: ttl.yield %[[ADD]] : !ttcore.tile<32x32, f32>
  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %bcast = ttl.tile_bcast %a_tile, %out_tile 2 : i32 : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    %add = ttl.tile_add %bcast, %b_tile : !ttcore.tile<32x32, f32>
    ttl.yield %add : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
