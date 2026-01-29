// Tests for init_sfpu CB selection when multiple inputs have different types.
// The pass should use the CB that is actually extracted in the loop body,
// not just the first input CB from the attribute list.
// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(ttl-lower-to-loops,ttl-insert-tile-regs-sync))' | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// Test: First input CB (f32) is unused, second input CB (bf16) is actually used.
// init_sfpu should use CB1 (bf16), not CB0 (f32).
// CHECK-LABEL: func.func @use_second_input_only
// CHECK:         %[[CB0:.*]] = ttl.bind_cb{cb_index = 0{{.*}}} : <[2, 2], !ttcore.tile<32x32, f32>, 1>
// CHECK:         %[[CB1:.*]] = ttl.bind_cb{cb_index = 1{{.*}}} : <[2, 2], !ttcore.tile<32x32, bf16>, 1>
// CHECK:         %[[CB2:.*]] = ttl.bind_cb{cb_index = 2{{.*}}} : <[2, 2], !ttcore.tile<32x32, bf16>, 1>
// CHECK:         ttl.attach_cb %arg0, %[[CB0]]
// CHECK:         ttl.attach_cb %arg1, %[[CB1]]
// CHECK:         ttl.attach_cb {{.*}}, %[[CB2]]
// init_sfpu should use CB1 (bf16) for input, not CB0 (f32)
// CHECK:         ttl.init_sfpu(%[[CB1]], %[[CB2]])
// CHECK-NOT:     ttl.init_sfpu(%[[CB0]]
// CHECK:         scf.for
func.func @use_second_input_only(%arg0: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                  %arg1: tensor<2x2x!ttcore.tile<32x32, bf16>>)
    -> tensor<2x2x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index

  // CB0: f32 type - will NOT be used in compute body
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  // CB1: bf16 type - will be used in compute
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>
  // CB2: output
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>

  %a_attached = ttl.attach_cb %arg0, %cb0
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_attached = ttl.attach_cb %arg1, %cb1
      : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>
  %out_attached = ttl.attach_cb %init, %cb2
      : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  // Reserve output CB for stores
  %out_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, bf16>, 1>
      -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  // Compute that only uses %b_attached (CB1, bf16), not %a_attached (CB0, f32).
  %result = ttl.compute
      ins(%a_attached, %b_attached : tensor<2x2x!ttcore.tile<32x32, f32>>,
                                     tensor<2x2x!ttcore.tile<32x32, bf16>>)
      outs(%out_attached : tensor<2x2x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%unused_f32: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, bf16>,
       %out_tile: !ttcore.tile<32x32, bf16>):
    // Only use the bf16 input, ignore the f32 input
    %tok, %tile = ttl.copy_tile %b_tile, %c0, %c0
        : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
    // Explicit store required
    ttl.store %tile, %out_view : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.yield %tile : !ttcore.tile<32x32, bf16>
  } -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, bf16>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Test: Both inputs used, but with different order in the compute body.
// init_sfpu should still work correctly when both CBs are extracted.
// CHECK-LABEL: func.func @both_inputs_used
// CHECK:         %[[CB0:.*]] = ttl.bind_cb{cb_index = 0{{.*}}} : <[2, 2], !ttcore.tile<32x32, bf16>, 1>
// CHECK:         %[[CB1:.*]] = ttl.bind_cb{cb_index = 1{{.*}}} : <[2, 2], !ttcore.tile<32x32, bf16>, 1>
// CHECK:         %[[CB2:.*]] = ttl.bind_cb{cb_index = 2{{.*}}} : <[2, 2], !ttcore.tile<32x32, bf16>, 1>
// init_sfpu can use either CB0 or CB1 since both are same type and shape
// CHECK:         ttl.init_sfpu(%[[CB0]], %[[CB2]])
// CHECK:         scf.for
func.func @both_inputs_used(%arg0: tensor<2x2x!ttcore.tile<32x32, bf16>>,
                             %arg1: tensor<2x2x!ttcore.tile<32x32, bf16>>)
    -> tensor<2x2x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>

  %a_attached = ttl.attach_cb %arg0, %cb0
      : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %b_attached = ttl.attach_cb %arg1, %cb1
      : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>
  %out_attached = ttl.attach_cb %init, %cb2
      : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  // Reserve output CB for stores
  %out_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, bf16>, 1>
      -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  // Both inputs are used - add them together
  %result = ttl.compute
      ins(%a_attached, %b_attached : tensor<2x2x!ttcore.tile<32x32, bf16>>,
                                     tensor<2x2x!ttcore.tile<32x32, bf16>>)
      outs(%out_attached : tensor<2x2x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, bf16>,
       %b_tile: !ttcore.tile<32x32, bf16>,
       %out_tile: !ttcore.tile<32x32, bf16>):
    %tok0, %tile0 = ttl.copy_tile %a_tile, %c0, %c0
        : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
    %tok1, %tile1 = ttl.copy_tile %b_tile, %c0, %c0
        : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
    %sum = ttl.tile_add %tile0, %tile1 : !ttcore.tile<32x32, bf16>
    // Explicit store required
    ttl.store %sum, %out_view : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.yield %sum : !ttcore.tile<32x32, bf16>
  } -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, bf16>>
}

// -----

#map_full = affine_map<(d0, d1) -> (d0, d1)>
#map_bcast = affine_map<(d0, d1) -> (0, d1)>

// Test: Broadcast scenario - small input CB (1x2) and full-size input CB (2x2).
// init_sfpu should prefer the CB that matches output shape (2x2), not the
// broadcast CB (1x2).
// CHECK-LABEL: func.func @broadcast_prefers_full_size
// CHECK:         %[[CB_BCAST:.*]] = ttl.bind_cb{cb_index = 0{{.*}}} : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
// CHECK:         %[[CB_FULL:.*]] = ttl.bind_cb{cb_index = 1{{.*}}} : <[2, 2], !ttcore.tile<32x32, bf16>, 1>
// CHECK:         %[[CB_OUT:.*]] = ttl.bind_cb{cb_index = 2{{.*}}} : <[2, 2], !ttcore.tile<32x32, bf16>, 1>
// init_sfpu should use CB_FULL (2x2) because it matches output shape, not CB_BCAST (1x2)
// CHECK:         ttl.init_sfpu(%[[CB_FULL]], %[[CB_OUT]])
// CHECK-NOT:     ttl.init_sfpu(%[[CB_BCAST]]
// CHECK:         scf.for
func.func @broadcast_prefers_full_size(%bcast_input: tensor<1x2x!ttcore.tile<32x32, bf16>>,
                                        %full_input: tensor<2x2x!ttcore.tile<32x32, bf16>>)
    -> tensor<2x2x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index

  // CB0: broadcast input (1x2) - smaller shape
  %cb_bcast = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
  // CB1: full-size input (2x2) - matches output shape
  %cb_full = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>
  // CB2: output (2x2)
  %cb_out = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>

  %bcast_attached = ttl.attach_cb %bcast_input, %cb_bcast
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %full_attached = ttl.attach_cb %full_input, %cb_full
      : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>
  %out_attached = ttl.attach_cb %init, %cb_out
      : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  // Reserve output CB for stores
  %out_view = ttl.cb_reserve %cb_out : <[2, 2], !ttcore.tile<32x32, bf16>, 1>
      -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  // Broadcast add: bcast_input is broadcast across rows
  %result = ttl.compute
      ins(%bcast_attached, %full_attached : tensor<1x2x!ttcore.tile<32x32, bf16>>,
                                            tensor<2x2x!ttcore.tile<32x32, bf16>>)
      outs(%out_attached : tensor<2x2x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map_bcast, #map_full, #map_full],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%bcast_tile: !ttcore.tile<32x32, bf16>,
       %full_tile: !ttcore.tile<32x32, bf16>,
       %out_tile: !ttcore.tile<32x32, bf16>):
    %tok0, %tile0 = ttl.copy_tile %bcast_tile, %c0, %c0
        : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
    %tok1, %tile1 = ttl.copy_tile %full_tile, %c0, %c0
        : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
    %sum = ttl.tile_add %tile0, %tile1 : !ttcore.tile<32x32, bf16>
    // Explicit store required
    ttl.store %sum, %out_view : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.yield %sum : !ttcore.tile<32x32, bf16>
  } -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, bf16>>
}
