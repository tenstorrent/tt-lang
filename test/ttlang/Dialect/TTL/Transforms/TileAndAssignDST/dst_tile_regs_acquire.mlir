// Summary: ensure ttl.acquire_dst is inserted ahead of DST copies in ttl.compute.
// RUN: ttlang-opt %s --ttl-tile-and-assign-dst --ttl-insert-tile-regs-sync --canonicalize --cse --split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: verify tile_regs_acquire wraps compute, commit/wait are inside before
// yield, stores are inserted before yield, and release follows the compute. Tile
// ops consume copied tiles.
// CHECK-LABEL:   func.func @acquire_insert
// CHECK:           %[[CB0:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2}
// CHECK:           %[[CB2:.*]] = ttl.bind_cb{cb_index = 2, buffer_factor = 2}
// CHECK:           ttl.init_sfpu(%[[CB0]], %[[CB2]])
// CHECK-NEXT:      ttl.tile_regs_acquire
// CHECK-NEXT:      %[[RES:.*]] = ttl.compute
// CHECK:           ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[B:.*]]: !ttcore.tile<32x32, f32>, %[[O:.*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:        %[[DTOK0:.*]], %[[DTILE0:.*]] = ttl.copy_tile %[[A]]
// CHECK-NEXT:        %[[DTOK1:.*]], %[[DTILE1:.*]] = ttl.copy_tile %[[B]]
// CHECK-NEXT:        %[[ADD:.*]] = ttl.tile_add %[[DTILE0]], %[[DTILE1]] {dst_idx = 0 : i32}
// CHECK-NEXT:        ttl.tile_regs_commit
// CHECK-NEXT:        ttl.tile_regs_wait
// CHECK-NEXT:        %[[V:.*]] = ttl.cb_reserve %[[CB2]]
// CHECK-NEXT:        ttl.store %[[ADD]], %[[V]]
// CHECK-NEXT:        ttl.yield %[[ADD]] : !ttcore.tile<32x32, f32>
// CHECK-NEXT:      } -> tensor<2x2x!ttcore.tile<32x32, f32>>
// CHECK-NEXT:      ttl.tile_regs_release
// CHECK-NEXT:      return %[[RES]]
func.func @acquire_insert(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                          %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                         tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %result_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %sum, %result_view : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Re-declare map for split input.
#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: ensure per-compute acquire, commit/wait before yield, store before yield, and release after.
// CHECK-LABEL:   func.func @acquire_two_computes
// CHECK:           %[[CB0:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2}
// CHECK:           %[[CB2:.*]] = ttl.bind_cb{cb_index = 2, buffer_factor = 2}
// CHECK:           ttl.init_sfpu(%[[CB0]], %[[CB2]])
// CHECK-NEXT:      ttl.tile_regs_acquire
// CHECK-NEXT:      %[[R0:.*]] = ttl.compute
// CHECK:           ^bb0
// CHECK:             %[[SUM0:.*]] = ttl.tile_add
// CHECK-NEXT:        ttl.tile_regs_commit
// CHECK-NEXT:        ttl.tile_regs_wait
// CHECK-NEXT:        %[[V0:.*]] = ttl.cb_reserve %[[CB2]]
// CHECK-NEXT:        ttl.store %[[SUM0]], %[[V0]]
// CHECK-NEXT:        ttl.yield %[[SUM0]] : !ttcore.tile<32x32, f32>
// CHECK-NEXT:      } -> tensor<2x2x!ttcore.tile<32x32, f32>>
// CHECK-NEXT:      ttl.tile_regs_release
// CHECK-NEXT:      %[[CB3:.*]] = ttl.bind_cb{cb_index = 3, buffer_factor = 2}
// CHECK-NEXT:      %[[R0CB:.*]] = ttl.attach_cb %[[R0]], %[[CB3]]
// CHECK-NEXT:      ttl.init_sfpu(%[[CB3]], %[[CB2]])
// CHECK-NEXT:      ttl.tile_regs_acquire
// CHECK-NEXT:      %[[R1:.*]] = ttl.compute
// CHECK:           ^bb0
// CHECK:             %[[SUM1:.*]] = ttl.tile_add
// CHECK-NEXT:        ttl.tile_regs_commit
// CHECK-NEXT:        ttl.tile_regs_wait
// CHECK-NEXT:        %[[V1:.*]] = ttl.cb_reserve %[[CB2]]
// CHECK-NEXT:        ttl.store %[[SUM1]], %[[V1]]
// CHECK-NEXT:        ttl.yield %[[SUM1]] : !ttcore.tile<32x32, f32>
// CHECK-NEXT:      } -> tensor<2x2x!ttcore.tile<32x32, f32>>
// CHECK-NEXT:      ttl.tile_regs_release
// CHECK-NEXT:      return %[[R1]]
func.func @acquire_two_computes(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %r0 = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                         tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %result_view0 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %sum, %result_view0 : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %r0_cb = ttl.attach_cb %r0, %cb3
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>)
      -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %r1 = ttl.compute
      ins(%r0_cb, %r0_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                           tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %result_view1 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %sum, %result_view1 : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %r1 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: op chain add->mul->exp with reg sync: acquire before compute, commit/wait inside compute, store before yield, release after.
// CHECK-LABEL:   func.func @acquire_chain_three_ops
// CHECK-SAME:      (%[[AARG:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[BARG:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[CARG:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>)
// CHECK:           %[[CB0:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2}
// CHECK:           %[[CB3:.*]] = ttl.bind_cb{cb_index = 3, buffer_factor = 2}
// CHECK:           ttl.init_sfpu(%[[CB0]], %[[CB3]])
// CHECK-NEXT:      ttl.tile_regs_acquire
// CHECK-NEXT:      %[[RES:.*]] = ttl.compute
// CHECK:           ^bb0
// CHECK:             %[[ADD:.*]] = ttl.tile_add
// CHECK-NEXT:        ttl.copy_tile
// CHECK-NEXT:        %[[MUL:.*]] = ttl.tile_mul %[[ADD]],
// CHECK-NEXT:        %[[EXP:.*]] = ttl.tile_exp %[[MUL]]
// CHECK-NEXT:        ttl.tile_regs_commit
// CHECK-NEXT:        ttl.tile_regs_wait
// CHECK-NEXT:        %[[V:.*]] = ttl.cb_reserve %[[CB3]]
// CHECK-NEXT:        ttl.store %[[EXP]], %[[V]]
// CHECK-NEXT:        ttl.yield %[[EXP]] : !ttcore.tile<32x32, f32>
// CHECK-NEXT:      } -> tensor<2x2x!ttcore.tile<32x32, f32>>
// CHECK-NEXT:      ttl.tile_regs_release
// CHECK-NEXT:      return %[[RES]]
func.func @acquire_chain_three_ops(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                   %b: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                   %c: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb, %b_cb, %c_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                                tensor<2x2x!ttcore.tile<32x32, f32>>,
                                tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %c_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %mul = ttl.tile_mul %sum, %c_tile : !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %mul : !ttcore.tile<32x32, f32>
    %result_view = ttl.cb_reserve %cb3 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %exp, %result_view : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
