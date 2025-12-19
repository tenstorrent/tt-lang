// Summary: ensure ttl.acquire_dst is inserted ahead of DST copies in ttl.compute.
// RUN: ttlang-opt %s --ttl-tile-and-assign-dst --ttl-insert-tile-regs-sync --canonicalize --cse --split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @acquire_insert
// Purpose: verify tile_regs_acquire precedes copy_tile ops, tile_regs_commit is
// inserted before yield, and wait/release follow the compute op. Tile ops
// consume copied tiles.
// CHECK: %[[RES:.*]] = ttl.compute
// CHECK: ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[B:.*]]: !ttcore.tile<32x32, f32>, %[[O:.*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:   ttl.tile_regs_acquire
// CHECK:   %[[DTOK0:.*]], %[[DTILE0:.*]] = ttl.copy_tile %[[A]], %{{.*}}, %{{.*}} : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK:   %[[DTOK1:.*]], %[[DTILE1:.*]] = ttl.copy_tile %[[B]], %{{.*}}, %{{.*}} : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK:   %[[ADD:.*]] = ttl.tile_add %[[DTILE0]], %[[DTILE1]] : !ttcore.tile<32x32, f32>
// CHECK:   ttl.tile_regs_commit
// CHECK:   ttl.yield %[[ADD]] : !ttcore.tile<32x32, f32>
// CHECK: } -> tensor<2x2x!ttcore.tile<32x32, f32>>
// CHECK: ttl.tile_regs_wait
// CHECK: ttl.tile_regs_release
// CHECK: return %[[RES]]
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
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Re-declare map for split input.
#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @acquire_two_computes
// Purpose: ensure wait/release are emitted per compute, and commits sit before yields.
// CHECK: %[[R0:.*]] = ttl.compute
// CHECK: ttl.tile_regs_acquire
// CHECK: ttl.tile_regs_commit
// CHECK: } -> tensor<2x2x!ttcore.tile<32x32, f32>>
// CHECK: ttl.tile_regs_wait
// CHECK: ttl.tile_regs_release
// CHECK: %[[R1:.*]] = ttl.compute
// CHECK: ttl.tile_regs_acquire
// CHECK: ttl.tile_regs_commit
// CHECK: } -> tensor<2x2x!ttcore.tile<32x32, f32>>
// CHECK: ttl.tile_regs_wait
// CHECK: ttl.tile_regs_release
// CHECK: return %[[R1]]
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
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %r1 : tensor<2x2x!ttcore.tile<32x32, f32>>
}
