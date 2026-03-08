// Summary: tile_store {acc = true} lowers to zero-init + add_binary_tile
// (compute phase) + deferred pack_tile (pack phase) through the full pipeline.
//
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @acc_store_pipeline
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:       ttkernel.tile_regs_acquire
// Zero-init the accumulator in DST[1].
// CHECK-NEXT:  ttkernel.fill_tile_init
// CHECK-NEXT:  ttkernel.fill_tile(%[[C1]], %[[ZERO]])
// Compute a + b into DST[0] (FPU binary path).
// CHECK:       ttkernel.add_tiles_init
// CHECK:       ttkernel.add_tiles
// Accumulate: DST[1] += DST[0].
// CHECK:       ttkernel.add_binary_tile_init
// CHECK-NEXT:  ttkernel.add_binary_tile(%[[C0]], %[[C1]], %[[C1]])
// CHECK:       ttkernel.tile_regs_commit
// Deferred pack from accumulator DST[1] to output CB.
// CHECK:       ttkernel.tile_regs_wait
// CHECK-NEXT:  ttkernel.pack_tile(%[[C1]],
// CHECK:       ttkernel.tile_regs_release

func.func @acc_store_pipeline(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                               %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %out_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
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
    ttl.tile_store %sum, %out_view {acc = true} : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
