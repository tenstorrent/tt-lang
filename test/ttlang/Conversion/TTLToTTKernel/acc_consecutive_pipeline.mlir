// Summary: Two consecutive acc stores to the same view through the full
// pipeline produce: fill_tile + add_binary (first) + add_binary (second)
// + pack_tile. FormAccumulationGroups marks both computes with the same
// group ID, and sync wraps the entire group.
//
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-form-accumulation-groups, ttl-set-compute-kernel-config, ttl-assign-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @acc_consecutive
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
//
// CHECK:       ttkernel.tile_regs_acquire
// Zero-init accumulator.
// CHECK:       ttkernel.fill_tile_init
// CHECK-NEXT:  ttkernel.fill_tile(%[[C1]], %[[ZERO]])
// First store: copy a -> DST[0], acc += DST[0].
// CHECK:       ttkernel.copy_tile
// CHECK:       ttkernel.add_binary_tile_init
// CHECK-NEXT:  ttkernel.add_binary_tile(%[[C0]], %[[C1]], %[[C1]])
// Second store: copy b -> DST[0], acc += DST[0].
// CHECK:       ttkernel.copy_tile
// CHECK:       ttkernel.add_binary_tile_init
// CHECK-NEXT:  ttkernel.add_binary_tile(%[[C0]], %[[C1]], %[[C1]])
// Deferred pack.
// CHECK:       ttkernel.tile_regs_commit
// CHECK:       ttkernel.tile_regs_wait
// CHECK-NEXT:  ttkernel.pack_tile(%[[C1]],
// CHECK:       ttkernel.tile_regs_release

func.func @acc_consecutive(%a: tensor<1x1x!ttcore.tile<32x32, f32>>,
                            %b: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %out_view = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %r0 = ttl.compute
      ins(%a_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i0 = ttl.iter_index 0 : index
    %j0 = ttl.iter_index 1 : index
    ttl.tile_store %a_tile, %out_view[%i0, %j0] {acc = true} : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %r1 = ttl.compute
      ins(%b_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i1 = ttl.iter_index 0 : index
    %j1 = ttl.iter_index 1 : index
    ttl.tile_store %b_tile, %out_view[%i1, %j1] {acc = true} : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  func.return %r1 : tensor<1x1x!ttcore.tile<32x32, f32>>
}
