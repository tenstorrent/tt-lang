// Summary: Two expression acc stores (add_tiles + mul_tiles) through the full
// pipeline verify that DST is zeroed before each FPU binary op. mul_tiles
// accumulates (DST[idx] += result), so without zeroing, stale values from a
// prior computation in the same sync region corrupt the result.
//
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-form-accumulation-groups, ttl-set-compute-kernel-config, ttl-assign-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @acc_expression
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
//
// CHECK:       ttkernel.tile_regs_acquire
// Zero-init accumulator in DST[1].
// CHECK:       ttkernel.fill_tile_init
// CHECK:       ttkernel.fill_tile(%[[C1]], %[[ZERO]])
// Zero expression DST[0] before first FPU op.
// CHECK:       ttkernel.fill_tile_init
// CHECK-NEXT:  ttkernel.fill_tile(%[[C0]], %[[ZERO]])
// First store: add_tiles (FPU, overwrites DST[0]).
// CHECK:       ttkernel.add_tiles_init
// CHECK-NEXT:  ttkernel.add_tiles({{.*}}, {{.*}}, %[[C0]], %[[C0]], %[[C0]])
// Accumulate: DST[1] += DST[0].
// CHECK:       ttkernel.add_binary_tile_init
// CHECK-NEXT:  ttkernel.add_binary_tile(%[[C0]], %[[C1]], %[[C1]])
// Zero expression DST[0] before second FPU op.
// CHECK:       ttkernel.fill_tile_init
// CHECK-NEXT:  ttkernel.fill_tile(%[[C0]], %[[ZERO]])
// Second store: mul_tiles (FPU, accumulates DST[0] += a*b).
// With zeroing: DST[0] = 0 + a*b = a*b. Without: DST[0] = (a+b) + a*b.
// CHECK:       ttkernel.mul_tiles_init
// CHECK-NEXT:  ttkernel.mul_tiles({{.*}}, {{.*}}, %[[C0]], %[[C0]], %[[C0]])
// Accumulate: DST[1] += DST[0].
// CHECK:       ttkernel.add_binary_tile_init
// CHECK-NEXT:  ttkernel.add_binary_tile(%[[C0]], %[[C1]], %[[C1]])
// Deferred pack from accumulator DST[1] to output CB.
// CHECK:       ttkernel.tile_regs_commit
// CHECK:       ttkernel.tile_regs_wait
// CHECK-NEXT:  ttkernel.pack_tile(%[[C1]],
// CHECK:       ttkernel.tile_regs_release

func.func @acc_expression(%a: tensor<1x1x!ttcore.tile<32x32, f32>>,
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

  // Compute 1: add_tiles (FPU binary) → tile_store {acc=true}
  %r0 = ttl.compute
      ins(%a_cb, %b_cb : tensor<1x1x!ttcore.tile<32x32, f32>>,
                         tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i0 = ttl.iter_index 0 : index
    %j0 = ttl.iter_index 1 : index
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %sum, %out_view[%i0, %j0] {acc = true} : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // Compute 2: mul_tiles (FPU binary) → tile_store {acc=true}
  %r1 = ttl.compute
      ins(%a_cb, %b_cb : tensor<1x1x!ttcore.tile<32x32, f32>>,
                         tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i1 = ttl.iter_index 0 : index
    %j1 = ttl.iter_index 1 : index
    %prod = ttl.tile_mul %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %prod, %out_view[%i1, %j1] {acc = true} : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  func.return %r1 : tensor<1x1x!ttcore.tile<32x32, f32>>
}
