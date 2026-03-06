// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst{enable-fpu-binary-ops=0}, ttl-subblock-compute-for-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s

// Purpose: Integration test for ttl-schedule-operations with init consolidation.
// Verifies: add + exp fused compute on 2x2 grid produces grouped ops with
// one init per op group instead of interleaved per-tile inits.

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @add_exp_scheduled
// CHECK:       scf.for
// CHECK:       ttkernel.tile_regs_acquire
//
// Copy tiles grouped by source CB (one init per CB):
// CHECK:       ttkernel.copy_tile_init(
// CHECK:       ttkernel.copy_tile(
// CHECK-NOT:   ttkernel.copy_tile_init
// CHECK:       ttkernel.copy_tile(
// CHECK:       ttkernel.copy_tile_init(
// CHECK:       ttkernel.copy_tile(
// CHECK-NOT:   ttkernel.copy_tile_init
// CHECK:       ttkernel.copy_tile(
//
// All add ops grouped together (one init):
// CHECK:       ttkernel.add_binary_tile_init
// CHECK:       ttkernel.add_binary_tile(
// CHECK-NOT:   ttkernel.add_binary_tile_init
// CHECK:       ttkernel.add_binary_tile(
//
// All exp_tiles grouped together (one init):
// CHECK:       ttkernel.exp_tile_init
// CHECK:       ttkernel.exp_tile(
// CHECK-NOT:   ttkernel.exp_tile_init
// CHECK:       ttkernel.exp_tile(
//
// CHECK:       ttkernel.tile_regs_commit
func.func @add_exp_scheduled(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                              %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %output = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>

  %a_ready = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_ready = ttl.cb_wait %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %output_cb = ttl.attach_cb %output, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_ready, %b_ready : tensor<2x2x!ttcore.tile<32x32, f32>>,
                               tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%output_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %sum : !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %result_view : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  ttl.cb_push %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
