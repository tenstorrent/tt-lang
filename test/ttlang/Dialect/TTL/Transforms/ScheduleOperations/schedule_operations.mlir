// FPU path (default): add uses add_tiles (0 DST input slots), dstPerIteration=1 (exp only).
// All 4 tiles fit in one subblock (no outer loop). add_tiles grouped, then exp grouped.
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst, ttl-subblock-compute-for-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s --check-prefix=FPU

// SFPU path: add uses copy_tile + add_binary_tile (dstPerIteration=2).
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst{enable-fpu-binary-ops=0}, ttl-subblock-compute-for-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s --check-prefix=SFPU

// Purpose: Integration test for ttl-schedule-operations with init consolidation.
// Verifies: add + exp fused compute on 2x2 grid produces grouped ops with
// one init per op group instead of interleaved per-tile inits.

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// FPU path: no copy_tile, all 4 tiles in one subblock, no loop.
// =============================================================================
// FPU-LABEL: func.func @add_exp_scheduled
// FPU: ttkernel.binary_op_init_common
// FPU: ttkernel.tile_regs_acquire
// All add_tiles grouped (one init):
// FPU: ttkernel.add_tiles_init
// FPU: ttkernel.add_tiles(
// FPU-NOT: ttkernel.add_tiles_init
// FPU: ttkernel.add_tiles(
// FPU: ttkernel.add_tiles(
// FPU: ttkernel.add_tiles(
// All exp_tiles grouped (one init):
// FPU: ttkernel.exp_tile_init
// FPU: ttkernel.exp_tile(
// FPU-NOT: ttkernel.exp_tile_init
// FPU: ttkernel.exp_tile(
// FPU: ttkernel.exp_tile(
// FPU: ttkernel.exp_tile(
// FPU: ttkernel.tile_regs_commit
// FPU-NOT: ttkernel.copy_tile
// FPU-NOT: ttkernel.add_binary_tile

// =============================================================================
// SFPU path: copy_tile + add_binary_tile, subblocked with loop.
// =============================================================================
// SFPU-LABEL: func.func @add_exp_scheduled
// SFPU:       scf.for
// SFPU:       ttkernel.tile_regs_acquire
// Copy tiles grouped by source CB (one init per CB):
// SFPU:       ttkernel.copy_tile_init(
// SFPU:       ttkernel.copy_tile(
// SFPU-NOT:   ttkernel.copy_tile_init
// SFPU:       ttkernel.copy_tile(
// SFPU:       ttkernel.copy_tile_init(
// SFPU:       ttkernel.copy_tile(
// SFPU-NOT:   ttkernel.copy_tile_init
// SFPU:       ttkernel.copy_tile(
// All add ops grouped together (one init):
// SFPU:       ttkernel.add_binary_tile_init
// SFPU:       ttkernel.add_binary_tile(
// SFPU-NOT:   ttkernel.add_binary_tile_init
// SFPU:       ttkernel.add_binary_tile(
// All exp_tiles grouped together (one init):
// SFPU:       ttkernel.exp_tile_init
// SFPU:       ttkernel.exp_tile(
// SFPU-NOT:   ttkernel.exp_tile_init
// SFPU:       ttkernel.exp_tile(
// SFPU:       ttkernel.tile_regs_commit
// SFPU-NOT:   ttkernel.add_tiles

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
