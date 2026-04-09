// FPU path (default): add uses add_tiles (0 DST input slots), dstPerIteration=1 (exp only).
// All 4 tiles fit in one subblock (no outer loop). add_tiles grouped, then exp grouped.
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst, ttl-subblock-compute-for-dst{subblock-sync=true}, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s --check-prefix=FPU

// SFPU path: add uses copy_tile + add_binary_tile (dstPerIteration=2).
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst{enable-fpu-binary-ops=0}, ttl-subblock-compute-for-dst{subblock-sync=true}, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s --check-prefix=SFPU

// auto-sync disabled (default): reserve/push stays at outer level.
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst{enable-fpu-binary-ops=0}, ttl-subblock-compute-for-dst{subblock-sync=false}, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s --check-prefix=MANUAL

// Purpose: Integration test for ttl-schedule-operations with init consolidation.
// Verifies: add + exp fused compute on 2x2 grid produces grouped ops with
// one init per op group instead of interleaved per-tile inits.

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// FPU path: no copy_tile, all 4 tiles in one subblock, no loop.
// =============================================================================
// FPU-LABEL: func.func @add_exp_scheduled
// FPU-DAG: %[[C0:.*]] = arith.constant 0 : index
// FPU-DAG: %[[C1:.*]] = arith.constant 1 : index
// FPU-DAG: %[[C2:.*]] = arith.constant 2 : index
// FPU-DAG: %[[C3:.*]] = arith.constant 3 : index
// FPU-DAG: %[[C4:.*]] = arith.constant 4 : i32
// FPU-DAG: %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// FPU-DAG: %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// FPU-DAG: %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// FPU: ttkernel.cb_wait_front(%[[CB0]], %[[C4]])
// FPU: ttkernel.cb_wait_front(%[[CB1]], %[[C4]])
// FPU: ttkernel.cb_reserve_back(%[[CB2]], %[[C4]])
// FPU: ttkernel.binary_op_init_common(%[[CB0]], %[[CB1]], %[[CB2]])
// FPU: ttkernel.tile_regs_acquire
// Grouped: all add_tiles, then all exp_tiles
// FPU: ttkernel.add_tiles_init(%[[CB0]], %[[CB1]])
// FPU: ttkernel.add_tiles(%[[CB0]], %[[CB1]], %[[C0]], %[[C0]], %[[C0]])
// FPU: ttkernel.add_tiles(%[[CB0]], %[[CB1]], %[[C1]], %[[C1]], %[[C1]])
// FPU: ttkernel.add_tiles(%[[CB0]], %[[CB1]], %[[C2]], %[[C2]], %[[C2]])
// FPU: ttkernel.add_tiles(%[[CB0]], %[[CB1]], %[[C3]], %[[C3]], %[[C3]])
// FPU: ttkernel.exp_tile_init
// FPU: ttkernel.exp_tile(%[[C0]])
// FPU: ttkernel.exp_tile(%[[C1]])
// FPU: ttkernel.exp_tile(%[[C2]])
// FPU: ttkernel.exp_tile(%[[C3]])
// FPU: ttkernel.tile_regs_commit
// FPU-NEXT: ttkernel.tile_regs_wait
// Pack phase: pack_tile after wait, cb_push_back after release
// FPU: ttkernel.pack_tile(%[[C0]], %[[CB2]], %[[C0]], true)
// FPU: ttkernel.pack_tile(%[[C1]], %[[CB2]], %[[C1]], true)
// FPU: ttkernel.pack_tile(%[[C2]], %[[CB2]], %[[C2]], true)
// FPU: ttkernel.pack_tile(%[[C3]], %[[CB2]], %[[C3]], true)
// FPU: ttkernel.tile_regs_release
// FPU: ttkernel.cb_push_back(%[[CB2]], %[[C4]])
// FPU-NOT: ttkernel.copy_tile
// FPU-NOT: ttkernel.add_binary_tile

// =============================================================================
// SFPU path: copy_tile + add_binary_tile, subblocked with loop.
// =============================================================================
// SFPU-LABEL: func.func @add_exp_scheduled
// SFPU-DAG: %[[SC0:.*]] = arith.constant 0 : index
// SFPU-DAG: %[[SC1:.*]] = arith.constant 1 : index
// SFPU-DAG: %[[SC2:.*]] = arith.constant 2 : index
// SFPU-DAG: %[[SC3:.*]] = arith.constant 3 : index
// SFPU-DAG: %[[SC2I:.*]] = arith.constant 2 : i32
// SFPU-DAG: %[[SC4:.*]] = arith.constant 4 : i32
// SFPU-DAG: %[[SCB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// SFPU-DAG: %[[SCB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// SFPU-DAG: %[[SCB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// SFPU: ttkernel.cb_wait_front(%[[SCB0]], %[[SC4]])
// SFPU: ttkernel.cb_wait_front(%[[SCB1]], %[[SC4]])
// SFPU: ttkernel.cb_reserve_back(%[[SCB2]], %[[SC4]])
// SFPU: ttkernel.init_sfpu(%[[SCB0]], %[[SCB2]])
// SFPU: scf.for %[[IV:.*]] = %[[SC0]] to %[[SC2]] step %[[SC1]]
// Per-subblock cb_reserve inside loop (outermost dim subblocked).
// SFPU:   ttkernel.cb_reserve_back(%[[SCB2]], %[[SC2I]])
// SFPU:   ttkernel.tile_regs_acquire
// SFPU:   %[[BASE:.*]] = affine.linearize_index [%[[IV]], %[[SC0]]] by (2, 2)
// Grouped within subblock: copies from CB0 for both tiles, copies from CB1,
// then adds, then exps
// SFPU:   ttkernel.copy_tile_init(%[[SCB0]])
// SFPU:   ttkernel.copy_tile(%[[SCB0]], %[[BASE]], %[[SC0]])
// SFPU:   %[[BASE1:.*]] = affine.linearize_index [%[[IV]], %[[SC1]]] by (2, 2)
// SFPU:   ttkernel.copy_tile(%[[SCB0]], %[[BASE1]], %[[SC2]])
// SFPU:   ttkernel.copy_tile_init(%[[SCB1]])
// SFPU:   ttkernel.copy_tile(%[[SCB1]], %[[BASE]], %[[SC1]])
// SFPU:   ttkernel.copy_tile(%[[SCB1]], %[[BASE1]], %[[SC3]])
// SFPU:   ttkernel.add_binary_tile_init
// SFPU:   ttkernel.add_binary_tile(%[[SC0]], %[[SC1]], %[[SC0]])
// SFPU:   ttkernel.add_binary_tile(%[[SC2]], %[[SC3]], %[[SC2]])
// SFPU:   ttkernel.exp_tile_init
// SFPU:   ttkernel.exp_tile(%[[SC0]])
// SFPU:   ttkernel.exp_tile(%[[SC2]])
// SFPU:   ttkernel.tile_regs_commit
// SFPU-NEXT: ttkernel.tile_regs_wait
// Pack phase: pack_tile after wait, using local subblock indices
// SFPU:   ttkernel.pack_tile(%[[SC0]], %[[SCB2]], %[[SC0]], true)
// SFPU:   ttkernel.pack_tile(%[[SC2]], %[[SCB2]], %[[SC1]], true)
// SFPU:   ttkernel.tile_regs_release
// Per-subblock cb_push inside loop.
// SFPU:   ttkernel.cb_push_back(%[[SCB2]], %[[SC2I]])
// SFPU: } {ttl.subblock_dim = 0 : index, ttl.subblock_loop_stride = 2 : index}
// SFPU-NOT: ttkernel.add_tiles

// =============================================================================
// Manual sync (auto-sync disabled): reserve/push at outer level, pack_tile
// uses linearized global indices (not local subblock indices).
// =============================================================================
// MANUAL-LABEL: func.func @add_exp_scheduled
// MANUAL-DAG: %[[MC4:.*]] = arith.constant 4 : i32
// Reserve with full block count before the loop (not per-subblock):
// MANUAL: ttkernel.cb_reserve_back(%{{.*}}, %[[MC4]])
// MANUAL: scf.for
// No per-subblock reserve/push inside the loop:
// MANUAL-NOT: ttkernel.cb_reserve_back
// MANUAL-NOT: ttkernel.cb_push_back
// Pack phase inside loop: pack after wait, release at end
// MANUAL: ttkernel.tile_regs_wait
// MANUAL: ttkernel.pack_tile
// MANUAL: ttkernel.pack_tile
// MANUAL: ttkernel.tile_regs_release
// MANUAL: }
// Push with full block count after the loop:
// MANUAL: ttkernel.cb_push_back(%{{.*}}, %[[MC4]])

func.func @add_exp_scheduled(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                              %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %output = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>

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
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %c0 = arith.constant 0 : index
    %sum = ttl.tile_add %a_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %sum into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %result_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  ttl.cb_push %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
