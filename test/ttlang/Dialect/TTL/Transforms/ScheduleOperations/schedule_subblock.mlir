// FPU path (default): add uses add_tiles (0 DST input slots), dstPerIteration=1 (tanh only).
// unrollFactor = min(4, 6) = 4. Subblock [1, 3] = 3 tiles fits in f32 capacity.
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst, ttl-subblock-compute-for-dst{subblock-sync=true}, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s --check-prefix=FPU

// SFPU path: add uses copy_tile + add_binary_tile (2 DST input slots), dstPerIteration=2.
// unrollFactor = min(2, 6) = 2. Subblock [2, 1] = 2 tiles.
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst{enable-fpu-binary-ops=0}, ttl-subblock-compute-for-dst{subblock-sync=true}, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s --check-prefix=SFPU

// auto-sync disabled: reserve/push stays at outer level.
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst{enable-fpu-binary-ops=0}, ttl-subblock-compute-for-dst{subblock-sync=false}, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s --check-prefix=MANUAL

// =============================================================================
// FPU path: 3 tiles per subblock, outer loop over 2 rows.
// After scheduling: add_tiles grouped, then tanhs grouped.
// =============================================================================
// FPU-LABEL: func.func @f32_subblock_scheduling
// FPU-DAG:       %[[C6_I32:.*]] = arith.constant 6 : i32
// FPU-DAG:       %[[C3_I32:.*]] = arith.constant 3 : i32
// FPU-DAG:       %[[C1:.*]] = arith.constant 1 : index
// FPU-DAG:       %[[C2:.*]] = arith.constant 2 : index
// FPU-DAG:       %[[C0:.*]] = arith.constant 0 : index
// FPU-DAG:       %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// FPU-DAG:       %[[CB_OUT:.*]] = ttkernel.get_compile_time_arg_val(2)
// FPU-DAG:       %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// FPU:           ttkernel.cb_wait_front(%[[CB0]], %[[C6_I32]])
// FPU-NEXT:      ttkernel.cb_wait_front(%[[CB1]], %[[C6_I32]])
// FPU-NEXT:      ttkernel.cb_reserve_back(%[[CB_OUT]], %[[C6_I32]])
// FPU-NEXT:      ttkernel.binary_op_init_common(%[[CB0]], %[[CB1]], %[[CB_OUT]])
// FPU-NEXT:      scf.for %[[IV:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// Per-subblock cb_reserve inside loop (outermost dim subblocked).
// FPU-NEXT:        ttkernel.cb_reserve_back(%[[CB_OUT]], %[[C3_I32]])
// FPU-NEXT:        ttkernel.tile_regs_acquire()
// Grouped within subblock: all add_tiles, then all tanh_tiles
// FPU-NEXT:        %[[IDX0:.*]] = affine.linearize_index [%[[IV]], %[[C0]]] by (2, 3)
// FPU-NEXT:        ttkernel.add_tiles_init(%[[CB0]], %[[CB1]])
// FPU-NEXT:        ttkernel.add_tiles(%[[CB0]], %[[CB1]], %[[IDX0]], %[[IDX0]], %[[C0]])
// FPU-NEXT:        %[[IDX1:.*]] = affine.linearize_index [%[[IV]], %[[C1]]] by (2, 3)
// FPU-NEXT:        ttkernel.add_tiles(%[[CB0]], %[[CB1]], %[[IDX1]], %[[IDX1]], %[[C1]])
// FPU-NEXT:        %[[IDX2:.*]] = affine.linearize_index [%[[IV]], %[[C2]]] by (2, 3)
// FPU-NEXT:        ttkernel.add_tiles(%[[CB0]], %[[CB1]], %[[IDX2]], %[[IDX2]], %[[C2]])
// FPU-NEXT:        ttkernel.tanh_tile_init()
// FPU-NEXT:        ttkernel.tanh_tile(%[[C0]])
// FPU-NEXT:        ttkernel.tanh_tile(%[[C1]])
// FPU-NEXT:        ttkernel.tanh_tile(%[[C2]])
// FPU-NEXT:        ttkernel.tile_regs_commit()
// FPU-NEXT:        ttkernel.tile_regs_wait()
// Pack phase: pack_tile after wait, using local subblock indices.
// FPU-NEXT:        ttkernel.pack_tile(%[[C0]], %[[CB_OUT]], %[[C0]], true)
// FPU-NEXT:        ttkernel.pack_tile(%[[C1]], %[[CB_OUT]], %[[C1]], true)
// FPU-NEXT:        ttkernel.pack_tile(%[[C2]], %[[CB_OUT]], %[[C2]], true)
// FPU-NEXT:        ttkernel.tile_regs_release()
// Per-subblock cb_push inside loop.
// FPU-NEXT:        ttkernel.cb_push_back(%[[CB_OUT]], %[[C3_I32]])
// FPU-NEXT:      } {ttl.subblock_dim = 0 : index, ttl.subblock_loop_stride = 3 : index}
// FPU-NOT: ttkernel.copy_tile
// FPU-NOT: ttkernel.add_binary_tile

// =============================================================================
// SFPU path: 2 tiles per subblock, outer loop covers 3 iterations.
// After scheduling: copies grouped by CB, then adds, then tanhs.
// 2 tiles * 2 copies = 4 copy_tile ops per sync region (within f32 capacity).
// =============================================================================
// SFPU-LABEL: func.func @f32_subblock_scheduling
// SFPU-DAG:   %[[SC0:.*]] = arith.constant 0 : index
// SFPU-DAG:   %[[SC1:.*]] = arith.constant 1 : index
// SFPU-DAG:   %[[SC2:.*]] = arith.constant 2 : index
// SFPU-DAG:   %[[SC3:.*]] = arith.constant 3 : index
// SFPU: ttkernel.init_sfpu
// SFPU: scf.for %[[IV:.*]] = %[[SC0]] to %[[SC3]] step %[[SC1]]
// SFPU:   ttkernel.tile_regs_acquire
// Grouped within subblock: copies from CB0 for both tiles, copies from CB1,
// then adds, then tanhs
// SFPU:       ttkernel.copy_tile_init(
// SFPU-NEXT:  ttkernel.copy_tile(
// SFPU:       ttkernel.copy_tile(
// SFPU:       ttkernel.copy_tile_init(
// SFPU-NEXT:  ttkernel.copy_tile(
// SFPU:       ttkernel.copy_tile(
// SFPU:       ttkernel.add_binary_tile_init
// SFPU-NEXT:  ttkernel.add_binary_tile(
// SFPU-NEXT:  ttkernel.add_binary_tile(
// SFPU-NEXT:  ttkernel.tanh_tile_init
// SFPU-NEXT:  ttkernel.tanh_tile(
// SFPU-NEXT:  ttkernel.tanh_tile(
// Pack phase
// SFPU-NEXT:  ttkernel.tile_regs_commit
// SFPU-NEXT:  ttkernel.tile_regs_wait
// SFPU:       ttkernel.pack_tile(
// SFPU:       ttkernel.pack_tile(
// SFPU:       ttkernel.tile_regs_release
// SFPU:   } {ttl.subblock_dim = 1 : index, ttl.subblock_loop_stride = 1 : index}
// SFPU-NOT:   ttkernel.add_tiles

// =============================================================================
// Manual sync: reserve/push at outer level, not per-subblock.
// =============================================================================
// MANUAL-LABEL: func.func @f32_subblock_scheduling
// MANUAL-DAG: %[[MC6:.*]] = arith.constant 6 : i32
// MANUAL: ttkernel.cb_reserve_back(%{{.*}}, %[[MC6]])
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
// MANUAL: ttkernel.cb_push_back(%{{.*}}, %[[MC6]])

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @f32_subblock_scheduling()
    attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [],
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>
  %lhs_ready = ttl.cb_wait %cb0 : <[2, 3], !ttcore.tile<32x32, f32>, 2> -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %lhs = ttl.attach_cb %lhs_ready, %cb0 : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %rhs_ready = ttl.cb_wait %cb2 : <[2, 3], !ttcore.tile<32x32, f32>, 2> -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %rhs = ttl.attach_cb %rhs_ready, %cb2 : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %out_view = ttl.cb_reserve %cb1 : <[2, 3], !ttcore.tile<32x32, f32>, 2> -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %out = ttl.attach_cb %out_view, %cb1 : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %empty = tensor.empty() : tensor<2x3x!ttcore.tile<32x32, f32>>
  %out_cb = ttl.attach_cb %empty, %cb1 : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%lhs, %rhs : tensor<2x3x!ttcore.tile<32x32, f32>>,
                        tensor<2x3x!ttcore.tile<32x32, f32>>)
      outs(%out_cb : tensor<2x3x!ttcore.tile<32x32, f32>>)
      {fp32_dest_acc_en = true,
       indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%lhs_tile: !ttcore.tile<32x32, f32>,
       %rhs_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %sum = ttl.tile_add %lhs_tile, %rhs_tile : !ttcore.tile<32x32, f32>
    %tanh = ttl.tile_tanh %sum : !ttcore.tile<32x32, f32>
    ttl.tile_store %tanh, %out_view[%i, %j] : !ttcore.tile<32x32, f32>, tensor<2x3x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x3x!ttcore.tile<32x32, f32>>
  ttl.cb_push %cb1 : <[2, 3], !ttcore.tile<32x32, f32>, 2>
  ttl.cb_pop %cb2 : <[2, 3], !ttcore.tile<32x32, f32>, 2>
  ttl.cb_pop %cb0 : <[2, 3], !ttcore.tile<32x32, f32>, 2>
  return
}
