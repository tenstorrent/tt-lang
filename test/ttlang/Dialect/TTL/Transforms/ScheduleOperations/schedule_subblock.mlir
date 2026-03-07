// FPU path (default): add uses add_tiles (0 DST input slots), dstPerIteration=1 (tanh only).
// unrollFactor = min(4, 6) = 4. Subblock [1, 3] = 3 tiles fits in f32 capacity.
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst, ttl-subblock-compute-for-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s --check-prefix=FPU

// SFPU path: add uses copy_tile + add_binary_tile (2 DST input slots), dstPerIteration=2.
// unrollFactor = min(2, 6) = 2. Subblock [2, 1] = 2 tiles.
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst{enable-fpu-binary-ops=0}, ttl-subblock-compute-for-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s --check-prefix=SFPU

// =============================================================================
// FPU path: 3 tiles per subblock, outer loop over 2 rows.
// After scheduling: add_tiles grouped, then tanhs grouped.
// =============================================================================
// FPU-LABEL: func.func @f32_subblock_scheduling
// FPU: ttkernel.binary_op_init_common
// FPU: scf.for
// FPU:   ttkernel.tile_regs_acquire
// FPU:   ttkernel.add_tiles_init
// FPU:   ttkernel.add_tiles(
// FPU:   ttkernel.add_tiles(
// FPU:   ttkernel.add_tiles(
// FPU:   ttkernel.tanh_tile_init
// FPU:   ttkernel.tanh_tile(
// FPU:   ttkernel.tanh_tile(
// FPU:   ttkernel.tanh_tile(
// FPU:   ttkernel.tile_regs_commit
// FPU:   ttkernel.tile_regs_wait
// FPU:   ttkernel.pack_tile(
// FPU:   ttkernel.pack_tile(
// FPU:   ttkernel.pack_tile(
// FPU:   ttkernel.tile_regs_release
// FPU-NOT: ttkernel.copy_tile
// FPU-NOT: ttkernel.add_binary_tile

// =============================================================================
// SFPU path: 2 tiles per subblock, outer loop covers 3 iterations.
// After scheduling: copies grouped by CB, then adds, then tanhs.
// 2 tiles * 2 copies = 4 copy_tile ops per sync region (within f32 capacity).
// =============================================================================
// SFPU-LABEL: func.func @f32_subblock_scheduling
// SFPU: ttkernel.init_sfpu
// SFPU: ttkernel.tile_regs_acquire
// SFPU:       ttkernel.copy_tile_init(
// SFPU-NEXT:  ttkernel.copy_tile(
// SFPU-NEXT:  ttkernel.copy_tile(
// SFPU-NEXT:  ttkernel.copy_tile_init(
// SFPU-NEXT:  ttkernel.copy_tile(
// SFPU-NEXT:  ttkernel.copy_tile(
// SFPU-NEXT:  ttkernel.add_binary_tile_init
// SFPU-NEXT:  ttkernel.add_binary_tile(
// SFPU-NEXT:  ttkernel.add_binary_tile(
// SFPU-NEXT:  ttkernel.tanh_tile_init
// SFPU-NEXT:  ttkernel.tanh_tile(
// SFPU-NEXT:  ttkernel.tanh_tile(
// SFPU-NEXT:  ttkernel.tile_regs_commit
// SFPU-NOT:   ttkernel.add_tiles

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
    %sum = ttl.tile_add %lhs_tile, %rhs_tile : !ttcore.tile<32x32, f32>
    %tanh = ttl.tile_tanh %sum : !ttcore.tile<32x32, f32>
    ttl.tile_store %tanh, %out_view : !ttcore.tile<32x32, f32>, tensor<2x3x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x3x!ttcore.tile<32x32, f32>>
  ttl.cb_push %cb1 : <[2, 3], !ttcore.tile<32x32, f32>, 2>
  ttl.cb_pop %cb2 : <[2, 3], !ttcore.tile<32x32, f32>, 2>
  ttl.cb_pop %cb0 : <[2, 3], !ttcore.tile<32x32, f32>, 2>
  return
}
