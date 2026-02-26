// Tests for scheduling edge cases: single-type (already sorted), single-tile,
// and multi-type chains that exercise different scheduler code paths.
//
// RUN: ttlang-opt %s --split-input-file \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst, ttl-subblock-compute-for-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s --check-prefix=CHECK

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 1: Single-type FPU binary (already sorted path)
// =============================================================================
// Purpose: With only one compute op type (add_tiles), all tile ops in the sync
// region are at the same depth/category/type. The scheduler's is_sorted check
// returns true and no reordering occurs.

// CHECK-LABEL: func.func @single_type_already_sorted
// CHECK:       ttkernel.tile_regs_acquire
//
// All add_tiles grouped (one init):
// CHECK:       ttkernel.add_tiles_init(
// CHECK:       ttkernel.add_tiles(
// CHECK-NOT:   ttkernel.add_tiles_init
// CHECK:       ttkernel.add_tiles(
//
// CHECK:       ttkernel.tile_regs_commit
func.func @single_type_already_sorted(
    %a: tensor<2x1x!ttcore.tile<32x32, bf16>>,
    %b: tensor<2x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<2x1x!ttcore.tile<32x32, bf16>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %output = tensor.empty() : tensor<2x1x!ttcore.tile<32x32, bf16>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>

  %a_ready = ttl.cb_wait %cb0 : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %b_ready = ttl.cb_wait %cb1 : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %output_cb = ttl.attach_cb %output, %cb2 : (tensor<2x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>

  %result_view = ttl.cb_reserve %cb2 : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>

  %result = ttl.compute
      ins(%a_ready, %b_ready : tensor<2x1x!ttcore.tile<32x32, bf16>>,
                               tensor<2x1x!ttcore.tile<32x32, bf16>>)
      outs(%output_cb : tensor<2x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, bf16>,
       %b_tile: !ttcore.tile<32x32, bf16>,
       %out_tile: !ttcore.tile<32x32, bf16>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, bf16>
    ttl.tile_store %sum, %result_view : !ttcore.tile<32x32, bf16>, tensor<2x1x!ttcore.tile<32x32, bf16>>
    ttl.yield %sum : !ttcore.tile<32x32, bf16>
  } -> tensor<2x1x!ttcore.tile<32x32, bf16>>

  ttl.cb_push %cb2 : <[2, 1], !ttcore.tile<32x32, bf16>, 1>

  func.return %result : tensor<2x1x!ttcore.tile<32x32, bf16>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 2: Three-type chain: copy + FPU binary + SFPU unary
// =============================================================================
// Purpose: Exercises all three main scheduling categories (CopyTile, FPUBinary,
// SFPUUnary) in a single sync region. With 2 tiles, the per-tile order is
// copy/add/exp interleaved. After scheduling: copies grouped, adds grouped,
// exps grouped.
//
// Chain: exp(a + b) where a needs copy_tile (for exp), add is FPU binary.
// Wait, that doesn't work because exp takes the add result, not a.
// Instead: abs(a) then add(abs_result, b) → copy + abs + add.
// But add has a computed operand → SFPU not FPU.
//
// Simpler: a + b (FPU) producing a result, then tanh of that result (SFPU).
// This gives us: add_tiles (FPU, depth 0), tanh_tile (SFPU, depth 1).
// With 2 tiles: add0, tanh0, add1, tanh1 → already sorted by depth.
//
// To force reordering, we need ops at the SAME depth with different categories.
// Copy + FPU at depth 0: copy(a)->dst for tanh, add(a,b)->dst for result.
// Then at depth 1: tanh(dst), add_result consumed.
//
// The real test that exercises reordering with mixed categories:
// tanh(a) + b: copy(a)->dst, tanh, copy(b)->dst, add_tiles(a,b).
// Wait, the add uses tanh result + b. If b is block arg, add is SFPU (one
// operand computed). So:
// depth 0: copy_tile(a), copy_tile(b)
// depth 1: tanh(dst0)
// depth 2: add(tanh_result, b_copy) → SFPU binary
// All at different depths → already sorted.
//
// For actual mixed-depth-0 reordering: need independent ops at depth 0.
// Two independent operations: FPU add(a,b) and copy_tile(c) for subsequent exp.
// depth 0: add_tiles(a,b) at FPUBinary=3, copy_tile(c) at CopyTile=0
// After scheduling: copy_tile(c) before add_tiles(a,b).

// CHECK-LABEL: func.func @copy_before_fpu_reorder
// CHECK:       ttkernel.tile_regs_acquire
//
// Copies first (category 0) -- all copy_tiles grouped:
// CHECK:       ttkernel.copy_tile_init(
// CHECK:       ttkernel.copy_tile(
// CHECK:       ttkernel.copy_tile(
//
// FPU binary second (category 3) -- add_tiles grouped:
// CHECK:       ttkernel.add_tiles_init(
// CHECK:       ttkernel.add_tiles(
// CHECK-NOT:   ttkernel.add_tiles_init
// CHECK:       ttkernel.add_tiles(
//
// SFPU unary third (category 4) -- exp_tiles grouped:
// CHECK:       ttkernel.exp_tile_init
// CHECK:       ttkernel.exp_tile(
// CHECK-NOT:   ttkernel.exp_tile_init
// CHECK:       ttkernel.exp_tile(
//
// CHECK:       ttkernel.tile_regs_commit
func.func @copy_before_fpu_reorder(
    %a: tensor<2x1x!ttcore.tile<32x32, bf16>>,
    %b: tensor<2x1x!ttcore.tile<32x32, bf16>>,
    %c: tensor<2x1x!ttcore.tile<32x32, bf16>>)
    -> (tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %out0 = tensor.empty() : tensor<2x1x!ttcore.tile<32x32, bf16>>
  %out1 = tensor.empty() : tensor<2x1x!ttcore.tile<32x32, bf16>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb4 = ttl.bind_cb {cb_index = 4, buffer_factor = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>

  %a_ready = ttl.cb_wait %cb0 : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %b_ready = ttl.cb_wait %cb1 : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %c_ready = ttl.cb_wait %cb2 : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %out0_cb = ttl.attach_cb %out0, %cb3 : (tensor<2x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %out1_cb = ttl.attach_cb %out1, %cb4 : (tensor<2x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>

  %rv0 = ttl.cb_reserve %cb3 : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %rv1 = ttl.cb_reserve %cb4 : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>

  %result:2 = ttl.compute
      ins(%a_ready, %b_ready, %c_ready : tensor<2x1x!ttcore.tile<32x32, bf16>>,
                                          tensor<2x1x!ttcore.tile<32x32, bf16>>,
                                          tensor<2x1x!ttcore.tile<32x32, bf16>>)
      outs(%out0_cb, %out1_cb : tensor<2x1x!ttcore.tile<32x32, bf16>>,
                                tensor<2x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map, #map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, bf16>,
       %b_tile: !ttcore.tile<32x32, bf16>,
       %c_tile: !ttcore.tile<32x32, bf16>,
       %o0: !ttcore.tile<32x32, bf16>,
       %o1: !ttcore.tile<32x32, bf16>):
    // FPU binary at depth 0: reads from CB, no copy needed
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, bf16>
    // SFPU unary: exp of c (needs copy_tile, independent of add)
    %exp = ttl.tile_exp %c_tile : !ttcore.tile<32x32, bf16>
    ttl.tile_store %sum, %rv0 : !ttcore.tile<32x32, bf16>, tensor<2x1x!ttcore.tile<32x32, bf16>>
    ttl.tile_store %exp, %rv1 : !ttcore.tile<32x32, bf16>, tensor<2x1x!ttcore.tile<32x32, bf16>>
    ttl.yield %sum, %exp : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>
  } -> (tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>)

  ttl.cb_push %cb3 : <[2, 1], !ttcore.tile<32x32, bf16>, 1>
  ttl.cb_push %cb4 : <[2, 1], !ttcore.tile<32x32, bf16>, 1>

  func.return %result#0, %result#1 : tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>
}
