// Tests for scheduling edge cases: single-type (already sorted), single-tile,
// and multi-type chains that exercise different scheduler code paths.
//
// RUN: ttlang-opt %s --split-input-file \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst{enable-fpu-binary-ops=0}, ttl-subblock-compute-for-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s --check-prefix=CHECK

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 1: Single-type SFPU binary (already sorted path)
// =============================================================================
// Purpose: With only one compute op type (add_binary_tile), all tile ops in
// the sync region are at the same depth/category/type. The scheduler's
// is_sorted check returns true and no reordering occurs.

// CHECK-LABEL: func.func @single_type_already_sorted
// CHECK:       ttkernel.tile_regs_acquire
//
// All add_binary_tile grouped (one init):
// CHECK:       ttkernel.add_binary_tile_init
// CHECK:       ttkernel.add_binary_tile(
// CHECK-NOT:   ttkernel.add_binary_tile_init
// CHECK:       ttkernel.add_binary_tile(
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
    ttl.yield
  } -> tensor<2x1x!ttcore.tile<32x32, bf16>>

  ttl.cb_push %cb2 : <[2, 1], !ttcore.tile<32x32, bf16>, 1>

  func.return %result : tensor<2x1x!ttcore.tile<32x32, bf16>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 2: Three-type chain: copy + SFPU binary + SFPU unary
// =============================================================================
// Purpose: Exercises reordering across three scheduling categories in a single
// sync region. Two independent depth-0 operations: SFPU binary add(a,b) and
// copy_tile(c) for subsequent exp. Without scheduling, add_binary_tile appears
// before copy_tile; after scheduling, copy_tile (category 0) is grouped before
// add_binary_tile (category 5).

// CHECK-LABEL: func.func @copy_before_sfpu_reorder
// CHECK:       ttkernel.tile_regs_acquire
//
// copy_tile ops grouped by input CB (copy_tile), then add, then exp (in later sync region):
// CHECK:       ttkernel.copy_tile_init(
// CHECK:       ttkernel.copy_tile(
// CHECK:       ttkernel.copy_tile(
//
// SFPU binary add grouped (one init):
// CHECK:       ttkernel.add_binary_tile_init
// CHECK:       ttkernel.add_binary_tile(
// CHECK-NOT:   ttkernel.add_binary_tile_init
// CHECK:       ttkernel.add_binary_tile(
//
// SFPU unary exp grouped (one init):
// CHECK:       ttkernel.exp_tile_init
// CHECK:       ttkernel.exp_tile(
// CHECK-NOT:   ttkernel.exp_tile_init
// CHECK:       ttkernel.exp_tile(
//
// CHECK:       ttkernel.tile_regs_commit
func.func @copy_before_sfpu_reorder(
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
    // SFPU binary at depth 0: reads from CB, no copy needed
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, bf16>
    // SFPU unary: exp of c (needs copy_tile, independent of add)
    %exp = ttl.tile_exp %c_tile : !ttcore.tile<32x32, bf16>
    ttl.tile_store %sum, %rv0 : !ttcore.tile<32x32, bf16>, tensor<2x1x!ttcore.tile<32x32, bf16>>
    ttl.tile_store %exp, %rv1 : !ttcore.tile<32x32, bf16>, tensor<2x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> (tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>)

  ttl.cb_push %cb3 : <[2, 1], !ttcore.tile<32x32, bf16>, 1>
  ttl.cb_push %cb4 : <[2, 1], !ttcore.tile<32x32, bf16>, 1>

  func.return %result#0, %result#1 : tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>
}
