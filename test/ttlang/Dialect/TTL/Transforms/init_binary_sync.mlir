// Tests for init_binary: parse/print roundtrip and sync insertion behavior.
// Verifies init_binary parses correctly and that TTLInsertTileRegsSync does
// NOT emit init ops (init ops are handled by ttkernel-insert-inits).
//
// Parse/print roundtrip:
// RUN: ttlang-opt %s --split-input-file | FileCheck %s --check-prefix=PARSE
// Sync insertion (no init ops expected):
// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(ttl-insert-tile-regs-sync))' | FileCheck %s

// =============================================================================
// Test 1: init_binary parses and prints correctly with three CBs
// =============================================================================
// PARSE-LABEL: func.func @init_binary_basic
// PARSE:         %[[CB0:.*]] = ttl.bind_cb
// PARSE:         %[[CB1:.*]] = ttl.bind_cb
// PARSE:         %[[CB2:.*]] = ttl.bind_cb
// PARSE:         ttl.init_binary(%[[CB0]], %[[CB1]], %[[CB2]]) : <[1, 1], f32, 2>, <[1, 1], f32, 2>, <[1, 1], f32, 2>
func.func @init_binary_basic() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
  ttl.init_binary(%cb0, %cb1, %cb2) : !ttl.cb<[1, 1], f32, 2>, !ttl.cb<[1, 1], f32, 2>, !ttl.cb<[1, 1], f32, 2>
  func.return
}

// -----

// =============================================================================
// Test 2: init_binary with different CB element types
// =============================================================================
// PARSE-LABEL: func.func @init_binary_different_types
// PARSE:         %[[ICB0:.*]] = ttl.bind_cb{{.*}}cb_index = 0
// PARSE:         %[[ICB1:.*]] = ttl.bind_cb{{.*}}cb_index = 1
// PARSE:         %[[OCB:.*]] = ttl.bind_cb{{.*}}cb_index = 16
// PARSE:         ttl.init_binary(%[[ICB0]], %[[ICB1]], %[[OCB]]) : <[1, 1], !ttcore.tile<32x32, bf16>, 1>, <[1, 1], !ttcore.tile<32x32, bf16>, 1>, <[1, 1], !ttcore.tile<32x32, f32>, 1>
func.func @init_binary_different_types() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %icb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %icb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %ocb = ttl.bind_cb {cb_index = 16, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  ttl.init_binary(%icb0, %icb1, %ocb) : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  func.return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 3: FPU binary add -> sync pass does NOT emit init ops
// =============================================================================
// CHECK-LABEL: func.func @fpu_binary_no_init_from_sync
// CHECK-NOT:     ttl.init_binary
// CHECK-NOT:     ttl.init_sfpu
// CHECK:         ttl.compute
func.func @fpu_binary_no_init_from_sync(
    %a: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %b: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %add = ttl.tile_add %a_tile, %b_tile {dst_idx = 0 : i32, ttl.fpu_binary} : !ttcore.tile<32x32, f32>
    ttl.yield %add : !ttcore.tile<32x32, f32>
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 4: SFPU-only (exp) -> sync pass does NOT emit init ops
// =============================================================================
// CHECK-LABEL: func.func @sfpu_only_no_init_from_sync
// CHECK-NOT:     ttl.init_sfpu
// CHECK-NOT:     ttl.init_binary
// CHECK:         ttl.compute
func.func @sfpu_only_no_init_from_sync(
    %a: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute
      ins(%a_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
    %tok, %tile = ttl.copy_tile %in, %c0, %c0 : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
    %exp = ttl.tile_exp %tile {dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>
    ttl.yield %exp : !ttcore.tile<32x32, bf16>
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 5: Mixed FPU + SFPU compute -> sync pass does NOT emit init ops
// =============================================================================
// CHECK-LABEL: func.func @mixed_fpu_sfpu_no_init_from_sync
// CHECK-NOT:     ttl.init_binary
// CHECK-NOT:     ttl.init_sfpu
// CHECK:         ttl.compute
func.func @mixed_fpu_sfpu_no_init_from_sync(
    %a: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %b: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    // FPU binary: both block args
    %add = ttl.tile_add %a_tile, %b_tile {dst_idx = 0 : i32, ttl.fpu_binary} : !ttcore.tile<32x32, f32>
    // SFPU unary: operates on DST result
    %exp = ttl.tile_exp %add {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 6: Binary add WITHOUT fpu_binary -> sync pass does NOT emit init ops
// =============================================================================
// This simulates the output of ttl-assign-dst{enable-fpu-binary-ops=0} where
// binary ops use copy_tile instead of FPU CB reads.
// CHECK-LABEL: func.func @sfpu_binary_add_no_init_from_sync
// CHECK-NOT:     ttl.init_sfpu
// CHECK-NOT:     ttl.init_binary
// CHECK:         ttl.compute
func.func @sfpu_binary_add_no_init_from_sync(
    %a: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %b: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    // SFPU binary: no fpu_binary attribute (as produced by enable-fpu-binary-ops=0)
    %tok_a, %tile_a = ttl.copy_tile %a_tile, %c0, %c0 {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
    %tok_b, %tile_b = ttl.copy_tile %b_tile, %c0, %c1 {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
    %add = ttl.tile_add %tile_a, %tile_b {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
    ttl.yield %add : !ttcore.tile<32x32, f32>
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<1x1x!ttcore.tile<32x32, f32>>
}
