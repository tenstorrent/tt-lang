// Verifies the ttkernel-combine-pack-tiles pass: consecutive pack_tile ops
// with contiguous DST and CB indices on the same DFB are combined into
// pack_tile_block. Non-contiguous, single, and interleaved cases are preserved.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttkernel-combine-pack-tiles))' --split-input-file | FileCheck %s

// 4 consecutive pack_tile ops with contiguous indices -> single pack_tile_block.

// CHECK-LABEL: func.func @four_contiguous
// CHECK: ttkernel.pack_tile_block(
// CHECK-NOT: ttkernel.pack_tile(
func.func @four_contiguous() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.pack_tile(%c1, %cb, %c1, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.pack_tile(%c2, %cb, %c2, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.pack_tile(%c3, %cb, %c3, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  return
}

// -----

// Two groups targeting different DFBs -> two separate pack_tile_block ops.

// CHECK-LABEL: func.func @two_dfbs
// CHECK: ttkernel.pack_tile_block(
// CHECK: ttkernel.pack_tile_block(
// CHECK-NOT: ttkernel.pack_tile(
func.func @two_dfbs() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %cb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ttkernel.pack_tile(%c0, %cb0, %c0, true) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.pack_tile(%c1, %cb0, %c1, true) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.pack_tile(%c0, %cb1, %c0, true) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.pack_tile(%c1, %cb1, %c1, true) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  return
}

// -----

// Non-contiguous DST indices (0, 2): not combined.

// CHECK-LABEL: func.func @non_contiguous_dst
// CHECK: ttkernel.pack_tile(
// CHECK: ttkernel.pack_tile(
// CHECK-NOT: ttkernel.pack_tile_block
func.func @non_contiguous_dst() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.pack_tile(%c2, %cb, %c1, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  return
}

// -----

// Single pack_tile: run of length 1, not combined.

// CHECK-LABEL: func.func @single_pack
// CHECK: ttkernel.pack_tile(
// CHECK-NOT: ttkernel.pack_tile_block
func.func @single_pack() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  return
}

// -----

// Intervening non-constant op (tile_regs_release) breaks the run into two
// separate groups. Both start at CB index 0, so each is combined independently.

// CHECK-LABEL: func.func @interleaved
// CHECK: ttkernel.pack_tile_block(
// CHECK: ttkernel.tile_regs_release
// CHECK: ttkernel.pack_tile_block(
// CHECK-NOT: ttkernel.pack_tile(
func.func @interleaved() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.pack_tile(%c1, %cb, %c1, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.pack_tile(%c1, %cb, %c1, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  return
}

// -----

// Non-contiguous CB indices (DST 0,1 but CB 0,2): not combined.

// CHECK-LABEL: func.func @non_contiguous_cb
// CHECK: ttkernel.pack_tile(
// CHECK: ttkernel.pack_tile(
// CHECK-NOT: ttkernel.pack_tile_block
func.func @non_contiguous_cb() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.pack_tile(%c1, %cb, %c2, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  return
}

// -----

// Non-zero CB start index: pack_tile_block always writes starting from CB
// index 0 (per op definition), so runs with non-zero first CB index cannot
// be combined.

// CHECK-LABEL: func.func @nonzero_base_no_combine
// CHECK: ttkernel.pack_tile(
// CHECK: ttkernel.pack_tile(
// CHECK-NOT: ttkernel.pack_tile_block
func.func @nonzero_base_no_combine() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  ttkernel.pack_tile(%c2, %cb, %c2, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.pack_tile(%c3, %cb, %c3, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  return
}
