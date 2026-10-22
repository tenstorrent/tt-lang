// RUN: ttlang-opt %s --split-input-file --canonicalize --cse --ttkernel-verify-hardware-config | FileCheck %s
// Summary: Declared MathInit effects keep inits and compute operations alive
// through canonicalization and CSE, and the verifier accepts configurations
// that hold on every path.

// Inits and consumers without results survive canonicalization and CSE.
// CHECK-LABEL: func.func @effects_survive_cleanup
// CHECK:       ttkernel.copy_tile_init(%[[CB:[0-9]+]])
// CHECK-NEXT:  ttkernel.copy_tile(%[[CB]],
// CHECK-NEXT:  ttkernel.copy_tile_init(%[[CB]])
// CHECK-NEXT:  ttkernel.copy_tile(%[[CB]],
func.func @effects_survive_cleanup() {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  ttkernel.copy_tile_init(%cb) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>) -> ()
  ttkernel.copy_tile(%cb, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  ttkernel.copy_tile_init(%cb) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>) -> ()
  ttkernel.copy_tile(%cb, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  func.return
}

// -----

// Consumers that require the same init share it, and DST synchronization
// preserves the MATH configuration.
// CHECK-LABEL: func.func @shared_init_across_sync
// CHECK:       ttkernel.rounding_op_tile_init
// CHECK:       ttkernel.floor_tile
// CHECK:       ttkernel.tile_regs_release
// CHECK:       ttkernel.ceil_tile
func.func @shared_init_across_sync() {
  %c0 = arith.constant 0 : index
  ttkernel.rounding_op_tile_init() : () -> ()
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.floor_tile(%c0) : (index) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.tile_regs_release() : () -> ()
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.ceil_tile(%c0) : (index) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// -----

// Bcast and reduce init keys exclude the output dataflow buffer.
// CHECK-LABEL: func.func @output_buffer_outside_key
// CHECK:       ttkernel.unary_bcast(
// CHECK:       ttkernel.reduce_tile(
func.func @output_buffer_outside_key() {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  ttkernel.unary_bcast_init(%cb0, %cb2, <col>) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>) -> ()
  ttkernel.unary_bcast(%cb0, %c0, %c0, <col>) {ttl.bcast_output_cb_index = 1 : index} : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  ttkernel.reduce_init(%cb0, %cb1, %cb2, <reduce_sum>, <reduce_dim_col>) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>) -> ()
  ttkernel.reduce_tile(%cb0, %cb1, %c0, %c0, %c0, <reduce_sum>, <reduce_dim_col>) {ttl.reduce_output_cb_index = 1 : index} : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
  func.return
}

// -----

// Omitted exp flags and the metal defaults are the same configuration.
// CHECK-LABEL: func.func @exp_defaults_match_explicit
func.func @exp_defaults_match_explicit() {
  %c0 = arith.constant 0 : index
  ttkernel.exp_tile_init() : () -> ()
  ttkernel.exp_tile(%c0) {approx = false, input_clamping = #ttkernel.input_clamping<clamp_to_negative>, scale = 1065353216 : i32} : (index) -> ()
  ttkernel.exp_tile_init() {approx = false, input_clamping = #ttkernel.input_clamping<clamp_to_negative>, scale = 1065353216 : i32} : () -> ()
  ttkernel.exp_tile(%c0) : (index) -> ()
  func.return
}

// -----

// mm_block_init_short keys include transpose and the block dimensions.
// CHECK-LABEL: func.func @matmul_block_dims_match
func.func @matmul_block_dims_match() {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %transpose = arith.constant 0 : i32
  %ct = arith.constant 1 : i32
  %rt = arith.constant 1 : i32
  %kt = arith.constant 1 : i32
  "ttkernel.mm_block_init_short"(%cb0, %cb1, %transpose, %ct, %rt, %kt) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, i32, i32, i32, i32) -> ()
  ttkernel.matmul_block(%cb0, %cb1, %c0, %c0, %c0, %transpose, %ct, %rt, %kt) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index, index, i32, i32, i32, i32) -> ()
  func.return
}
