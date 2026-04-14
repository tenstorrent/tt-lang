// Negative tests for ttkernel-insert-l1-accumulation with --strict-f32-acc.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttkernel-insert-l1-accumulation{strict-f32-acc=true})' --verify-diagnostics --split-input-file

// L1 acc loop with subblock loop inside: strict-f32-acc should error.

func.func @strict_f32_subblock_error() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  // expected-error @below {{output block exceeds f32 DST capacity}}
  scf.for %iv = %c0 to %c4 step %c1 {
    scf.for %sb = %c0 to %c2 step %c1 {
      ttkernel.tile_regs_acquire() : () -> ()
      ttkernel.matmul_block(%cb, %cb, %c0, %c0, %c0, %c0_i32, %c1_i32, %c1_i32, %c1_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index, index, i32, i32, i32, i32) -> ()
      ttkernel.tile_regs_commit() : () -> ()
      ttkernel.tile_regs_wait() : () -> ()
      ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
      ttkernel.tile_regs_release() : () -> ()
    } {ttl.subblock_loop_stride = 1 : index}
  } {ttl.l1_acc_loop}
  return
}

// -----

// L1 acc loop WITHOUT subblock loops: strict-f32-acc should pass.

// expected-no-diagnostics
func.func @strict_f32_no_subblock_ok() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  scf.for %iv = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_loop}
  ttkernel.cb_push_back(%cb, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  return
}

// -----

// Reduction loop (compiler-generated) with subblock: strict-f32-acc should
// NOT error (only user-written l1_acc_loop triggers the check).

// expected-no-diagnostics
func.func @strict_f32_reduction_loop_ok() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb_in = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  %cb_scaler = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  %cb_out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %iv = %c0 to %c2 step %c1 {
    scf.for %sb = %c0 to %c2 step %c1 {
      ttkernel.tile_regs_acquire() : () -> ()
      ttkernel.reduce_tile(%cb_in, %cb_scaler, %c0, %c0, %c0, <reduce_sum>, <reduce_dim_col>) : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
      ttkernel.tile_regs_commit() : () -> ()
      ttkernel.tile_regs_wait() : () -> ()
      ttkernel.pack_tile(%c0, %cb_out, %c0, true) : (index, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, index) -> ()
      ttkernel.tile_regs_release() : () -> ()
    } {ttl.subblock_loop_stride = 1 : index}
  } {ttl.reduction_loop}
  return
}
