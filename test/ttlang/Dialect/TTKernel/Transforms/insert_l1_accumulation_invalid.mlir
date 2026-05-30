// Negative tests for ttkernel-insert-l1-accumulation.
// Verifies that annotated accumulation loops carry explicit initial-mode
// metadata instead of relying on TTKernel IR inference.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttkernel-insert-l1-accumulation)' --split-input-file --verify-diagnostics

// -----

// Test: ttl.l1_acc_loop without ttl.l1_acc_initial metadata.
func.func @missing_initial_mode() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // expected-error @below {{'scf.for' op requires ttl.l1_acc_initial overwrite or accumulate_existing metadata}}
  scf.for %iv = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_loop}
  return
}

// -----

// Test: L1 packer accumulation rejects unsupported output formats.
func.func @unsupported_l1_acc_output_format() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bfp_bf8>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %iv = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    // expected-error @below {{'ttkernel.pack_tile' op L1 packer accumulation does not support output data type bfp_bf8}}
    ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bfp_bf8>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_initial = 0 : i32, ttl.l1_acc_loop}
  return
}
