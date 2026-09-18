// Negative tests for ttkernel-insert-l1-accumulation.
// Verifies that annotated accumulation loops carry explicit metadata instead
// of relying on TTKernel IR inference.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttkernel-insert-l1-accumulation)' --split-input-file --verify-diagnostics

// -----

// Test: ttl.l1_acc_loop without ttl.l1_acc_initial metadata.
func.func @missing_initial_mode() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // expected-error @below {{'scf.for' op requires ttl.l1_acc_initial metadata with value overwrite or accumulate_existing; run ttl-lower-accumulation-scopes before ttkernel-insert-l1-accumulation}}
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

// Test: ttl.l1_acc_loop without ttl.l1_acc_scope_id metadata.
func.func @missing_scope_id() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // expected-error @below {{'scf.for' op requires ttl.l1_acc_scope_id metadata; run ttl-lower-accumulation-scopes before ttkernel-insert-l1-accumulation}}
  scf.for %iv = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_initial = 0 : i32, ttl.l1_acc_loop}
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
    // expected-error @below {{'ttkernel.pack_tile' op L1 packer accumulation does not support output data type bfp_bf8; use a supported output data type or select another accumulation strategy}}
    ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bfp_bf8>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_initial = 0 : i32, ttl.l1_acc_loop, ttl.l1_acc_scope_id = 0 : i64}
  return
}

// -----

// Test: L1 packer accumulation validates pack_tile_block output formats.
func.func @unsupported_l1_acc_output_format_pack_block() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bfp_bf8>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %iv = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    // expected-error @below {{'ttkernel.pack_tile_block' op L1 packer accumulation does not support output data type bfp_bf8; use a supported output data type or select another accumulation strategy}}
    ttkernel.pack_tile_block(%c0, %cb, %c4) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bfp_bf8>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_initial = 0 : i32, ttl.l1_acc_loop, ttl.l1_acc_scope_id = 0 : i64}
  return
}

// -----

// Test: nested L1 accumulation loops must belong to one semantic scope.
func.func @nested_mismatched_scope_ids() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %outer = %c0 to %c2 step %c1 {
    // expected-error @below {{'scf.for' op nested independent L1 accumulation scopes are not supported (#648); nested loops that belong to one accumulation must use matching ttl.l1_acc_scope_id metadata}}
    scf.for %inner = %c0 to %c2 step %c1 {
      ttkernel.tile_regs_acquire() : () -> ()
      ttkernel.tile_regs_commit() : () -> ()
      ttkernel.tile_regs_wait() : () -> ()
      ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
      ttkernel.tile_regs_release() : () -> ()
    } {ttl.l1_acc_initial = 0 : i32, ttl.l1_acc_loop, ttl.l1_acc_scope_id = 1 : i64}
  } {ttl.l1_acc_initial = 0 : i32, ttl.l1_acc_loop, ttl.l1_acc_scope_id = 0 : i64}
  return
}
