// Negative tests for the ttkernel-insert-inits pass.
// Verifies that malformed sync regions produce errors instead of silently
// miscompiling.

// RUN: ttlang-opt %s --ttkernel-insert-inits --split-input-file --verify-diagnostics

// -----

// Test: tile_regs_acquire without matching tile_regs_release.
func.func @missing_release() {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  // expected-error @below {{'ttkernel.tile_regs_acquire' op tile_regs_acquire without matching tile_regs_release}}
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%cb, %c0, %c0) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.exp_tile(%c0) : (index) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %cb, %c0, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  func.return
}
