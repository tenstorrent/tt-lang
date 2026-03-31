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

// -----

// Test: sync region packs to output CBs with different data formats.
// bf16 vs f32 element types require different PACK routing, so this must error.
func.func @multiple_output_cbs_different_formats() {
  %cb_bf16 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %cb_f32 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%cb_bf16, %c0, %c0) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.exp_tile(%c0) : (index) -> ()
  ttkernel.pack_tile(%c0, %cb_bf16, %c0, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  // expected-error @below {{'ttkernel.pack_tile' op sync region packs to output CBs with different data formats}}
  ttkernel.pack_tile(%c0, %cb_f32, %c0, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, f32>>, index) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}
