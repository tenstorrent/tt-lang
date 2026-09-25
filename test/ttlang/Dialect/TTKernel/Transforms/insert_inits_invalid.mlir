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

// -----

// Test: the slab helpers are chosen once per sync region, so the TopK stages
// of a region cannot disagree about the mode that selects them.
func.func @topk_conflicting_slab_modes() {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c32_i32 = arith.constant 32 : i32
  ttkernel.tile_regs_acquire() : () -> ()
  // expected-note @below {{slab mode established by this stage}}
  ttkernel.topk_local_sort(%c0, %c0_i32, %c4_i32, %c0_i32) {fused = true} : (index, i32, i32, i32) -> ()
  // expected-error @below {{'ttkernel.topk_merge' op TopK stages in one sync region must share one slab mode}}
  ttkernel.topk_merge(%c0, %c0_i32, %c32_i32) : (index, i32, i32) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}
