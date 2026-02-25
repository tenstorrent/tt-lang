// Verify that unary_bcast without the ttl.bcast_output_cb_index attribute
// produces an error during init insertion.

// RUN: ttlang-opt %s --ttkernel-insert-inits --split-input-file --verify-diagnostics

func.func @bcast_missing_output_cb_attr() {
  %in_cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %out_cb = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  ttkernel.tile_regs_acquire() : () -> ()
  // expected-error @below {{missing ttl.bcast_output_cb_index attribute}}
  ttkernel.unary_bcast(%in_cb, %c0, %c0, <col>) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %out_cb, %c0, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}
