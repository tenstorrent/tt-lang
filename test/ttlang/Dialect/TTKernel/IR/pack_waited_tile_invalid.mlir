// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// Verify that waited packing carries the complete-ring proof.

func.func @mismatched_proof_tile_count() {
  %c0 = arith.constant 0 : index
  %dfb = ttkernel.get_compile_time_arg_val(0)
      : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttkernel.pack_waited_tile' op acquired_tiles must equal DFB capacity 2, got 1}}
  ttkernel.pack_waited_tile(%c0, %dfb, %c0, true)
      {acquired_tiles = 1 : i64}
      : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  return
}

// -----

// Verify that waited packing uses absolute output indexing.
func.func @ordered_packing() {
  %c0 = arith.constant 0 : index
  %dfb = ttkernel.get_compile_time_arg_val(0)
      : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttkernel.pack_waited_tile' op requires out_of_order packing}}
  ttkernel.pack_waited_tile(%c0, %dfb, %c0, false)
      {acquired_tiles = 1 : i64}
      : (index, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, index) -> ()
  return
}
