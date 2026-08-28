// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

func.func @unsupported_data_type() {
  %c0 = arith.constant 0 : index
  %cb = ttkernel.get_compile_time_arg_val(0)
      : () -> !ttkernel.cb<14, !ttcore.tile<1x32, bfp_bf8>>
  // expected-error @below {{'ttkernel.pack_rows' op supports only bf16 and f32 output data types}}
  ttkernel.pack_rows(%c0, %cb, %c0) {row_count = 28 : i64}
      : (index, !ttkernel.cb<14, !ttcore.tile<1x32, bfp_bf8>>, index) -> ()
  func.return
}

// -----

func.func @row_count_too_large() {
  %c0 = arith.constant 0 : index
  %cb = ttkernel.get_compile_time_arg_val(0)
      : () -> !ttkernel.cb<64, !ttcore.tile<1x32, bf16>>
  // expected-error @below {{'ttkernel.pack_rows' op row_count must be at most 64, got 65}}
  ttkernel.pack_rows(%c0, %cb, %c0) {row_count = 65 : i64}
      : (index, !ttkernel.cb<64, !ttcore.tile<1x32, bf16>>, index) -> ()
  func.return
}

// -----

func.func @output_capacity_too_small() {
  %c0 = arith.constant 0 : index
  %cb = ttkernel.get_compile_time_arg_val(0)
      : () -> !ttkernel.cb<1, !ttcore.tile<1x32, bf16>>
  // expected-error @below {{'ttkernel.pack_rows' op packed prefix requires 896 bytes, but output dataflow buffer capacity is 64 bytes}}
  ttkernel.pack_rows(%c0, %cb, %c0) {row_count = 28 : i64}
      : (index, !ttkernel.cb<1, !ttcore.tile<1x32, bf16>>, index) -> ()
  func.return
}
