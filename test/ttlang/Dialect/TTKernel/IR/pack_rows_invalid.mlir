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

func.func @output_must_contain_tiles() {
  %c0 = arith.constant 0 : index
  %cb = ttkernel.get_compile_time_arg_val(0)
      : () -> !ttkernel.cb<14, bf16>
  // expected-error @below {{'ttkernel.pack_rows' op output dataflow buffer must contain tile elements}}
  ttkernel.pack_rows(%c0, %cb, %c0) {row_count = 28 : i64}
      : (index, !ttkernel.cb<14, bf16>, index) -> ()
  func.return
}

// -----

func.func @row_count_must_be_positive() {
  %c0 = arith.constant 0 : index
  %cb = ttkernel.get_compile_time_arg_val(0)
      : () -> !ttkernel.cb<64, !ttcore.tile<1x32, bf16>>
  // expected-error @below {{'ttkernel.pack_rows' op attribute 'row_count' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive whose maximum value is 64}}
  ttkernel.pack_rows(%c0, %cb, %c0) {row_count = 0 : i64}
      : (index, !ttkernel.cb<64, !ttcore.tile<1x32, bf16>>, index) -> ()
  func.return
}

// -----

func.func @row_count_too_large() {
  %c0 = arith.constant 0 : index
  %cb = ttkernel.get_compile_time_arg_val(0)
      : () -> !ttkernel.cb<64, !ttcore.tile<1x32, bf16>>
  // expected-error @below {{'ttkernel.pack_rows' op attribute 'row_count' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive whose maximum value is 64}}
  ttkernel.pack_rows(%c0, %cb, %c0) {row_count = 65 : i64}
      : (index, !ttkernel.cb<64, !ttcore.tile<1x32, bf16>>, index) -> ()
  func.return
}

// -----

func.func @output_capacity_too_small() {
  %c0 = arith.constant 0 : index
  %cb = ttkernel.get_compile_time_arg_val(0)
      : () -> !ttkernel.cb<1, !ttcore.tile<1x32, bf16>>
  // expected-error @below {{'ttkernel.pack_rows' op packed prefix requires 896 bytes, but only 64 bytes remain in the output dataflow buffer}}
  ttkernel.pack_rows(%c0, %cb, %c0) {row_count = 28 : i64}
      : (index, !ttkernel.cb<1, !ttcore.tile<1x32, bf16>>, index) -> ()
  func.return
}

// -----

func.func @output_index_leaves_insufficient_capacity() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cb = ttkernel.get_compile_time_arg_val(0)
      : () -> !ttkernel.cb<14, !ttcore.tile<1x32, bf16>>
  // expected-error @below {{'ttkernel.pack_rows' op packed prefix requires 896 bytes, but only 832 bytes remain in the output dataflow buffer}}
  ttkernel.pack_rows(%c0, %cb, %c1) {row_count = 28 : i64}
      : (index, !ttkernel.cb<14, !ttcore.tile<1x32, bf16>>, index) -> ()
  func.return
}

// -----

func.func @negative_output_index() {
  %c0 = arith.constant 0 : index
  %cm1 = arith.constant -1 : index
  %cb = ttkernel.get_compile_time_arg_val(0)
      : () -> !ttkernel.cb<14, !ttcore.tile<1x32, bf16>>
  // expected-error @below {{'ttkernel.pack_rows' op out_index must be nonnegative}}
  ttkernel.pack_rows(%c0, %cb, %cm1) {row_count = 28 : i64}
      : (index, !ttkernel.cb<14, !ttcore.tile<1x32, bf16>>, index) -> ()
  func.return
}

// -----

func.func @output_index_exceeds_capacity() {
  %c0 = arith.constant 0 : index
  %c14 = arith.constant 14 : index
  %cb = ttkernel.get_compile_time_arg_val(0)
      : () -> !ttkernel.cb<14, !ttcore.tile<1x32, bf16>>
  // expected-error @below {{'ttkernel.pack_rows' op out_index 14 exceeds output dataflow buffer capacity of 14 pages}}
  ttkernel.pack_rows(%c0, %cb, %c14) {row_count = 28 : i64}
      : (index, !ttkernel.cb<14, !ttcore.tile<1x32, bf16>>, index) -> ()
  func.return
}
