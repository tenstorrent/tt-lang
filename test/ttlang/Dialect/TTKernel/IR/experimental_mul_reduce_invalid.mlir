// Verifies target-level constraints of the experimental multiply-reduction
// block operation.
// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// The fixed helper accepts at most eight retained product tiles.
module {
  func.func @too_many_tiles() {
    %input = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<18, !ttcore.tile<32x32, bf16>>
    %output = ttkernel.get_compile_time_arg_val(1)
        : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
    // expected-error @below {{'ttkernel.experimental_mul_reduce_block' op num_tiles must be in the range [1, 8]}}
    ttkernel.experimental_mul_reduce_block(%input, %input, %output)
        num_tiles = 9 scale = 1.000000e+00 dtype = <bf16>
        : (!ttkernel.cb<18, !ttcore.tile<32x32, bf16>>,
           !ttkernel.cb<18, !ttcore.tile<32x32, bf16>>,
           !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    return
  }
}

// -----

// The target helper supports bf16 DFBs only.
module {
  func.func @unsupported_dtype() {
    %input = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<8, !ttcore.tile<32x32, f32>>
    %output = ttkernel.get_compile_time_arg_val(1)
        : () -> !ttkernel.cb<2, !ttcore.tile<32x32, f32>>
    // expected-error @below {{'ttkernel.experimental_mul_reduce_block' op supports bf16 DFBs only}}
    ttkernel.experimental_mul_reduce_block(%input, %input, %output)
        num_tiles = 4 scale = 1.000000e+00 dtype = <f32>
        : (!ttkernel.cb<8, !ttcore.tile<32x32, f32>>,
           !ttkernel.cb<8, !ttcore.tile<32x32, f32>>,
           !ttkernel.cb<2, !ttcore.tile<32x32, f32>>) -> ()
    return
  }
}

// -----

// The semantic reduction scale must be finite and positive.
module {
  func.func @nonfinite_scale() {
    %input = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>
    %output = ttkernel.get_compile_time_arg_val(1)
        : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
    // expected-error @below {{'ttkernel.experimental_mul_reduce_block' op scale must be finite and positive}}
    ttkernel.experimental_mul_reduce_block(%input, %input, %output)
        num_tiles = 4 scale = 0x7FC00000 dtype = <bf16>
        : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>,
           !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>,
           !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    return
  }
}

// -----

// Both inputs and the output must use one tile data format.
module {
  func.func @mismatched_dfb_types() {
    %lhs = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>
    %rhs = ttkernel.get_compile_time_arg_val(1)
        : () -> !ttkernel.cb<8, !ttcore.tile<16x32, bf16>>
    %output = ttkernel.get_compile_time_arg_val(2)
        : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
    // expected-error @below {{'ttkernel.experimental_mul_reduce_block' op input and output dataflow buffer types must match}}
    ttkernel.experimental_mul_reduce_block(%lhs, %rhs, %output)
        num_tiles = 4 scale = 1.000000e+00 dtype = <bf16>
        : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>,
           !ttkernel.cb<8, !ttcore.tile<16x32, bf16>>,
           !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    return
  }
}
