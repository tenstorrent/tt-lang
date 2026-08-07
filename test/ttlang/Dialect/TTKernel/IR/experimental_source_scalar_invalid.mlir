// Verifies target-level constraints of retained source-scalar multiplication.
// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// A source-scalar consumer must fit the helper's eight-tile upper bound.
module {
  func.func @too_many_tiles() {
    %input = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<18, !ttcore.tile<32x32, bf16>>
    %scalar = arith.constant 0 : index
    %output = arith.constant 0 : index
    // expected-error @below {{'ttkernel.experimental_source_scalar_mul' op num_tiles must be in the range [1, 8]}}
    ttkernel.experimental_source_scalar_mul(%input, %scalar, %output)
        num_tiles = 9 dtype = <bf16>
        : (!ttkernel.cb<18, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    return
  }
}

// -----

// The target helper supports bf16 DFBs only.
module {
  func.func @unsupported_dtype() {
    %input = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<6, !ttcore.tile<32x32, f32>>
    %scalar = arith.constant 0 : index
    %output = arith.constant 0 : index
    // expected-error @below {{'ttkernel.experimental_source_scalar_mul' op supports bf16 DFBs only}}
    ttkernel.experimental_source_scalar_mul(%input, %scalar, %output)
        num_tiles = 3 dtype = <f32>
        : (!ttkernel.cb<6, !ttcore.tile<32x32, f32>>, index, index) -> ()
    return
  }
}

// -----

// The explicit dtype must agree with the input DFB tile type.
module {
  func.func @mismatched_dtype() {
    %input = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<6, !ttcore.tile<32x32, f32>>
    %scalar = arith.constant 0 : index
    %output = arith.constant 0 : index
    // expected-error @below {{'ttkernel.experimental_source_scalar_mul' op dtype must match the input tile data type}}
    ttkernel.experimental_source_scalar_mul(%input, %scalar, %output)
        num_tiles = 3 dtype = <bf16>
        : (!ttkernel.cb<6, !ttcore.tile<32x32, f32>>, index, index) -> ()
    return
  }
}
