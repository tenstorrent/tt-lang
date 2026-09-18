// Verifies target-level constraints of the experimental row-normalization
// block operation.
// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// One row must fit the helper's eight-tile upper bound.
module {
  func.func @too_many_tiles() {
    %input = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<18, !ttcore.tile<32x32, bf16>>
    %gamma = ttkernel.get_compile_time_arg_val(1)
        : () -> !ttkernel.cb<18, !ttcore.tile<32x32, bf16>>
    %output = ttkernel.get_compile_time_arg_val(2)
        : () -> !ttkernel.cb<18, !ttcore.tile<32x32, bf16>>
    // expected-error @below {{'ttkernel.experimental_row_normalization_block' op num_tiles must be in the range [1, 8]}}
    ttkernel.experimental_row_normalization_block(%input, %gamma, %output)
        num_tiles = 9 scale = 1.000000e+00 epsilon = 1.000000e-05
        has_gamma = true dtype = <bf16>
        : (!ttkernel.cb<18, !ttcore.tile<32x32, bf16>>,
           !ttkernel.cb<18, !ttcore.tile<32x32, bf16>>,
           !ttkernel.cb<18, !ttcore.tile<32x32, bf16>>) -> ()
    return
  }
}

// -----

// The target helper supports bf16 DFBs only.
module {
  func.func @unsupported_dtype() {
    %input = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<6, !ttcore.tile<32x32, f32>>
    %output = ttkernel.get_compile_time_arg_val(1)
        : () -> !ttkernel.cb<6, !ttcore.tile<32x32, f32>>
    // expected-error @below {{'ttkernel.experimental_row_normalization_block' op supports bf16 DFBs only}}
    ttkernel.experimental_row_normalization_block(%input, %input, %output)
        num_tiles = 3 scale = 1.000000e+00 epsilon = 1.000000e-05
        has_gamma = false dtype = <f32>
        : (!ttkernel.cb<6, !ttcore.tile<32x32, f32>>,
           !ttkernel.cb<6, !ttcore.tile<32x32, f32>>,
           !ttkernel.cb<6, !ttcore.tile<32x32, f32>>) -> ()
    return
  }
}

// -----

// The semantic scale must be finite and positive.
module {
  func.func @nonfinite_scale() {
    %input = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<6, !ttcore.tile<32x32, bf16>>
    %output = ttkernel.get_compile_time_arg_val(1)
        : () -> !ttkernel.cb<6, !ttcore.tile<32x32, bf16>>
    // expected-error @below {{'ttkernel.experimental_row_normalization_block' op scale must be finite and positive}}
    ttkernel.experimental_row_normalization_block(%input, %input, %output)
        num_tiles = 3 scale = 0x7FC00000 epsilon = 1.000000e-05
        has_gamma = false dtype = <bf16>
        : (!ttkernel.cb<6, !ttcore.tile<32x32, bf16>>,
           !ttkernel.cb<6, !ttcore.tile<32x32, bf16>>,
           !ttkernel.cb<6, !ttcore.tile<32x32, bf16>>) -> ()
    return
  }
}

// -----

// Epsilon must be finite and positive.
module {
  func.func @nonpositive_epsilon() {
    %input = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<6, !ttcore.tile<32x32, bf16>>
    %output = ttkernel.get_compile_time_arg_val(1)
        : () -> !ttkernel.cb<6, !ttcore.tile<32x32, bf16>>
    // expected-error @below {{'ttkernel.experimental_row_normalization_block' op epsilon must be finite and positive}}
    ttkernel.experimental_row_normalization_block(%input, %input, %output)
        num_tiles = 3 scale = 1.000000e+00 epsilon = 0.000000e+00
        has_gamma = false dtype = <bf16>
        : (!ttkernel.cb<6, !ttcore.tile<32x32, bf16>>,
           !ttkernel.cb<6, !ttcore.tile<32x32, bf16>>,
           !ttkernel.cb<6, !ttcore.tile<32x32, bf16>>) -> ()
    return
  }
}

// -----

// Disabled gamma multiplication uses the input DFB as the placeholder.
module {
  func.func @distinct_disabled_gamma() {
    %input = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<6, !ttcore.tile<32x32, bf16>>
    %gamma = ttkernel.get_compile_time_arg_val(1)
        : () -> !ttkernel.cb<6, !ttcore.tile<32x32, bf16>>
    %output = ttkernel.get_compile_time_arg_val(2)
        : () -> !ttkernel.cb<6, !ttcore.tile<32x32, bf16>>
    // expected-error @below {{'ttkernel.experimental_row_normalization_block' op gamma_cb must equal input_cb when has_gamma is false}}
    ttkernel.experimental_row_normalization_block(%input, %gamma, %output)
        num_tiles = 3 scale = 1.000000e+00 epsilon = 1.000000e-05
        has_gamma = false dtype = <bf16>
        : (!ttkernel.cb<6, !ttcore.tile<32x32, bf16>>,
           !ttkernel.cb<6, !ttcore.tile<32x32, bf16>>,
           !ttkernel.cb<6, !ttcore.tile<32x32, bf16>>) -> ()
    return
  }
}
