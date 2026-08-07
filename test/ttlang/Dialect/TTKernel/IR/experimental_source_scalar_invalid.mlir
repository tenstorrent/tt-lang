// Verifies target-level constraints of retained source-scalar multiplication.
// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// A source-scalar consumer must fit the helper's eight-tile upper bound.
module {
  func.func @too_many_tiles() {
    %input = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<18, !ttcore.tile<32x32, bf16>>
    %scalar = arith.constant 0 : index
    ttkernel.experimental_source_scalar_acquire(%scalar, %scalar)
        : (index, index) -> ()
    // expected-error @below {{'ttkernel.experimental_source_scalar_apply_mul' op num_tiles must be in the range [1, 8]}}
    ttkernel.experimental_source_scalar_apply_mul(%input)
        num_tiles = 9 dtype = <bf16>
        : (!ttkernel.cb<18, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.experimental_source_scalar_release
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
    ttkernel.experimental_source_scalar_acquire(%scalar, %scalar)
        : (index, index) -> ()
    // expected-error @below {{'ttkernel.experimental_source_scalar_apply_mul' op supports bf16 DFBs only}}
    ttkernel.experimental_source_scalar_apply_mul(%input)
        num_tiles = 3 dtype = <f32>
        : (!ttkernel.cb<6, !ttcore.tile<32x32, f32>>) -> ()
    ttkernel.experimental_source_scalar_release
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
    ttkernel.experimental_source_scalar_acquire(%scalar, %scalar)
        : (index, index) -> ()
    // expected-error @below {{'ttkernel.experimental_source_scalar_apply_mul' op dtype must match the input tile data type}}
    ttkernel.experimental_source_scalar_apply_mul(%input)
        num_tiles = 3 dtype = <bf16>
        : (!ttkernel.cb<6, !ttcore.tile<32x32, f32>>) -> ()
    ttkernel.experimental_source_scalar_release
    return
  }
}

// -----

// A consumer requires a preceding acquire in the same block.
module {
  func.func @consumer_without_acquire() {
    %input = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<6, !ttcore.tile<32x32, bf16>>
    // expected-error @below {{'ttkernel.experimental_source_scalar_apply_mul' op requires an active source scalar}}
    ttkernel.experimental_source_scalar_apply_mul(%input)
        num_tiles = 3 dtype = <bf16>
        : (!ttkernel.cb<6, !ttcore.tile<32x32, bf16>>) -> ()
    return
  }
}

// -----

// A release requires a preceding acquire in the same block.
module {
  func.func @release_without_acquire() {
    // expected-error @below {{'ttkernel.experimental_source_scalar_release' op requires an active source scalar}}
    ttkernel.experimental_source_scalar_release
    return
  }
}

// -----

// The hardware source register permits only one active scalar.
module {
  func.func @overlapping_acquires() {
    %scalar = arith.constant 0 : index
    ttkernel.experimental_source_scalar_acquire(%scalar, %scalar)
        : (index, index) -> ()
    // expected-error @below {{'ttkernel.experimental_source_scalar_acquire' op cannot acquire while another source scalar is active}}
    ttkernel.experimental_source_scalar_acquire(%scalar, %scalar)
        : (index, index) -> ()
    ttkernel.experimental_source_scalar_release
    return
  }
}

// -----

// Every acquired source scalar must be released in the same block.
module {
  func.func @missing_release() {
    %scalar = arith.constant 0 : index
    // expected-error @below {{'ttkernel.experimental_source_scalar_acquire' op requires a matching source-scalar release in the same block}}
    ttkernel.experimental_source_scalar_acquire(%scalar, %scalar)
        : (index, index) -> ()
    return
  }
}

// -----

// Other compute operations may invalidate the retained source register.
module {
  func.func @intervening_operation() {
    %scalar = arith.constant 0 : index
    ttkernel.experimental_source_scalar_acquire(%scalar, %scalar)
        : (index, index) -> ()
    // expected-error @below {{'ttkernel.tile_regs_commit' op only source-scalar consumers may execute between acquire and release}}
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.experimental_source_scalar_release
    return
  }
}
