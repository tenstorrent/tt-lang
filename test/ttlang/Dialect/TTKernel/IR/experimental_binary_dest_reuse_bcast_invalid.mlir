// Verifies constraints on column-broadcast multiplication with DST reuse.
// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// The device-validated target operation supports multiplication only.
module {
  func.func @unsupported_operation() {
    %input = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
    // expected-error @below {{'ttkernel.experimental.binary_dest_reuse_bcast_tiles_init' op supports multiplication only}}
    ttkernel.experimental.binary_dest_reuse_bcast_tiles_init(
        %input, <add>, <col>, <dest_to_srca>)
        : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    return
  }
}

// -----

// The target operation supports column broadcast only.
module {
  func.func @unsupported_broadcast() {
    %input = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
    // expected-error @below {{'ttkernel.experimental.binary_dest_reuse_bcast_tiles_init' op supports column broadcast only}}
    ttkernel.experimental.binary_dest_reuse_bcast_tiles_init(
        %input, <mul>, <row>, <dest_to_srca>)
        : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    return
  }
}

// -----

// The DFB operand must occupy source B so DST can be reused as source A.
module {
  func.func @unsupported_reuse_direction() {
    %input = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
    // expected-error @below {{'ttkernel.experimental.binary_dest_reuse_bcast_tiles_init' op requires dest_to_srca because broadcast applies to source B}}
    ttkernel.experimental.binary_dest_reuse_bcast_tiles_init(
        %input, <mul>, <col>, <dest_to_srcb>)
        : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    return
  }
}

// -----

// Application requires initialization in the same block.
module {
  func.func @missing_initialization() {
    %input = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
    %index = arith.constant 0 : index
    // expected-error @below {{'ttkernel.experimental.binary_dest_reuse_bcast_tiles' op requires a preceding broadcast destination-reuse initialization in the same block}}
    ttkernel.experimental.binary_dest_reuse_bcast_tiles(
        %input, %index, %index, <mul>, <col>, <dest_to_srca>)
        : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    return
  }
}

// -----

// Application and initialization must refer to the same DFB and attributes.
module {
  func.func @mismatched_initialization() {
    %first_input = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
    %second_input = ttkernel.get_compile_time_arg_val(1)
        : () -> !ttkernel.cb<3, !ttcore.tile<32x32, bf16>>
    %index = arith.constant 0 : index
    ttkernel.experimental.binary_dest_reuse_bcast_tiles_init(
        %first_input, <mul>, <col>, <dest_to_srca>)
        : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    // expected-error @below {{'ttkernel.experimental.binary_dest_reuse_bcast_tiles' op requires a preceding broadcast destination-reuse initialization with identical operands and attributes}}
    ttkernel.experimental.binary_dest_reuse_bcast_tiles(
        %second_input, %index, %index, <mul>, <col>, <dest_to_srca>)
        : (!ttkernel.cb<3, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    return
  }
}
