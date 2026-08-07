// Verifies a retained source scalar shared by multiple consumers.
// RUN: ttlang-opt %s -o /dev/null

// One acquire and release may contain several source-scalar consumers.
module {
  func.func @multiple_consumers() {
    %first_input = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<6, !ttcore.tile<32x32, bf16>>
    %second_input = ttkernel.get_compile_time_arg_val(1)
        : () -> !ttkernel.cb<6, !ttcore.tile<32x32, bf16>>
    %scalar = arith.constant 0 : index
    ttkernel.experimental_source_scalar_acquire(%scalar, %scalar)
        : (index, index) -> ()
    ttkernel.experimental_source_scalar_apply_mul(%first_input)
        num_tiles = 3 dtype = <bf16>
        : (!ttkernel.cb<6, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.experimental_source_scalar_apply_mul(%second_input)
        num_tiles = 3 dtype = <bf16>
        : (!ttkernel.cb<6, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.experimental_source_scalar_release
    return
  }
}
