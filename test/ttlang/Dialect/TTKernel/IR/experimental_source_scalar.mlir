// Verifies a retained source scalar shared by consumers with distinct outputs.
// RUN: ttlang-opt %s --convert-ttkernel-to-emitc | FileCheck %s

// One acquire and release may contain several source-scalar consumers.
module {
  func.func @multiple_consumers()
      attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %first_input = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<6, !ttcore.tile<32x32, bf16>>
    %second_input = ttkernel.get_compile_time_arg_val(1)
        : () -> !ttkernel.cb<6, !ttcore.tile<32x32, bf16>>
    %scalar = arith.constant 6 : index
    %first_output = arith.constant 0 : index
    %second_output = arith.constant 3 : index
    // CHECK-LABEL: func.func @multiple_consumers
    // CHECK:      %[[SCALAR:.*]] = "emitc.constant"() <{value = 6 : index}>
    // CHECK:      %[[FIRST_OUTPUT:.*]] = "emitc.constant"() <{value = 0 : index}>
    // CHECK:      %[[SECOND_OUTPUT:.*]] = "emitc.constant"() <{value = 3 : index}>
    // CHECK:      emitc.call_opaque "experimental::source_scalar_acquire"(%[[SCALAR]])
    ttkernel.experimental_source_scalar_acquire(%scalar) : (index) -> ()
    // CHECK:      emitc.call_opaque "experimental::source_scalar_mul"(%{{.*}}, %[[FIRST_OUTPUT]])
    ttkernel.experimental_source_scalar_apply_mul(%first_input, %first_output)
        num_tiles = 3 dtype = <bf16>
        : (!ttkernel.cb<6, !ttcore.tile<32x32, bf16>>, index) -> ()
    // CHECK:      emitc.call_opaque "experimental::source_scalar_mul"(%{{.*}}, %[[SECOND_OUTPUT]])
    ttkernel.experimental_source_scalar_apply_mul(%second_input, %second_output)
        num_tiles = 3 dtype = <bf16>
        : (!ttkernel.cb<6, !ttcore.tile<32x32, bf16>>, index) -> ()
    // CHECK:      emitc.call_opaque "experimental::source_scalar_release"()
    ttkernel.experimental_source_scalar_release
    return
  }
}
