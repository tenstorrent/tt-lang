// reconfig_data_format_srca lowers to the compute API of the same name.
// RUN: ttlang-opt --convert-ttkernel-to-emitc %s | ttlang-translate --ttkernel-to-cpp | FileCheck %s

// CHECK: #include "api/compute/reconfig_data_format.h"
// CHECK-LABEL: void kernel_main()
// CHECK: reconfig_data_format_srca(get_compile_time_arg_val(0), get_compile_time_arg_val(1));
// CHECK: reconfig_data_format_srca(get_compile_time_arg_val(1));
func.func @kernel_main() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %old = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  %new = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, u16>>
  ttkernel.reconfig_data_format_srca(%old, %new) : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<1, !ttcore.tile<32x32, u16>>) -> ()
  ttkernel.reconfig_data_format_srca(%new) : (!ttkernel.cb<1, !ttcore.tile<32x32, u16>>) -> ()
  return
}
