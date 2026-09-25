// Verifies helper include selection, deduplication, and absence when unused.
// RUN: ttlang-opt --split-input-file --convert-ttkernel-to-emitc %s | ttlang-translate --split-input-file --ttkernel-to-cpp | FileCheck %s --implicit-check-not="asm volatile" --implicit-check-not="namespace dfb_reset_detail" --implicit-check-not="namespace dfb_reconfiguration_detail"

// CHECK: #include "api/compute/common.h"
// CHECK: #include "ttlang/Target/TTKernel/LLKs/experimental_dfb_reconfiguration.h"
// CHECK-NEXT: #include "ttlang/Target/TTKernel/LLKs/experimental_dfb_reset.h"
// CHECK-NOT: #include
// CHECK-LABEL: void kernel_main()
// CHECK: experimental::reset_dfb_interfaces({{.*}}, {{.*}}, {{.*}});
// CHECK: experimental::reconfigure_dfb_interfaces({{.*}});
// CHECK: experimental::reconfigure_dfb_interfaces({{.*}});
func.func @kernel_main() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %address = arith.constant 4096 : i32
  %mask = arith.constant 1 : i32
  ttkernel.opaque_call "experimental::reset_dfb_interfaces"(%address, %mask, %mask) {header = "<cstdint>"} : (i32, i32, i32) -> ()
  ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces"(%address) {header = "<cstdint>"} : (i32) -> ()
  ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces"(%address) {header = "<cstdint>"} : (i32) -> ()
  return
}

// -----

// CHECK: #include "api/dataflow/dataflow_api.h"
// CHECK: #include "ttlang/Target/TTKernel/LLKs/experimental_dfb_reconfiguration.h"
// CHECK-NOT: experimental_dfb_reset.h
// CHECK-LABEL: void kernel_main()
// CHECK: experimental::reconfigure_dfb_interfaces({{.*}});
func.func @kernel_main() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %address = arith.constant 4096 : i32
  ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces"(%address) {header = "<cstdint>"} : (i32) -> ()
  return
}

// -----

// CHECK: #include "api/dataflow/dataflow_api.h"
// CHECK-NOT: experimental_dfb_
// CHECK-LABEL: void kernel_main()
func.func @kernel_main() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  return
}
