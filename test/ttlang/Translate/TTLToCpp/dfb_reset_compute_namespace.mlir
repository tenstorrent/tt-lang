// Verifies global reset qualification when LLKOperand exposes ckernel::experimental.
// RUN: ttlang-opt --convert-ttkernel-to-emitc %s -o %t.emitc.mlir
// RUN: ttlang-translate --ttkernel-to-cpp %t.emitc.mlir | FileCheck %s

// CHECK: #include "api/compute/experimental/2_0/llk_operand.h"
// CHECK: namespace experimental {
// CHECK: FORCE_INLINE void reset_dfb_interfaces(uint32_t synchronizationAddress,
// CHECK-LABEL: void kernel_main()
// CHECK: using namespace ckernel;
// CHECK: {{^ *}}::experimental::reset_dfb_interfaces({{.*}}, {{.*}}, {{.*}});

func.func @reset_with_compute_namespace() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  // The real compute API also declares an experimental namespace.
  emitc.verbatim "using namespace ckernel;"
  %synchronization_address = arith.constant 4096 : i32
  %low_mask = arith.constant 1 : i32
  %high_mask = arith.constant 2 : i32
  ttkernel.opaque_call "::experimental::reset_dfb_interfaces"(%synchronization_address, %low_mask, %high_mask) {header = "api/compute/experimental/2_0/llk_operand.h", unsigned_arg_indices = array<i32: 0, 1, 2>} : (i32, i32, i32) -> ()
  return
}
