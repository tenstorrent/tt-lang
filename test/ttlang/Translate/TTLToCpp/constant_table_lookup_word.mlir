// RUN: ttlang-opt --convert-ttkernel-to-emitc %s -o %t.emitc.mlir
// RUN: ttlang-translate --ttkernel-to-cpp %t.emitc.mlir | FileCheck %s

// Verify that a kernel using only a one-word table includes its helper.

// CHECK: constant_table_lookup_word
// CHECK-LABEL: void kernel_main()
// CHECK: experimental::constant_table_lookup_word<4>({{.*}}, 0x853U)
func.func @kernel_main() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %index = arith.constant 1 : index
  %value = ttkernel.experimental.constant_table_lookup %index, [3, 5, 8] : index
  return
}
