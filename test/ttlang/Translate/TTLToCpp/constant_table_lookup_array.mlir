// RUN: ttlang-opt --convert-ttkernel-to-emitc %s -o %t.emitc.mlir
// RUN: ttlang-translate --ttkernel-to-cpp %t.emitc.mlir | FileCheck %s

// Verify that a kernel using only an array-backed table includes its helper and
// qualifies the call in the global namespace.

// CHECK: namespace experimental {
// CHECK: constant_table_lookup(std::size_t index,
// CHECK-LABEL: void kernel_main()
// CHECK: ::experimental::constant_table_lookup<4>({{.*}}, __ttlang_constant_table_{{[A-F0-9]+}})
func.func @kernel_main() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %index = arith.constant 1 : index
  %value = ttkernel.experimental.constant_table_lookup %index,
      [0, 1, 2, 3, 4, 5, 6, 7, 8] : index
  return
}
