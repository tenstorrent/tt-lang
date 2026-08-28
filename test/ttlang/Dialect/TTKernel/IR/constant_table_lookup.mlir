// RUN: ttlang-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @fold_constant_index
// CHECK:         %[[VALUE:.*]] = arith.constant 5 : index
// CHECK-NEXT:    return %[[VALUE]] : index
// CHECK-NOT:     ttkernel.experimental.constant_table_lookup
func.func @fold_constant_index() -> index {
  %index = arith.constant 1 : index
  %value = ttkernel.experimental.constant_table_lookup %index, [3, 5, 8] : index
  return %value : index
}
