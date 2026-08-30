// RUN: ttlang-opt %s --canonicalize | FileCheck %s

// Summary: Verifies constant-table lookup canonicalization and dynamic-index
// preservation.

// A constant index is replaced by an arith constant with the selected value.
// CHECK-LABEL: func.func @canonicalize_constant_index
// CHECK-NEXT:    %[[VALUE:.*]] = arith.constant 5 : index
// CHECK-NEXT:    return %[[VALUE]] : index
func.func @canonicalize_constant_index() -> index {
  %index = arith.constant 1 : index
  %value = ttkernel.experimental.constant_table_lookup %index, [3, 5, 8] : index
  return %value : index
}

// A dynamic index preserves the table lookup and its input.
// CHECK-LABEL: func.func @retain_dynamic_index(%[[INDEX:.*]]: index)
// CHECK-NEXT:    %[[VALUE:.*]] = ttkernel.experimental.constant_table_lookup
// CHECK-SAME:      %[[INDEX]], [3, 5, 8] : index
// CHECK-NEXT:    return %[[VALUE]] : index
func.func @retain_dynamic_index(%index : index) -> index {
  %value = ttkernel.experimental.constant_table_lookup %index, [3, 5, 8] : index
  return %value : index
}
