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
// CHECK-LABEL: func.func @retain_dynamic_index
// CHECK-SAME: (%[[INDEX:.*]]: index)
// CHECK-NEXT:    %[[VALUE:.*]] = ttkernel.experimental.constant_table_lookup
// CHECK-SAME:      %[[INDEX]], [3, 5, 8] : index
// CHECK-NEXT:    return %[[VALUE]] : index
func.func @retain_dynamic_index(%index : index) -> index {
  %value = ttkernel.experimental.constant_table_lookup %index, [3, 5, 8] : index
  return %value : index
}

// A row-major lookup with one constant column retains only that column.
// CHECK-LABEL: func.func @slice_constant_column
// CHECK-SAME: (%[[ROW:.*]]: index)
// CHECK-NEXT: %[[VALUE:.*]] = ttkernel.experimental.constant_table_lookup
// CHECK-SAME: %[[ROW]], [2, 6, 10] : index
// CHECK-NEXT: return %[[VALUE]] : index
func.func @slice_constant_column(%row: index) -> index {
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  %row_offset = arith.muli %row, %c4 : index
  %index = arith.addi %row_offset, %c2 : index
  %value = ttkernel.experimental.constant_table_lookup %index,
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] : index
  return %value : index
}

// An incomplete final row prevents column slicing.
// CHECK-LABEL: func.func @retain_incomplete_table
// CHECK-SAME: (%[[ROW:.*]]: index)
// CHECK: %[[ROW_OFFSET:.*]] = arith.muli %[[ROW]], %{{.*}} : index
// CHECK-NEXT: %[[INDEX:.*]] = arith.addi %[[ROW_OFFSET]], %{{.*}} : index
// CHECK-NEXT: %[[VALUE:.*]] = ttkernel.experimental.constant_table_lookup
// CHECK-SAME: %[[INDEX]], [0, 1, 2, 3, 4, 5] : index
func.func @retain_incomplete_table(%row: index) -> index {
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  %row_offset = arith.muli %row, %c4 : index
  %index = arith.addi %row_offset, %c2 : index
  %value = ttkernel.experimental.constant_table_lookup %index,
      [0, 1, 2, 3, 4, 5] : index
  return %value : index
}
