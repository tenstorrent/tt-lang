// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// Summary: Verifies constant-table lookup table and constant-index errors.

// An immutable table must contain at least one element.
func.func @reject_empty_table() -> index {
  %index = arith.constant 0 : index
  // expected-error @below {{requires at least one table value}}
  %value = ttkernel.experimental.constant_table_lookup %index, [] : index
  return %value : index
}

// -----

// Negative table entries cannot represent runtime argument values.
func.func @reject_negative_table_value() -> index {
  %index = arith.constant 0 : index
  // expected-error @below {{requires non-negative table values}}
  %value = ttkernel.experimental.constant_table_lookup %index, [-1, 2] : index
  return %value : index
}

// -----

// A negative constant index is outside the table bounds.
func.func @reject_negative_index() -> index {
  %index = arith.constant -1 : index
  // expected-error @below {{constant index -1 is outside the table bounds \[0, 2\)}}
  %value = ttkernel.experimental.constant_table_lookup %index, [1, 2] : index
  return %value : index
}

// -----

// A constant index equal to the table size is outside the table bounds.
func.func @reject_upper_bound_index() -> index {
  %index = arith.constant 2 : index
  // expected-error @below {{constant index 2 is outside the table bounds \[0, 2\)}}
  %value = ttkernel.experimental.constant_table_lookup %index, [1, 2] : index
  return %value : index
}
