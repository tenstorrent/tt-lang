// Negative tests for ttl-lower-scalar-cmpf: unsupported predicates emit errors.
// RUN: not ttlang-opt --ttl-lower-scalar-cmpf --split-input-file %s 2>&1 | FileCheck %s

// -----

// Unsupported predicate oge.
// CHECK: error: 'arith.cmpf' op unsupported cmpf predicate for soft-float lowering
module {
  func.func @cmpf_oge_unsupported(%a_int: i32, %b_int: i32) -> i1 {
    %a = builtin.unrealized_conversion_cast %a_int : i32 to f32
    %b = builtin.unrealized_conversion_cast %b_int : i32 to f32
    %cmp = arith.cmpf oge, %a, %b : f32
    return %cmp : i1
  }
}

// -----

// Unsupported predicate ole.
// CHECK: error: 'arith.cmpf' op unsupported cmpf predicate for soft-float lowering
module {
  func.func @cmpf_ole_unsupported(%a_int: i32, %b_int: i32) -> i1 {
    %a = builtin.unrealized_conversion_cast %a_int : i32 to f32
    %b = builtin.unrealized_conversion_cast %b_int : i32 to f32
    %cmp = arith.cmpf ole, %a, %b : f32
    return %cmp : i1
  }
}
