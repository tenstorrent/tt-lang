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

// -----

// Unsupported float type f16 (neither f32 nor bf16).
// CHECK: error: 'arith.cmpf' op unsupported float type for scalar comparison:
module {
  func.func @cmpf_f16_unsupported(%a: f16, %b: f16) -> i1 {
    %cmp = arith.cmpf ogt, %a, %b : f16
    return %cmp : i1
  }
}

// -----

// Unresolvable operand: arith.addf result is not from raw_element_read or constant.
// CHECK: error: 'arith.cmpf' op could not resolve float operand to integer bit pattern
module {
  func.func @cmpf_unresolvable_operand(%a_int: i32, %x: f32, %y: f32) -> i1 {
    %sum = arith.addf %x, %y : f32
    %a = builtin.unrealized_conversion_cast %a_int : i32 to f32
    %cmp = arith.cmpf ogt, %a, %sum : f32
    return %cmp : i1
  }
}

// -----

// bf16 unsupported predicate oge.
// CHECK: error: 'arith.cmpf' op unsupported cmpf predicate for soft-float lowering
module {
  func.func @cmpf_oge_bf16_unsupported(%a_int: i16, %b_int: i16) -> i1 {
    %a = builtin.unrealized_conversion_cast %a_int : i16 to bf16
    %b = builtin.unrealized_conversion_cast %b_int : i16 to bf16
    %cmp = arith.cmpf oge, %a, %b : bf16
    return %cmp : i1
  }
}

// -----

// Unordered predicate uno (representative of unordered predicate family).
// CHECK: error: 'arith.cmpf' op unsupported cmpf predicate for soft-float lowering
module {
  func.func @cmpf_uno_unsupported(%a_int: i32, %b_int: i32) -> i1 {
    %a = builtin.unrealized_conversion_cast %a_int : i32 to f32
    %b = builtin.unrealized_conversion_cast %b_int : i32 to f32
    %cmp = arith.cmpf uno, %a, %b : f32
    return %cmp : i1
  }
}
