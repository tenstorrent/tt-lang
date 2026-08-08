// Negative tests for ttkernel-lower-scalar-fp-types: unsupported predicates and
// float types cause legalization failure.
// RUN: ttlang-opt --ttkernel-lower-scalar-fp-types --verify-diagnostics --split-input-file %s

// -----

// Unsupported predicate oeq.
module {
  func.func @cmpf_oeq_unsupported(%a_int: i32, %b_int: i32) -> i1 {
    %a = builtin.unrealized_conversion_cast %a_int : i32 to f32
    %b = builtin.unrealized_conversion_cast %b_int : i32 to f32
    // expected-error @below {{failed to legalize operation 'arith.cmpf'}}
    %cmp = arith.cmpf oeq, %a, %b : f32
    return %cmp : i1
  }
}

// -----

// Unsupported predicate one.
module {
  func.func @cmpf_one_unsupported(%a_int: i32, %b_int: i32) -> i1 {
    %a = builtin.unrealized_conversion_cast %a_int : i32 to f32
    %b = builtin.unrealized_conversion_cast %b_int : i32 to f32
    // expected-error @below {{failed to legalize operation 'arith.cmpf'}}
    %cmp = arith.cmpf one, %a, %b : f32
    return %cmp : i1
  }
}

// -----

// Unsupported predicate oge.
module {
  func.func @cmpf_oge_unsupported(%a_int: i32, %b_int: i32) -> i1 {
    %a = builtin.unrealized_conversion_cast %a_int : i32 to f32
    %b = builtin.unrealized_conversion_cast %b_int : i32 to f32
    // expected-error @below {{failed to legalize operation 'arith.cmpf'}}
    %cmp = arith.cmpf oge, %a, %b : f32
    return %cmp : i1
  }
}

// -----

// Unsupported predicate ole.
module {
  func.func @cmpf_ole_unsupported(%a_int: i32, %b_int: i32) -> i1 {
    %a = builtin.unrealized_conversion_cast %a_int : i32 to f32
    %b = builtin.unrealized_conversion_cast %b_int : i32 to f32
    // expected-error @below {{failed to legalize operation 'arith.cmpf'}}
    %cmp = arith.cmpf ole, %a, %b : f32
    return %cmp : i1
  }
}

// -----

// Unsupported float type f16 (neither f32 nor bf16).
module {
  func.func @cmpf_f16_unsupported(%a_int: i16, %b_int: i16) -> i1 {
    %a = builtin.unrealized_conversion_cast %a_int : i16 to f16
    %b = builtin.unrealized_conversion_cast %b_int : i16 to f16
    // expected-error @below {{'arith.cmpf' op unsupported float type for scalar comparison; only f32 and bf16 are supported}}
    %cmp = arith.cmpf ogt, %a, %b : f16
    return %cmp : i1
  }
}

// -----

// bf16 unsupported predicate oge.
module {
  func.func @cmpf_oge_bf16_unsupported(%a_int: i16, %b_int: i16) -> i1 {
    %a = builtin.unrealized_conversion_cast %a_int : i16 to bf16
    %b = builtin.unrealized_conversion_cast %b_int : i16 to bf16
    // expected-error @below {{failed to legalize operation 'arith.cmpf'}}
    %cmp = arith.cmpf oge, %a, %b : bf16
    return %cmp : i1
  }
}

// -----

// Unordered predicate uno (representative of unordered predicate family).
module {
  func.func @cmpf_uno_unsupported(%a_int: i32, %b_int: i32) -> i1 {
    %a = builtin.unrealized_conversion_cast %a_int : i32 to f32
    %b = builtin.unrealized_conversion_cast %b_int : i32 to f32
    // expected-error @below {{failed to legalize operation 'arith.cmpf'}}
    %cmp = arith.cmpf uno, %a, %b : f32
    return %cmp : i1
  }
}
