// RUN: ttlang-execution-count-test %s | FileCheck %s

// This file tests exact and unknown operation counts in structured control
// flow. Each annotated operation is checked relative to one function-body
// invocation.

// A straight-line operation executes once.
func.func @straight_line() {
  %zero = arith.constant 0 : index
  %target = arith.addi %zero, %zero {
    test.expected_count = 1 : i64,
    test.label = "straight_line"
  } : index
  return
}
// CHECK-LABEL: straight_line = 1

// Large rectangular loop nests use trip-count multiplication, not iteration
// enumeration.
func.func @large_rectangular_nest() attributes {
    test.max_enumerated_iterations = 1 : i64} {
  %zero = arith.constant 0 : index
  %upper = arith.constant 1000000 : index
  %step = arith.constant 1 : index
  scf.for %outer = %zero to %upper step %step {
    scf.for %inner = %zero to %upper step %step {
      %target = arith.addi %outer, %inner {
        test.expected_count = 1000000000000 : i64,
        test.label = "large_rectangular_nest"
      } : index
    }
  }
  return
}
// CHECK-LABEL: large_rectangular_nest = 1000000000000

// A function-argument value supplied by the analysis context can define an
// exact trip count.
func.func @context_bound(%upper: index {test.value = 7 : i64}) {
  %zero = arith.constant 0 : index
  %step = arith.constant 1 : index
  scf.for %iteration = %zero to %upper step %step {
    %target = arith.addi %iteration, %iteration {
      test.expected_count = 7 : i64,
      test.label = "context_bound"
    } : index
  }
  return
}
// CHECK-LABEL: context_bound = 7

// A consumer value with the wrong bit width is not a valid fact for the SSA
// value and cannot prove the loop count.
func.func @context_value_width_mismatch(
    %upper: index {test.value = 7 : i32}) {
  %zero = arith.constant 0 : index
  %step = arith.constant 1 : index
  scf.for %iteration = %zero to %upper step %step {
    %target = arith.addi %iteration, %iteration {
      test.expected_count = "unknown",
      test.label = "context_value_width_mismatch"
    } : index
  }
  return
}
// CHECK-LABEL: context_value_width_mismatch = unknown

// Context values compose through the integer expressions used by launch-role
// predicates and loop bounds.
func.func @context_integer_expression(
    %coordinate: index {test.value = 2 : i64}) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %true = arith.constant true
  %false = arith.constant false
  %upper_base = arith.addi %coordinate, %two : index
  %upper = arith.muli %upper_base, %one : index
  %coordinate_i32 = arith.index_cast %coordinate : index to i32
  %one_i32 = arith.constant 1 : i32
  %difference = arith.subi %coordinate_i32, %one_i32 : i32
  %is_one = arith.cmpi eq, %difference, %one_i32 : i32
  %selected_or_false = arith.ori %is_one, %false : i1
  %selected_xor_false = arith.xori %selected_or_false, %false : i1
  %selected = arith.andi %selected_xor_false, %true : i1
  scf.for %iteration = %zero to %upper step %one {
    scf.if %selected {
      %target = arith.addi %iteration, %iteration {
        test.expected_count = 4 : i64,
        test.label = "context_integer_expression"
      } : index
    }
  }
  return
}
// CHECK-LABEL: context_integer_expression = 4

// A loop bound derived from an outer induction variable is enumerated when the
// complete nest remains compile-time evaluable.
func.func @dependent_inner_bound() {
  %zero = arith.constant 0 : index
  %upper = arith.constant 4 : index
  %step = arith.constant 1 : index
  scf.for %outer = %zero to %upper step %step {
    scf.for %inner = %zero to %outer step %step {
      %target = arith.addi %outer, %inner {
        test.expected_count = 6 : i64,
        test.label = "dependent_inner_bound"
      } : index
    }
  }
  return
}
// CHECK-LABEL: dependent_inner_bound = 6

// An induction-dependent branch is counted once for each selected iteration.
func.func @induction_dependent_branch() {
  %zero = arith.constant 0 : index
  %upper = arith.constant 8 : index
  %three = arith.constant 3 : index
  %step = arith.constant 1 : index
  scf.for %iteration = %zero to %upper step %step {
    %selected = arith.cmpi slt, %iteration, %three : index
    scf.if %selected {
      %target = arith.addi %iteration, %iteration {
        test.expected_count = 3 : i64,
        test.label = "induction_dependent_branch"
      } : index
    }
  }
  return
}
// CHECK-LABEL: induction_dependent_branch = 3

// An unselected region has an exact count of zero.
func.func @unselected_region() {
  %false = arith.constant false
  %zero = arith.constant 0 : index
  scf.if %false {
    %target = arith.addi %zero, %zero {
      test.expected_count = 0 : i64,
      test.label = "unselected_region"
    } : index
  }
  return
}
// CHECK-LABEL: unselected_region = 0

// Nested coordinate-like predicates exclude operations in an unselected else
// branch even when the inner condition would select its region independently.
func.func @nested_unselected_region(
    %coordinate: index {test.value = 0 : i64}) {
  %zero = arith.constant 0 : index
  %five = arith.constant 5 : index
  %is_zero = arith.cmpi eq, %coordinate, %zero : index
  scf.if %is_zero {
  } else {
    %is_five = arith.cmpi eq, %coordinate, %five : index
    scf.if %is_five {
      %target = arith.addi %coordinate, %coordinate {
        test.expected_count = 0 : i64,
        test.label = "nested_unselected_region"
      } : index
    }
  }
  return
}
// CHECK-LABEL: nested_unselected_region = 0

// SCF structured branch operations with conservative invocation lower bounds
// still have exact counts when their selection semantics are evaluable.
func.func @selected_index_switch(
    %selector: index {test.value = 2 : i64}) {
  %zero = arith.constant 0 : index
  scf.index_switch %selector
  case 1 {
    scf.yield
  }
  case 2 {
    %target = arith.addi %zero, %zero {
      test.expected_count = 1 : i64,
      test.label = "selected_index_switch"
    } : index
    scf.yield
  }
  default {
    scf.yield
  }
  return
}
// CHECK-LABEL: selected_index_switch = 1

// A constant selector proves that operations in another case do not execute.
func.func @unselected_index_switch(
    %selector: index {test.value = 2 : i64}) {
  %zero = arith.constant 0 : index
  scf.index_switch %selector
  case 1 {
    %target = arith.addi %zero, %zero {
      test.expected_count = 0 : i64,
      test.label = "unselected_index_switch"
    } : index
    scf.yield
  }
  case 2 {
    scf.yield
  }
  default {
    scf.yield
  }
  return
}
// CHECK-LABEL: unselected_index_switch = 0

// A structured region with unconditional entry executes once.
func.func @execute_region() {
  %zero = arith.constant 0 : index
  scf.execute_region {
    %target = arith.addi %zero, %zero {
      test.expected_count = 1 : i64,
      test.label = "execute_region"
    } : index
    scf.yield
  }
  return
}
// CHECK-LABEL: execute_region = 1

// A consumer can define exact invocation semantics for a non-loop region.
func.func @context_region_invocation_count() {
  %zero = arith.constant 0 : index
  scf.execute_region {
    %target = arith.addi %zero, %zero {
      test.expected_count = 3 : i64,
      test.label = "context_region_invocation_count"
    } : index
    scf.yield
  } {test.region_invocation_count = 3 : i64}
  return
}
// CHECK-LABEL: context_region_invocation_count = 3

// An exact zero from an outer loop makes dynamic nested control irrelevant.
func.func @zero_trip_outer_loop(%condition: i1) {
  %zero = arith.constant 0 : index
  %step = arith.constant 1 : index
  scf.for %iteration = %zero to %zero step %step {
    scf.if %condition {
      %target = arith.addi %iteration, %iteration {
        test.expected_count = 0 : i64,
        test.label = "zero_trip_outer_loop"
      } : index
    }
  }
  return
}
// CHECK-LABEL: zero_trip_outer_loop = 0

// Signed and unsigned loops use the comparison semantics encoded by scf.for.
func.func @signed_loop_crossing_zero() {
  %lower = arith.constant -2 : i8
  %upper = arith.constant 2 : i8
  %step = arith.constant 1 : i8
  scf.for %iteration = %lower to %upper step %step : i8 {
    %target = arith.addi %iteration, %iteration {
      test.expected_count = 4 : i64,
      test.label = "signed_loop_crossing_zero"
    } : i8
  }
  return
}
// CHECK-LABEL: signed_loop_crossing_zero = 4

// Unsigned comparison makes the same bit patterns a zero-trip loop.
func.func @unsigned_loop_crossing_zero() {
  %lower = arith.constant -2 : i8
  %upper = arith.constant 2 : i8
  %step = arith.constant 1 : i8
  scf.for unsigned %iteration = %lower to %upper step %step : i8 {
    %target = arith.addi %iteration, %iteration {
      test.expected_count = 0 : i64,
      test.label = "unsigned_loop_crossing_zero"
    } : i8
  }
  return
}
// CHECK-LABEL: unsigned_loop_crossing_zero = 0

// Runtime loop bounds and branch conditions produce unknown counts.
func.func @dynamic_loop_bound(%upper: index) {
  %zero = arith.constant 0 : index
  %step = arith.constant 1 : index
  scf.for %iteration = %zero to %upper step %step {
    %target = arith.addi %iteration, %iteration {
      test.expected_count = "unknown",
      test.label = "dynamic_loop_bound"
    } : index
  }
  return
}
// CHECK-LABEL: dynamic_loop_bound = unknown

// A runtime branch condition cannot prove an exact count.
func.func @dynamic_branch(%condition: i1) {
  %zero = arith.constant 0 : index
  scf.if %condition {
    %target = arith.addi %zero, %zero {
      test.expected_count = "unknown",
      test.label = "dynamic_branch"
    } : index
  }
  return
}
// CHECK-LABEL: dynamic_branch = unknown

// A loop without an exact trip-count model remains unknown.
func.func @unsupported_loop(%upper: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %result = scf.while (%iteration = %zero) : (index) -> index {
    %condition = arith.cmpi slt, %iteration, %upper : index
    scf.condition(%condition) %iteration : index
  } do {
  ^bb0(%iteration: index):
    %target = arith.addi %iteration, %one {
      test.expected_count = "unknown",
      test.label = "unsupported_loop"
    } : index
    scf.yield %target : index
  }
  return
}
// CHECK-LABEL: unsupported_loop = unknown

// Multi-block region control flow is unknown until block execution counts are
// proven.
func.func @multi_block(%condition: i1) {
  %zero = arith.constant 0 : index
  cf.cond_br %condition, ^selected, ^exit

^selected:
  %target = arith.addi %zero, %zero {
    test.expected_count = "unknown",
    test.label = "multi_block"
  } : index
  cf.br ^exit

^exit:
  return
}
// CHECK-LABEL: multi_block = unknown

// Enumeration limits and arithmetic overflow produce unknown rather than an
// incomplete or wrapped count.
func.func @enumeration_limit() attributes {
    test.max_enumerated_iterations = 2 : i64} {
  %zero = arith.constant 0 : index
  %upper = arith.constant 8 : index
  %four = arith.constant 4 : index
  %step = arith.constant 1 : index
  scf.for %iteration = %zero to %upper step %step {
    %selected = arith.cmpi slt, %iteration, %four : index
    scf.if %selected {
      %target = arith.addi %iteration, %iteration {
        test.expected_count = "unknown",
        test.label = "enumeration_limit"
      } : index
    }
  }
  return
}
// CHECK-LABEL: enumeration_limit = unknown

// The proof limit permits exactly the configured number of iterations.
func.func @enumeration_limit_boundary() attributes {
    test.max_enumerated_iterations = 8 : i64} {
  %zero = arith.constant 0 : index
  %upper = arith.constant 8 : index
  %four = arith.constant 4 : index
  %step = arith.constant 1 : index
  scf.for %iteration = %zero to %upper step %step {
    %selected = arith.cmpi slt, %iteration, %four : index
    scf.if %selected {
      %target = arith.addi %iteration, %iteration {
        test.expected_count = 4 : i64,
        test.label = "enumeration_limit_boundary"
      } : index
    }
  }
  return
}
// CHECK-LABEL: enumeration_limit_boundary = 4

// A product larger than uint64 is unknown rather than wrapped.
func.func @count_overflow() {
  %zero = arith.constant 0 : index
  %upper = arith.constant 4294967296 : index
  %step = arith.constant 1 : index
  scf.for %outer = %zero to %upper step %step {
    scf.for %inner = %zero to %upper step %step {
      %target = arith.addi %outer, %inner {
        test.expected_count = "unknown",
        test.label = "count_overflow"
      } : index
    }
  }
  return
}
// CHECK-LABEL: count_overflow = unknown
