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
// CHECK: straight_line = 1

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
// CHECK: large_rectangular_nest = 1000000000000

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
// CHECK: context_bound = 7

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
// CHECK: dependent_inner_bound = 6

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
// CHECK: induction_dependent_branch = 3

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
// CHECK: unselected_region = 0

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
// CHECK: selected_index_switch = 1

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
// CHECK: execute_region = 1

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
// CHECK: dynamic_loop_bound = unknown

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
// CHECK: dynamic_branch = unknown

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
// CHECK: unsupported_loop = unknown

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
// CHECK: multi_block = unknown

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
// CHECK: enumeration_limit = unknown

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
// CHECK: count_overflow = unknown
