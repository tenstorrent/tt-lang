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

// Context values compose through the integer expressions used by launch-node
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
      %same_block = arith.subi %iteration, %iteration {
        test.expected_count = 3 : i64,
        test.label = "induction_dependent_branch_same_block"
      } : index
    }
  }
  return
}
// CHECK-LABEL: induction_dependent_branch = 3
// CHECK-NEXT: induction_dependent_branch_same_block = 3

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

// An induction-dependent branch forces enumeration with unsigned loop semantics.
func.func @unsigned_loop_enumeration() {
  %lower = arith.constant -2 : i8
  %upper = arith.constant -1 : i8
  %step = arith.constant 1 : i8
  scf.for unsigned %iteration = %lower to %upper step %step : i8 {
    %selected = arith.cmpi ult, %iteration, %upper : i8
    scf.if %selected {
      %target = arith.addi %iteration, %iteration {
        test.expected_count = 1 : i64,
        test.label = "unsigned_loop_enumeration"
      } : i8
    }
  }
  return
}
// CHECK-LABEL: unsigned_loop_enumeration = 1

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

// An unconditional chain executes every block once.
func.func @multi_block_chain() {
  %zero = arith.constant 0 : index
  cf.br ^middle

^middle:
  %target = arith.addi %zero, %zero {
    test.expected_count = 1 : i64,
    test.label = "multi_block_chain"
  } : index
  cf.br ^exit

^exit:
  return
}
// CHECK-LABEL: multi_block_chain = 1

// A constant condition proves both the selected and unselected block counts.
func.func @constant_block_branch() {
  %true = arith.constant true
  %zero = arith.constant 0 : index
  cf.cond_br %true, ^selected, ^unselected

^selected:
  %selected_target = arith.addi %zero, %zero {
    test.expected_count = 1 : i64,
    test.label = "constant_block_branch_selected"
  } : index
  cf.br ^merge

^unselected:
  %unselected_target = arith.addi %zero, %zero {
    test.expected_count = 0 : i64,
    test.label = "constant_block_branch_unselected"
  } : index
  cf.br ^merge

^merge:
  %merge_target = arith.addi %zero, %zero {
    test.expected_count = 1 : i64,
    test.label = "constant_block_branch_merge"
  } : index
  return
}
// CHECK-LABEL: constant_block_branch_selected = 1
// CHECK-LABEL: constant_block_branch_unselected = 0
// CHECK-LABEL: constant_block_branch_merge = 1

// Sparse constant propagation forwards a selector through a block argument to
// prove the selected and unselected structured regions.
func.func @block_argument_region_selector() {
  %true = arith.constant true
  %zero = arith.constant 0 : index
  cf.br ^select(%true : i1)

^select(%condition: i1):
  scf.if %condition {
    %selected = arith.addi %zero, %zero {
      test.expected_count = 1 : i64,
      test.label = "block_argument_region_selector_selected"
    } : index
  } else {
    %unselected = arith.addi %zero, %zero {
      test.expected_count = 0 : i64,
      test.label = "block_argument_region_selector_unselected"
    } : index
  }
  return
}
// CHECK-LABEL: block_argument_region_selector_selected = 1
// CHECK-NEXT: block_argument_region_selector_unselected = 0

// Equal constants from runtime-selected predecessor blocks remain constant at
// their merge and select the nested structured region.
func.func @joined_block_argument_region_selector(%runtime_condition: i1) {
  %false = arith.constant false
  %zero = arith.constant 0 : index
  cf.cond_br %runtime_condition, ^left, ^right

^left:
  cf.br ^merge(%false : i1)

^right:
  cf.br ^merge(%false : i1)

^merge(%condition: i1):
  scf.if %condition {
    %target = arith.addi %zero, %zero {
      test.expected_count = 0 : i64,
      test.label = "joined_block_argument_region_selector"
    } : index
  }
  return
}
// CHECK-LABEL: joined_block_argument_region_selector = 0

// Sparse constant propagation preserves a loop-carried constant used by a
// nested structured region.
func.func @loop_carried_region_selector() {
  %false = arith.constant false
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %three = arith.constant 3 : index
  %result = scf.for %iteration = %zero to %three step %one
      iter_args(%condition = %false) -> i1 {
    scf.if %condition {
      %target = arith.addi %iteration, %iteration {
        test.expected_count = 0 : i64,
        test.label = "loop_carried_region_selector"
      } : index
    }
    scf.yield %condition : i1
  }
  return
}
// CHECK-LABEL: loop_carried_region_selector = 0

// The merge post-dominates the entry and therefore executes once even though
// neither branch count is exact.
func.func @dynamic_block_diamond(%condition: i1) {
  %zero = arith.constant 0 : index
  cf.cond_br %condition, ^then, ^else

^then:
  %then_target = arith.addi %zero, %zero {
    test.expected_count = "unknown",
    test.label = "dynamic_block_diamond_then"
  } : index
  cf.br ^merge

^else:
  %else_target = arith.addi %zero, %zero {
    test.expected_count = "unknown",
    test.label = "dynamic_block_diamond_else"
  } : index
  cf.br ^merge

^merge:
  %merge_target = arith.addi %zero, %zero {
    test.expected_count = 1 : i64,
    test.label = "dynamic_block_diamond_merge"
  } : index
  return
}
// CHECK-LABEL: dynamic_block_diamond_then = unknown
// CHECK-LABEL: dynamic_block_diamond_else = unknown
// CHECK-LABEL: dynamic_block_diamond_merge = 1

// Parallel successor edges to the same block still produce one block
// invocation.
func.func @duplicate_successor(%condition: i1) {
  %zero = arith.constant 0 : index
  cf.cond_br %condition, ^merge, ^merge

^merge:
  %target = arith.addi %zero, %zero {
    test.expected_count = 1 : i64,
    test.label = "duplicate_successor"
  } : index
  return
}
// CHECK-LABEL: duplicate_successor = 1

// A conditional block has an unknown count while its merge still executes
// once.
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
  %merge_target = arith.addi %zero, %zero {
    test.expected_count = 1 : i64,
    test.label = "multi_block_merge"
  } : index
  return
}
// CHECK-LABEL: multi_block = unknown
// CHECK-LABEL: multi_block_merge = 1

// A disconnected block has an exact count of zero.
func.func @unreachable_block() {
  %zero = arith.constant 0 : index
  return

^unreachable:
  %target = arith.addi %zero, %zero {
    test.expected_count = 0 : i64,
    test.label = "unreachable_block"
  } : index
  return
}
// CHECK-LABEL: unreachable_block = 0

// A block predicate derived from an enclosing induction variable is
// enumerated with the loop.
func.func @induction_dependent_block_branch() {
  %zero = arith.constant 0 : index
  %upper = arith.constant 8 : index
  %three = arith.constant 3 : index
  %step = arith.constant 1 : index
  scf.for %iteration = %zero to %upper step %step {
    %selected = arith.cmpi slt, %iteration, %three : index
    scf.execute_region {
      cf.cond_br %selected, ^selected, ^exit

    ^selected:
      %target = arith.addi %iteration, %iteration {
        test.expected_count = 3 : i64,
        test.label = "induction_dependent_block_branch"
      } : index
      cf.br ^exit

    ^exit:
      scf.yield
    }
  }
  return
}
// CHECK-LABEL: induction_dependent_block_branch = 3

// A possible CFG cycle does not prove that its blocks or subsequent blocks
// execute a finite number of times.
func.func @possible_block_cycle(%condition: i1) {
  %zero = arith.constant 0 : index
  %entry_target = arith.addi %zero, %zero {
    test.expected_count = 1 : i64,
    test.label = "possible_block_cycle_entry"
  } : index
  cf.br ^loop

^loop:
  %loop_target = arith.addi %zero, %zero {
    test.expected_count = "unknown",
    test.label = "possible_block_cycle_loop"
  } : index
  cf.cond_br %condition, ^loop, ^exit

^exit:
  %exit_target = arith.addi %zero, %zero {
    test.expected_count = "unknown",
    test.label = "possible_block_cycle_exit"
  } : index
  return
}
// CHECK-LABEL: possible_block_cycle_entry = 1
// CHECK-LABEL: possible_block_cycle_loop = unknown
// CHECK-LABEL: possible_block_cycle_exit = unknown

// A possible non-exiting cycle makes the sibling target unknown while an
// unconditional prefix remains exact.
func.func @non_exiting_cycle_sibling(%condition: i1) {
  %zero = arith.constant 0 : index
  cf.br ^prefix

^prefix:
  %prefix_target = arith.addi %zero, %zero {
    test.expected_count = 1 : i64,
    test.label = "non_exiting_cycle_sibling_prefix"
  } : index
  cf.cond_br %condition, ^spin, ^target

^spin:
  cf.br ^spin

^target:
  %target = arith.addi %zero, %zero {
    test.expected_count = "unknown",
    test.label = "non_exiting_cycle_sibling_target"
  } : index
  return
}
// CHECK-LABEL: non_exiting_cycle_sibling_prefix = 1
// CHECK-LABEL: non_exiting_cycle_sibling_target = unknown

// A non-exiting cycle spanning multiple blocks has the same effect as a
// self-loop on an alternative branch.
func.func @non_exiting_multi_block_cycle(%condition: i1) {
  %zero = arith.constant 0 : index
  cf.cond_br %condition, ^spin_a, ^target

^spin_a:
  cf.br ^spin_b

^spin_b:
  cf.br ^spin_a

^target:
  %target = arith.addi %zero, %zero {
    test.expected_count = "unknown",
    test.label = "non_exiting_multi_block_cycle"
  } : index
  return
}
// CHECK-LABEL: non_exiting_multi_block_cycle = unknown

// A cycle with two entry edges is irreducible. Its blocks and any possible
// exit remain unknown while the unconditional prefix executes once.
func.func @irreducible_cycle(%entry_condition: i1, %exit_condition: i1) {
  %zero = arith.constant 0 : index
  %prefix = arith.addi %zero, %zero {
    test.expected_count = 1 : i64,
    test.label = "irreducible_cycle_prefix"
  } : index
  cf.cond_br %entry_condition, ^left, ^right

^left:
  %left_target = arith.addi %zero, %zero {
    test.expected_count = "unknown",
    test.label = "irreducible_cycle_left"
  } : index
  cf.cond_br %exit_condition, ^exit, ^right

^right:
  %right_target = arith.addi %zero, %zero {
    test.expected_count = "unknown",
    test.label = "irreducible_cycle_right"
  } : index
  cf.cond_br %exit_condition, ^exit, ^left

^exit:
  %exit_target = arith.addi %zero, %zero {
    test.expected_count = "unknown",
    test.label = "irreducible_cycle_exit"
  } : index
  return
}
// CHECK-LABEL: irreducible_cycle_prefix = 1
// CHECK-LABEL: irreducible_cycle_left = unknown
// CHECK-LABEL: irreducible_cycle_right = unknown
// CHECK-LABEL: irreducible_cycle_exit = unknown

// A disconnected cycle cannot change the count of a block reached directly
// from the entry, even when the cycle also has an edge to that block.
func.func @disconnected_cycle_to_reachable_block(%condition: i1) {
  %zero = arith.constant 0 : index
  cf.br ^target

^cycle:
  cf.cond_br %condition, ^cycle, ^target

^target:
  %target = arith.addi %zero, %zero {
    test.expected_count = 1 : i64,
    test.label = "disconnected_cycle_to_reachable_block"
  } : index
  return
}
// CHECK-LABEL: disconnected_cycle_to_reachable_block = 1

// A non-exiting CFG cycle inside a structured loop must not produce an exact
// sibling count multiplied by the loop trip count.
func.func @nested_non_exiting_cycle(%condition: i1) {
  %zero = arith.constant 0 : index
  %upper = arith.constant 4 : index
  %step = arith.constant 1 : index
  scf.for %iteration = %zero to %upper step %step {
    scf.execute_region {
      cf.cond_br %condition, ^spin, ^target

    ^spin:
      cf.br ^spin

    ^target:
      %target = arith.addi %iteration, %iteration {
        test.expected_count = "unknown",
        test.label = "nested_non_exiting_cycle"
      } : index
      scf.yield
    }
  }
  return
}
// CHECK-LABEL: nested_non_exiting_cycle = unknown

// Removing a cycle with a constant branch restores exact downstream counts.
func.func @unselected_block_cycle() {
  %false = arith.constant false
  %zero = arith.constant 0 : index
  cf.br ^loop

^loop:
  cf.cond_br %false, ^loop, ^exit

^exit:
  %target = arith.addi %zero, %zero {
    test.expected_count = 1 : i64,
    test.label = "unselected_block_cycle"
  } : index
  return
}
// CHECK-LABEL: unselected_block_cycle = 1

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

// The enumeration limit permits exactly the configured number of iterations.
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

// Zero extension preserves the unsigned value of a narrow bit pattern.
func.func @zero_extension() {
  %zero = arith.constant 0 : i16
  %narrow = arith.constant -2 : i8
  %upper = arith.extui %narrow : i8 to i16
  %step = arith.constant 1 : i16
  scf.for %iteration = %zero to %upper step %step : i16 {
    %target = arith.addi %iteration, %iteration {
      test.expected_count = 254 : i64,
      test.label = "zero_extension"
    } : i16
  }
  return
}
// CHECK-LABEL: zero_extension = 254

// Sign extension preserves the negative value and produces a zero-trip loop.
func.func @sign_extension() {
  %zero = arith.constant 0 : i16
  %narrow = arith.constant -2 : i8
  %upper = arith.extsi %narrow : i8 to i16
  %step = arith.constant 1 : i16
  scf.for %iteration = %zero to %upper step %step : i16 {
    %target = arith.addi %iteration, %iteration {
      test.expected_count = 0 : i64,
      test.label = "sign_extension"
    } : i16
  }
  return
}
// CHECK-LABEL: sign_extension = 0

// Truncation uses the narrowed bit pattern when computing a trip count.
func.func @integer_truncation() {
  %wide = arith.constant 261 : i32
  %upper = arith.trunci %wide : i32 to i8
  %zero = arith.constant 0 : i8
  %step = arith.constant 1 : i8
  scf.for %iteration = %zero to %upper step %step : i8 {
    %target = arith.addi %iteration, %iteration {
      test.expected_count = 5 : i64,
      test.label = "integer_truncation"
    } : i8
  }
  return
}
// CHECK-LABEL: integer_truncation = 5

// Some fold hooks first replace an operand and return their own result. The
// evaluator repeats that fold on a clone and leaves the input IR unchanged.
func.func @in_place_fold_before_constant_fold() {
  %narrow = arith.constant 5 : i16
  %extended = arith.extui %narrow : i16 to i32
  %upper = arith.trunci %extended : i32 to i8
  %zero = arith.constant 0 : i8
  %step = arith.constant 1 : i8
  scf.for %iteration = %zero to %upper step %step : i8 {
    %target = arith.addi %iteration, %iteration {
      test.expected_count = 5 : i64,
      test.label = "in_place_fold_before_constant_fold"
    } : i8
  }
  return
}
// CHECK-LABEL: in_place_fold_before_constant_fold = 5

// A fold may replace its result with another SSA value whose exact value was
// evaluated independently.
func.func @external_fold_replacement(%condition: i1) {
  %five = arith.constant 5 : index
  %upper = arith.select %condition, %five, %five : index
  %zero = arith.constant 0 : index
  %step = arith.constant 1 : index
  scf.for %iteration = %zero to %upper step %step {
    %target = arith.addi %iteration, %iteration {
      test.expected_count = 5 : i64,
      test.label = "external_fold_replacement"
    } : index
  }
  return
}
// CHECK-LABEL: external_fold_replacement = 5

// Unsigned index casting zero-extends a narrow value with its sign bit set.
func.func @unsigned_index_cast() {
  %narrow = arith.constant -2 : i8
  %upper = arith.index_castui %narrow : i8 to index
  %zero = arith.constant 0 : index
  %step = arith.constant 1 : index
  scf.for %iteration = %zero to %upper step %step {
    %target = arith.addi %iteration, %iteration {
      test.expected_count = 254 : i64,
      test.label = "unsigned_index_cast"
    } : index
  }
  return
}
// CHECK-LABEL: unsigned_index_cast = 254

// A non-unit step uses the ceiling of the iteration-distance quotient.
func.func @non_unit_step() {
  %zero = arith.constant 0 : index
  %upper = arith.constant 10 : index
  %step = arith.constant 3 : index
  scf.for %iteration = %zero to %upper step %step {
    %target = arith.addi %iteration, %iteration {
      test.expected_count = 4 : i64,
      test.label = "non_unit_step"
    } : index
  }
  return
}
// CHECK-LABEL: non_unit_step = 4

// Unsigned comparison treats an all-ones bit pattern as greater than zero.
func.func @unsigned_predicate() {
  %all_ones = arith.constant -1 : i32
  %zero = arith.constant 0 : i32
  %selected = arith.cmpi ugt, %all_ones, %zero : i32
  scf.if %selected {
    %target = arith.addi %zero, %zero {
      test.expected_count = 1 : i64,
      test.label = "unsigned_predicate"
    } : i32
  }
  return
}
// CHECK-LABEL: unsigned_predicate = 1

// A constant false condition selects the else region exactly once.
func.func @selected_else_region() {
  %false = arith.constant false
  %zero = arith.constant 0 : index
  scf.if %false {
  } else {
    %target = arith.addi %zero, %zero {
      test.expected_count = 1 : i64,
      test.label = "selected_else_region"
    } : index
  }
  return
}
// CHECK-LABEL: selected_else_region = 1

// An unmatched index-switch selector selects the default region exactly once.
func.func @selected_index_switch_default() {
  %selector = arith.constant 7 : index
  %zero = arith.constant 0 : index
  scf.index_switch %selector
  case 1 {
    scf.yield
  }
  default {
    %target = arith.addi %zero, %zero {
      test.expected_count = 1 : i64,
      test.label = "selected_index_switch_default"
    } : index
    scf.yield
  }
  return
}
// CHECK-LABEL: selected_index_switch_default = 1

// Nested enumeration cannot consume the remaining budget and then underflow
// it in a later outer iteration.
func.func @nested_enumeration_budget() attributes {
    test.max_enumerated_iterations = 3 : i64} {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %three = arith.constant 3 : index
  scf.for %outer = %zero to %three step %one {
    scf.for %middle = %zero to %outer step %one {
      scf.for %inner = %zero to %middle step %one {
        %target = arith.addi %outer, %inner {
          test.expected_count = "unknown",
          test.label = "nested_enumeration_budget"
        } : index
      }
    }
  }
  return
}
// CHECK-LABEL: nested_enumeration_budget = unknown

// A failed induction-independent attempt consumes the iterations it examines,
// so speculative work is included in the enumeration limit.
func.func @failed_attempt_consumes_budget() attributes {
    test.max_enumerated_iterations = 2 : i64} {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  scf.for %outer = %zero to %two step %one {
    scf.for %inner = %zero to %two step %one {
      %selected = arith.cmpi eq, %outer, %zero : index
      scf.if %selected {
        %target = arith.addi %outer, %inner {
          test.expected_count = "unknown",
          test.label = "failed_attempt_consumes_budget"
        } : index
      }
    }
  }
  return
}
// CHECK-LABEL: failed_attempt_consumes_budget = unknown

// Shared expression operands are evaluated once per induction environment.
// The depth makes non-memoized recursive evaluation prohibitively expensive.
func.func @shared_expression_dag() {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %limit = arith.constant 100000000 : index
  scf.for %iteration = %zero to %two step %one {
    %value0 = arith.addi %iteration, %one : index
    %value1 = arith.addi %value0, %value0 : index
    %value2 = arith.addi %value1, %value1 : index
    %value3 = arith.addi %value2, %value2 : index
    %value4 = arith.addi %value3, %value3 : index
    %value5 = arith.addi %value4, %value4 : index
    %value6 = arith.addi %value5, %value5 : index
    %value7 = arith.addi %value6, %value6 : index
    %value8 = arith.addi %value7, %value7 : index
    %value9 = arith.addi %value8, %value8 : index
    %value10 = arith.addi %value9, %value9 : index
    %value11 = arith.addi %value10, %value10 : index
    %value12 = arith.addi %value11, %value11 : index
    %value13 = arith.addi %value12, %value12 : index
    %value14 = arith.addi %value13, %value13 : index
    %value15 = arith.addi %value14, %value14 : index
    %value16 = arith.addi %value15, %value15 : index
    %value17 = arith.addi %value16, %value16 : index
    %value18 = arith.addi %value17, %value17 : index
    %value19 = arith.addi %value18, %value18 : index
    %value20 = arith.addi %value19, %value19 : index
    %value21 = arith.addi %value20, %value20 : index
    %value22 = arith.addi %value21, %value21 : index
    %selected = arith.cmpi slt, %value22, %limit : index
    scf.if %selected {
      %target = arith.addi %iteration, %iteration {
        test.expected_count = 2 : i64,
        test.label = "shared_expression_dag"
      } : index
    }
  }
  return
}
// CHECK-LABEL: shared_expression_dag = 2

// Values depending on two enclosing induction variables are recomputed for
// each environment, and both bindings are restored after enumeration.
func.func @two_induction_variables() {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  scf.for %outer = %zero to %two step %one {
    scf.for %inner = %zero to %two step %one {
      %sum = arith.addi %outer, %inner : index
      %selected = arith.cmpi slt, %sum, %two : index
      scf.if %selected {
        %target = arith.addi %outer, %inner {
          test.expected_count = 3 : i64,
          test.label = "two_induction_variables"
        } : index
      }
    }
  }
  return
}
// CHECK-LABEL: two_induction_variables = 3
