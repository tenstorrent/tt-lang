// Verifies ttl-materialize-loop-state removes tensor-valued scf.for iter_args
// using accumulate stores for additive recurrences and compiler DFB state for
// general tensor recurrences.
//
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-materialize-loop-state))' --split-input-file | FileCheck %s

// Additive recurrence lowers to an accumulate store.
// CHECK-LABEL: func.func @carried_add
// CHECK-SAME: (%[[INIT:[^:]+]]: tensor<1x1x!ttcore.tile<32x32, bf16>>)
func.func @carried_add(
    %init: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %iter = %c0 to %c1 step %c1 iter_args(%acc = %init) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %contribution = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sum = ttl.add %acc, %contribution : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield %sum : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  %reserve = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %attached = ttl.attach_cb %reserve, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %loop, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}
// CHECK: %[[RESERVE:.*]] = ttl.cb_reserve
// CHECK-NEXT: ttl.store %[[INIT]], %[[RESERVE]]
// CHECK-NEXT: scf.for
// CHECK-NEXT: %[[CONTRIB:.*]] = ttl.cb_wait
// CHECK-NEXT: ttl.store %[[CONTRIB]], %[[RESERVE]] {accumulate}
// CHECK-NEXT: }
// CHECK-NOT: ttl.add
// CHECK-NOT: ttl.attach_cb

// -----

// Additive recurrence whose contribution is defined outside the loop
// body: the wait+attach are emitted once before the loop and the
// accumulator path reuses the outer SSA value on every iteration.
// CHECK-LABEL: func.func @carried_add_outer_contribution
// CHECK-SAME: (%[[INIT:[^:]+]]: tensor<1x1x!ttcore.tile<32x32, bf16>>)
func.func @carried_add_outer_contribution(
    %init: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %contribution_wait = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %contribution = ttl.attach_cb %contribution_wait, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %iter = %c0 to %c1 step %c1 iter_args(%acc = %init) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %sum = ttl.add %acc, %contribution : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield %sum : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  %reserve = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %loop, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}
// CHECK: %[[WAIT:.*]] = ttl.cb_wait
// CHECK-NEXT: %[[CONTRIB:.*]] = ttl.attach_cb %[[WAIT]]
// CHECK: %[[RESERVE:.*]] = ttl.cb_reserve
// CHECK-NEXT: ttl.store %[[INIT]], %[[RESERVE]]
// CHECK-NEXT: scf.for
// CHECK-NEXT: ttl.store %[[CONTRIB]], %[[RESERVE]] {accumulate}
// CHECK-NEXT: }
// CHECK-NOT: ttl.add

// -----

// Commuted additive recurrence lowers to the same accumulate store form.
// CHECK-LABEL: func.func @commuted_carried_add
// CHECK-SAME: (%[[INIT:[^:]+]]: tensor<1x1x!ttcore.tile<32x32, bf16>>)
func.func @commuted_carried_add(
    %init: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %iter = %c0 to %c1 step %c1 iter_args(%acc = %init) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %contribution = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sum = ttl.add %contribution, %acc : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield %sum : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  %reserve = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %loop, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}
// CHECK: %[[RESERVE:.*]] = ttl.cb_reserve
// CHECK-NEXT: ttl.store %[[INIT]], %[[RESERVE]]
// CHECK-NEXT: scf.for
// CHECK-NEXT: %[[CONTRIB:.*]] = ttl.cb_wait
// CHECK-NEXT: ttl.store %[[CONTRIB]], %[[RESERVE]] {accumulate}
// CHECK-NEXT: }
// CHECK-NOT: ttl.add

// -----

// Unary recurrence lowers through compiler-allocated DFB state.
// CHECK-LABEL: func.func @unary_recurrence
// CHECK-SAME: (%[[INIT:[^:]+]]: tensor<1x1x!ttcore.tile<32x32, bf16>>)
func.func @unary_recurrence(
    %init: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %iter = %c0 to %c1 step %c1 iter_args(%state = %init) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %next = ttl.relu %state : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield %next : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  %reserve = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %loop, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}
// CHECK: ttl.bind_cb{cb_index = 1, block_count = 2} {ttl.compiler_allocated}
// CHECK: ttl.store %[[INIT]]
// CHECK: scf.for
// CHECK-NOT: iter_args
// CHECK: %[[WAIT:.*]] = ttl.cb_wait
// CHECK-NEXT: %[[CURRENT:.*]] = ttl.attach_cb %[[WAIT]]
// CHECK-NEXT: %[[NEXT:.*]] = ttl.relu %[[CURRENT]]
// CHECK-NEXT: %[[NEXT_RESERVE:.*]] = ttl.cb_reserve
// CHECK-NEXT: ttl.store %[[NEXT]], %[[NEXT_RESERVE]]
// CHECK-NEXT: }
// CHECK-NEXT: %[[FINAL_WAIT:.*]] = ttl.cb_wait
// CHECK-NEXT: %[[FINAL:.*]] = ttl.attach_cb %[[FINAL_WAIT]]
// CHECK: ttl.store %[[FINAL]]

// -----

// Binary recurrence can read the previous state and another input.
// CHECK-LABEL: func.func @binary_recurrence
func.func @binary_recurrence(
    %init: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %input_wait = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %iter = %c0 to %c1 step %c1 iter_args(%state = %init) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %next = ttl.mul %state, %input : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield %next : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  %reserve = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %loop, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}
// CHECK: ttl.compiler_allocated
// CHECK: %[[INPUT_WAIT:.*]] = ttl.cb_wait
// CHECK-NEXT: %[[INPUT:.*]] = ttl.attach_cb %[[INPUT_WAIT]]
// CHECK: scf.for {{.*}} {
// CHECK-NOT: iter_args
// CHECK-NEXT: %[[WAIT:.*]] = ttl.cb_wait
// CHECK-NEXT: %[[CURRENT:.*]] = ttl.attach_cb %[[WAIT]]
// CHECK-NEXT: %[[NEXT:.*]] = ttl.mul %[[CURRENT]], %[[INPUT]]
// CHECK-NEXT: %[[RES:.*]] = ttl.cb_reserve
// CHECK-NEXT: ttl.store %[[NEXT]], %[[RES]]
// CHECK-NEXT: }

// -----

// Add result used in body (beyond just the yield) falls back to general
// DFB state because the accumulator match requires the add to have a
// single use.
// CHECK-LABEL: func.func @add_with_in_body_use
func.func @add_with_in_body_use(
    %init: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %iter = %c0 to %c1 step %c1 iter_args(%acc = %init) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %contribution = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sum = ttl.add %acc, %contribution : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sink_reserve = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %sum, %sink_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield %sum : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  %reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %loop, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}
// CHECK: ttl.compiler_allocated
// CHECK: ttl.add
// CHECK-NOT: {accumulate}

// -----

// Mixed tensor and scalar iter args: tensor state is materialized and scalar
// state remains loop-carried.
// CHECK-LABEL: func.func @preserve_scalar_iter_arg
// CHECK-SAME: %[[INIT:[^:]+]]: tensor<1x1x!ttcore.tile<32x32, bf16>>
func.func @preserve_scalar_iter_arg(
    %init: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %counter_init: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %one = arith.constant 1 : i32
  %loop:2 = scf.for %iter = %c0 to %c1 step %c1 iter_args(%state = %init, %counter = %counter_init) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, i32) {
    %next_state = ttl.relu %state : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %next_counter = arith.addi %counter, %one : i32
    scf.yield %next_state, %next_counter : tensor<1x1x!ttcore.tile<32x32, bf16>>, i32
  }
  func.return %loop#1 : i32
}
// CHECK: ttl.store %[[INIT]]
// CHECK: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%[[COUNTER:.*]] =
// CHECK: ttl.attach_cb
// CHECK: %[[NEXT_COUNTER:.*]] = arith.addi %[[COUNTER]]
// CHECK: scf.yield %[[NEXT_COUNTER]]
// CHECK: return %[[LOOP]]

// -----

// Multiple tensor iter args can mix accumulation and general DFB state.
// CHECK-LABEL: func.func @mixed_tensor_states
func.func @mixed_tensor_states(
    %acc_init: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %state_init: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop:2 = scf.for %iter = %c0 to %c1 step %c1 iter_args(%acc = %acc_init, %state = %state_init) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %contribution = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sum = ttl.add %acc, %contribution : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %next_state = ttl.relu %state : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield %sum, %next_state : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  %reserve = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %loop#0, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}
// CHECK: ttl.compiler_allocated
// CHECK: ttl.store %{{.*}}, %{{.*}}
// CHECK: scf.for
// CHECK-NOT: iter_args
// CHECK: ttl.store %{{.*}} {accumulate}
// CHECK: ttl.relu
// CHECK: ttl.store

// -----

// Result with multiple users falls back to DFB state rather than accumulation.
// CHECK-LABEL: func.func @add_result_multiple_users
func.func @add_result_multiple_users(
    %init: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %iter = %c0 to %c1 step %c1 iter_args(%acc = %init) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %contribution = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sum = ttl.add %acc, %contribution : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield %sum : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  %reserve0 = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %loop, %reserve0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve1 = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %loop, %reserve1 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}
// CHECK: ttl.compiler_allocated
// CHECK: ttl.add
// CHECK-NOT: {accumulate}

// -----

// Conditional recurrence remains in the loop and stores the scf.if result to
// DFB state.
// CHECK-LABEL: func.func @conditional_recurrence
func.func @conditional_recurrence(
    %init: tensor<1x1x!ttcore.tile<32x32, bf16>>, %cond: i1)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %iter = %c0 to %c1 step %c1 iter_args(%state = %init) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %next = scf.if %cond -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
      %then_value = ttl.relu %state : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      scf.yield %then_value : tensor<1x1x!ttcore.tile<32x32, bf16>>
    } else {
      scf.yield %state : tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    scf.yield %next : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  func.return %loop : tensor<1x1x!ttcore.tile<32x32, bf16>>
}
// CHECK: ttl.compiler_allocated
// CHECK: scf.if
// CHECK: ttl.store %{{.*}}
// CHECK: ttl.cb_wait
// CHECK: ttl.attach_cb

// -----

// Zero-trip semantics are represented by the pre-loop initial store and the
// post-loop final wait.
// CHECK-LABEL: func.func @zero_trip_shape
// CHECK-SAME: (%[[INIT:[^:]+]]: tensor<1x1x!ttcore.tile<32x32, bf16>>)
func.func @zero_trip_shape(
    %init: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %iter = %c0 to %c0 step %c1 iter_args(%state = %init) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %next = ttl.relu %state : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield %next : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  func.return %loop : tensor<1x1x!ttcore.tile<32x32, bf16>>
}
// CHECK: ttl.store %[[INIT]]
// CHECK: scf.for
// CHECK: }
// CHECK-NEXT: %[[FINAL_WAIT:.*]] = ttl.cb_wait
// CHECK-NEXT: %[[FINAL:.*]] = ttl.attach_cb %[[FINAL_WAIT]]
// CHECK-NEXT: return %[[FINAL]]
