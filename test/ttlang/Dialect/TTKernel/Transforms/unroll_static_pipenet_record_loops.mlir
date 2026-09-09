// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttkernel-unroll-static-pipenet-record-loops))' | FileCheck %s --check-prefix=PASS --implicit-check-not=ttl.pipenet_local_record_loop
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttkernel-unroll-static-pipenet-record-loops),canonicalize)' | FileCheck %s

// Summary: Verifies static local PipeNet record loops are fully unrolled and
// their temporary compiler marker does not remain on dynamic loops.

func.func private @consume(index)

// PASS-LABEL: func.func @unroll_static
// PASS-NOT:     scf.for
// PASS:         return
// CHECK-LABEL: func.func @unroll_static
// CHECK-NOT:     scf.for
// CHECK-NOT:     ttkernel.experimental.constant_table_lookup
// CHECK-DAG:     %[[THREE:.*]] = arith.constant 3 : index
// CHECK-DAG:     %[[FIVE:.*]] = arith.constant 5 : index
// CHECK-DAG:     %[[EIGHT:.*]] = arith.constant 8 : index
// CHECK:         call @consume(%[[THREE]]) : (index) -> ()
// CHECK:         call @consume(%[[FIVE]]) : (index) -> ()
// CHECK:         call @consume(%[[EIGHT]]) : (index) -> ()
// CHECK:         return
func.func @unroll_static() {
  %lower = arith.constant 0 : index
  %upper = arith.constant 3 : index
  %step = arith.constant 1 : index
  scf.for %record = %lower to %upper step %step {
    %value = ttkernel.experimental.constant_table_lookup %record, [3, 5, 8] : index
    func.call @consume(%value) : (index) -> ()
  } {ttl.pipenet_local_record_loop}
  return
}

// PASS-LABEL: func.func @inline_single_iteration
// PASS-NOT:     scf.for
// PASS:         return
// CHECK-LABEL: func.func @inline_single_iteration
// CHECK-NOT:     scf.for
// CHECK:         %[[THIRTEEN:.*]] = arith.constant 13 : index
// CHECK:         call @consume(%[[THIRTEEN]]) : (index) -> ()
// CHECK:         return
func.func @inline_single_iteration() {
  %lower = arith.constant 1 : index
  %upper = arith.constant 2 : index
  %step = arith.constant 1 : index
  scf.for %record = %lower to %upper step %step {
    %value = ttkernel.experimental.constant_table_lookup %record, [11, 13] : index
    func.call @consume(%value) : (index) -> ()
  } {ttl.pipenet_local_record_loop}
  return
}

// PASS-LABEL: func.func @erase_zero_iterations
// PASS-NOT:     scf.for
// PASS-NOT:     call @consume
// PASS:         return
// CHECK-LABEL: func.func @erase_zero_iterations
// CHECK-NOT:     scf.for
// CHECK-NOT:     call @consume
// CHECK:         return
func.func @erase_zero_iterations() {
  %bound = arith.constant 2 : index
  %step = arith.constant 1 : index
  scf.for %record = %bound to %bound step %step {
    func.call @consume(%record) : (index) -> ()
  } {ttl.pipenet_local_record_loop}
  return
}

// PASS-LABEL: func.func @retain_dynamic
// PASS:         scf.for
// PASS:         call @consume
// PASS:         return
// CHECK-LABEL: func.func @retain_dynamic
// CHECK:         scf.for
// CHECK-NOT:     ttl.pipenet_local_record_loop
// CHECK:         call @consume
// CHECK:         return
func.func @retain_dynamic(%upper : index) {
  %lower = arith.constant 0 : index
  %step = arith.constant 1 : index
  scf.for %record = %lower to %upper step %step {
    func.call @consume(%record) : (index) -> ()
  } {ttl.pipenet_local_record_loop}
  return
}

// PASS-LABEL: func.func @retain_unmarked
// PASS:         scf.for
// PASS:         call @consume
// PASS:         return
// CHECK-LABEL: func.func @retain_unmarked
// CHECK:         scf.for
// CHECK-NOT:     ttl.pipenet_local_record_loop
// CHECK:         call @consume
// CHECK:         return
func.func @retain_unmarked() {
  %lower = arith.constant 0 : index
  %upper = arith.constant 3 : index
  %step = arith.constant 1 : index
  scf.for %record = %lower to %upper step %step {
    func.call @consume(%record) : (index) -> ()
  }
  return
}

// Nested callbacks can produce nested record loops. Post-order unrolling must
// fully expand both loops without invalidating the outer loop handle.
// PASS-LABEL: func.func @unroll_nested
// PASS-NOT:     scf.for
// PASS:         return
// CHECK-LABEL: func.func @unroll_nested
// CHECK-NOT:     scf.for
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[TWO:.*]] = arith.constant 2 : index
// CHECK:         call @consume(%[[ZERO]]) : (index) -> ()
// CHECK:         call @consume(%[[ONE]]) : (index) -> ()
// CHECK:         call @consume(%[[ONE]]) : (index) -> ()
// CHECK:         call @consume(%[[TWO]]) : (index) -> ()
// CHECK:         return
func.func @unroll_nested() {
  %lower = arith.constant 0 : index
  %upper = arith.constant 2 : index
  %step = arith.constant 1 : index
  scf.for %outer = %lower to %upper step %step {
    scf.for %inner = %lower to %upper step %step {
      %record = arith.addi %outer, %inner : index
      func.call @consume(%record) : (index) -> ()
    } {ttl.pipenet_local_record_loop}
  } {ttl.pipenet_local_record_loop}
  return
}
