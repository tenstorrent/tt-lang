// RUN: ttlang-opt %s -pass-pipeline='builtin.module(canonicalize,ttkernel-unroll-static-pipenet-record-loops,canonicalize)' | FileCheck %s

func.func private @consume(index)

// CHECK-LABEL: func.func @unroll_static
// CHECK-NOT:     scf.for
// CHECK-NOT:     ttkernel.experimental.constant_table_lookup
// CHECK-DAG:     %[[THREE:.*]] = arith.constant 3 : index
// CHECK-DAG:     %[[FIVE:.*]] = arith.constant 5 : index
// CHECK-DAG:     %[[EIGHT:.*]] = arith.constant 8 : index
// CHECK:         call @consume(%[[THREE]]) : (index) -> ()
// CHECK:         call @consume(%[[FIVE]]) : (index) -> ()
// CHECK:         call @consume(%[[EIGHT]]) : (index) -> ()
// CHECK-NOT:     scf.for
// CHECK-NOT:     ttkernel.experimental.constant_table_lookup
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

// CHECK-LABEL: func.func @inline_single_iteration
// CHECK-NOT:     scf.for
// CHECK-NOT:     ttkernel.experimental.constant_table_lookup
// CHECK:         %[[THIRTEEN:.*]] = arith.constant 13 : index
// CHECK:         call @consume(%[[THIRTEEN]]) : (index) -> ()
// CHECK-NOT:     scf.for
// CHECK-NOT:     ttkernel.experimental.constant_table_lookup
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

// CHECK-LABEL: func.func @retain_dynamic
// CHECK:         scf.for
// CHECK:         ttl.pipenet_local_record_loop
func.func @retain_dynamic(%upper : index) {
  %lower = arith.constant 0 : index
  %step = arith.constant 1 : index
  scf.for %record = %lower to %upper step %step {
    func.call @consume(%record) : (index) -> ()
  } {ttl.pipenet_local_record_loop}
  return
}

// CHECK-LABEL: func.func @retain_unmarked
// CHECK:         scf.for
func.func @retain_unmarked() {
  %lower = arith.constant 0 : index
  %upper = arith.constant 3 : index
  %step = arith.constant 1 : index
  scf.for %record = %lower to %upper step %step {
    func.call @consume(%record) : (index) -> ()
  }
  return
}
