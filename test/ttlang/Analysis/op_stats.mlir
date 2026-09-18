// RUN: ttlang-op-stats %s | FileCheck %s

// This file tests deterministic per-function operation statistics. Exact
// dynamic counts compose across loops and induction-dependent conditions,
// while an unproven occurrence makes its operation-name aggregate unknown.

func.func @exact_counts() {
  %zero = arith.constant 0 : index
  %upper = arith.constant 4 : index
  %two = arith.constant 2 : index
  %step = arith.constant 1 : index
  scf.for %iteration = %zero to %upper step %step {
    %sum = arith.addi %iteration, %iteration : index
    %selected = arith.cmpi slt, %iteration, %two : index
    scf.if %selected {
      %selected_sum = arith.addi %sum, %iteration : index
    }
  }
  return
}

// CHECK-LABEL: func @exact_counts
// CHECK-NEXT:    arith.addi static_occurrences=2 dynamic_instances=6
// CHECK-NEXT:    arith.cmpi static_occurrences=1 dynamic_instances=4
// CHECK-NEXT:    arith.constant static_occurrences=4 dynamic_instances=4
// CHECK-NEXT:    func.return static_occurrences=1 dynamic_instances=1
// CHECK-NEXT:    scf.for static_occurrences=1 dynamic_instances=1
// CHECK-NEXT:    scf.if static_occurrences=1 dynamic_instances=4
// CHECK-NEXT:    scf.yield static_occurrences=2 dynamic_instances=6

// An unknown loop bound leaves operations inside the loop unproven.
func.func @unknown_count(%upper: index) {
  %zero = arith.constant 0 : index
  %step = arith.constant 1 : index
  scf.for %iteration = %zero to %upper step %step {
    %product = arith.muli %iteration, %iteration : index
  }
  return
}

// CHECK-LABEL: func @unknown_count
// CHECK-NEXT:    arith.constant static_occurrences=2 dynamic_instances=2
// CHECK-NEXT:    arith.muli static_occurrences=1 dynamic_instances=unknown
// CHECK-NEXT:    func.return static_occurrences=1 dynamic_instances=1
// CHECK-NEXT:    scf.for static_occurrences=1 dynamic_instances=1
// CHECK-NEXT:    scf.yield static_occurrences=1 dynamic_instances=unknown
