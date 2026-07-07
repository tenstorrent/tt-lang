// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-specialize-cores)' | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-specialize-cores,canonicalize,cse)' | FileCheck %s --check-prefix=FOLDED

// Summary: ttl-specialize-cores specializes a kernel function carrying a
// `ttl.operation_grid` attribute for its launch grid. Each emitted clone is
// tagged with `ttl.core_coord` (the list of launch coordinates it serves) and
// has `ttl.core_x` / `ttl.core_y` const-folded to a representative coordinate.
// When the coordinate only drives control flow, coordinates with identical
// control flow are grouped into a single clone; when the coordinate feeds a
// data value, each coordinate gets its own clone.

// -- Test 1: coordinate used as data -> one clone per coordinate. ------------
// The addi feeds the return, a data use, so no de-duplication happens and the
// 2x1 grid produces exactly two specialized clones with no leftover template
// function or coordinate ops.

// CHECK-NOT: ttl.operation_grid
// CHECK-NOT: ttl.core_x
// CHECK-NOT: ttl.core_y

// CHECK-LABEL: func.func @k_c0_0
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 0]]
// CHECK-DAG:     arith.constant 0 : index

// CHECK-LABEL: func.func @k_c1_0
// CHECK-SAME:    ttl.core_coord = {{\[\[}}1, 0]]
// CHECK-DAG:     arith.constant 1 : index
// CHECK-DAG:     arith.constant 0 : index

// FOLDED-LABEL: func.func @k_c0_0
// FOLDED:         %[[C0:.*]] = arith.constant 0 : index
// FOLDED:         return %[[C0]] : index

// FOLDED-LABEL: func.func @k_c1_0
// FOLDED:         %[[C1:.*]] = arith.constant 1 : index
// FOLDED:         return %[[C1]] : index

module {
  func.func @k() -> index attributes {ttl.operation_grid = [2 : i64, 1 : i64]} {
    %x = ttl.core_x : index
    %y = ttl.core_y : index
    %s = arith.addi %x, %y : index
    return %s : index
  }
}

// -----

// -- Test 2: coordinate only drives a predicate -> control-flow de-dup. ------
// core_y feeds only an scf.if condition, so the 1x4 grid collapses to two
// clones: one shared by y in {1, 2, 3} (predicate false) and one for y == 0
// (predicate true). The clone for the larger group is emitted first.

// CHECK-LABEL: func.func @k2_c0_1
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 1], [0, 2], [0, 3]]

// CHECK-LABEL: func.func @k2_c0_0
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 0]]

// No per-coordinate clone for the merged interior nodes must survive.
// CHECK-NOT: func.func @k2_c0_2
// CHECK-NOT: func.func @k2_c0_3

// FOLDED-LABEL: func.func @k2_c0_1
// FOLDED:         %[[C9:.*]] = arith.constant 9 : index
// FOLDED:         return %[[C9]] : index

// FOLDED-LABEL: func.func @k2_c0_0
// FOLDED:         %[[C7:.*]] = arith.constant 7 : index
// FOLDED:         return %[[C7]] : index

module {
  func.func @k2() -> index attributes {ttl.operation_grid = [1 : i64, 4 : i64]} {
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %c9 = arith.constant 9 : index
    %y = ttl.core_y : index
    %pred = arith.cmpi eq, %y, %c0 : index
    %r = scf.if %pred -> (index) {
      scf.yield %c7 : index
    } else {
      scf.yield %c9 : index
    }
    return %r : index
  }
}

// -----

// -- Test 3: folding eliminates the untaken control-flow path. ---------------
// Each branch holds a uniquely identifiable op that cannot be constant-folded
// (it depends on the runtime %arg): `arith.muli` in the "then" path and
// `arith.addi` in the "else" path. This isolates dead-path elimination from
// constant folding of the yielded values. After ttl-specialize-cores const-
// folds the coordinate, canonicalize proves the scf.if condition constant and
// deletes the untaken region (and the now-unused op), and cse cleans up.
//
// Grid is 1x3, so the corner (y == 0) is one clone and the interior nodes
// (y in {1, 2}) de-duplicate into a second clone; the interior clone is
// emitted first. Only the taken branch's op survives in each clone, and no
// scf.if remains.

// FOLDED-LABEL: func.func @sel_c0_1
// FOLDED:         arith.addi
// FOLDED-NOT:     arith.muli
// FOLDED-NOT:     scf.if

// FOLDED-LABEL: func.func @sel_c0_0
// FOLDED:         arith.muli
// FOLDED-NOT:     arith.addi
// FOLDED-NOT:     scf.if

module {
  func.func @sel(%arg: index) -> index attributes {ttl.operation_grid = [1 : i64, 3 : i64]} {
    %c0 = arith.constant 0 : index
    %y = ttl.core_y : index
    %pred = arith.cmpi eq, %y, %c0 : index
    %r = scf.if %pred -> (index) {
      %hot = arith.muli %arg, %arg : index
      scf.yield %hot : index
    } else {
      %cold = arith.addi %arg, %arg : index
      scf.yield %cold : index
    }
    return %r : index
  }
}
