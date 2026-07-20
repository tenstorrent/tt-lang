// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttkernel-specialize-cores)' | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttkernel-specialize-cores,canonicalize,cse)' | FileCheck %s --check-prefix=FOLDED

// Summary: per-core specialization is a single module pass run at the TTKernel
// level. For every kernel whose control flow branches on a core coordinate (an
// `scf.if` whose condition is derived from `ttkernel.my_logical_x_` /
// `my_logical_y_`), it clones the function once per launch coordinate, replaces
// the coordinate reads with constants for that core, and tags the clone with
// `ttl.core_coord`. Downstream canonicalize/cse fold the now-constant
// conditions and delete the untaken regions. Kernels that only use coordinates
// as data (no branch) are left as a single whole-grid binary. The pass does not
// special-case pipe participants: any kernel that branches on a coordinate is
// cloned, whether or not the module uses pipes.

// -- Test 1: coordinate used only as data -> no branch, no specialization. ----
// The coordinates feed an addi that reaches the return (a data use) and drive
// no control flow, so the pass leaves the function as a single whole-grid
// kernel with its coordinate reads intact.

// CHECK-LABEL: func.func @kdata
// CHECK-NOT:     ttl.core_coord
// CHECK:         my_logical_x_
// CHECK-NOT:   func.func @kdata_c

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @kdata() -> index {
    %x = "ttkernel.my_logical_x_"() : () -> index
    %y = "ttkernel.my_logical_y_"() : () -> index
    %s = arith.addi %x, %y : index
    return %s : index
  }
}

// -----

// -- Test 2: coordinate drives a predicate -> per-core clones. ----------------
// core_y feeds an scf.if condition, so the 1x4 grid yields four clones (one per
// row). In each clone the coordinate read is replaced by a constant and the
// clone is tagged with the single coordinate it serves. canonicalize/cse then
// fold the condition: c0_0 takes the "then" (7); the rest take the "else" (9).

// CHECK-NOT:   func.func @k2()
// CHECK-LABEL: func.func @k2_c0_0
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 0]]
// CHECK-NOT:     my_logical_y_
// CHECK-LABEL: func.func @k2_c0_1
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 1]]
// CHECK-LABEL: func.func @k2_c0_2
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 2]]
// CHECK-LABEL: func.func @k2_c0_3
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 3]]

// FOLDED-LABEL: func.func @k2_c0_0
// FOLDED:         %[[C7:.*]] = arith.constant 7 : index
// FOLDED:         return %[[C7]] : index
// FOLDED-LABEL: func.func @k2_c0_1
// FOLDED:         %[[C9:.*]] = arith.constant 9 : index
// FOLDED:         return %[[C9]] : index
// FOLDED-LABEL: func.func @k2_c0_2
// FOLDED:         %[[C9A:.*]] = arith.constant 9 : index
// FOLDED:         return %[[C9A]] : index
// FOLDED-LABEL: func.func @k2_c0_3
// FOLDED:         %[[C9B:.*]] = arith.constant 9 : index
// FOLDED:         return %[[C9B]] : index

module attributes {ttl.launch_grid = [1 : i64, 4 : i64]} {
  func.func @k2() -> index {
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %c9 = arith.constant 9 : index
    %y = "ttkernel.my_logical_y_"() : () -> index
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
// constant folding of the yielded values. After the pass forces each clone's
// coordinate constant, canonicalize deletes the untaken region (and its
// now-unused op) and cse cleans up. Grid is 1x3, so y == 0 takes the "then"
// (muli) and y in {1, 2} take the "else" (addi); no scf.if survives.

// FOLDED-LABEL: func.func @sel_c0_0
// FOLDED:         arith.muli
// FOLDED-NOT:     arith.addi
// FOLDED-NOT:     scf.if
// FOLDED-LABEL: func.func @sel_c0_1
// FOLDED:         arith.addi
// FOLDED-NOT:     arith.muli
// FOLDED-NOT:     scf.if
// FOLDED-LABEL: func.func @sel_c0_2
// FOLDED:         arith.addi
// FOLDED-NOT:     arith.muli
// FOLDED-NOT:     scf.if

module attributes {ttl.launch_grid = [1 : i64, 3 : i64]} {
  func.func @sel(%arg: index) -> index {
    %c0 = arith.constant 0 : index
    %y = "ttkernel.my_logical_y_"() : () -> index
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

// -----

// -- Test 4: pipe participants are specialized like any other kernel. --------
// The module used a pipe, so a semaphore op is present. The pass no longer
// special-cases pipes: because core_y drives an scf.if, the 1x2 grid is cloned
// per row. The semaphore op is carried into each clone unchanged.

// CHECK-NOT:   func.func @kpipe()
// CHECK-LABEL: func.func @kpipe_c0_0
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 0]]
// CHECK:         ttkernel.get_semaphore
// CHECK-LABEL: func.func @kpipe_c0_1
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 1]]
// CHECK:         ttkernel.get_semaphore

module attributes {ttl.launch_grid = [1 : i64, 2 : i64]} {
  func.func @kpipe() -> index {
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %c9 = arith.constant 9 : index
    %sem = ttkernel.get_semaphore(%c0) : (index) -> !ttkernel.local_semaphore
    %y = "ttkernel.my_logical_y_"() : () -> index
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

// -- Test 5: single-core launch grid is a legitimate no-op. ------------------
// A valid `ttl.launch_grid` whose product is <= 1 has nothing to specialize
// CHECK-LABEL: func.func @ksingle
// CHECK-NOT:     ttl.core_coord
// CHECK:         my_logical_y_
// CHECK-NOT:   func.func @ksingle_c

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @ksingle() -> index {
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %c9 = arith.constant 9 : index
    %y = "ttkernel.my_logical_y_"() : () -> index
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

// -- Test 6: functions with symbol uses are skipped, others still clone. -----
// `kcallee` branches on a coordinate but is referenced by `kcaller`, so it is
// left unspecialized (erasing it would leave a dangling call). `kleaf` has no
// symbol uses and is still cloned per core.

// CHECK-LABEL: func.func @kcallee
// CHECK-NOT:     ttl.core_coord
// CHECK:         my_logical_y_
// CHECK-NOT:   func.func @kcallee_c
// CHECK-LABEL: func.func @kcaller
// CHECK:         call @kcallee
// CHECK-NOT:   func.func @kleaf()
// CHECK-LABEL: func.func @kleaf_c0_0
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 0]]
// CHECK-LABEL: func.func @kleaf_c0_1
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 1]]

module attributes {ttl.launch_grid = [1 : i64, 2 : i64]} {
  func.func @kcallee() -> index {
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %c9 = arith.constant 9 : index
    %y = "ttkernel.my_logical_y_"() : () -> index
    %pred = arith.cmpi eq, %y, %c0 : index
    %r = scf.if %pred -> (index) {
      scf.yield %c7 : index
    } else {
      scf.yield %c9 : index
    }
    return %r : index
  }
  func.func @kcaller() -> index {
    %r = func.call @kcallee() : () -> index
    return %r : index
  }
  func.func @kleaf() -> index {
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %c9 = arith.constant 9 : index
    %y = "ttkernel.my_logical_y_"() : () -> index
    %pred = arith.cmpi eq, %y, %c0 : index
    %r = scf.if %pred -> (index) {
      scf.yield %c7 : index
    } else {
      scf.yield %c9 : index
    }
    return %r : index
  }
}
