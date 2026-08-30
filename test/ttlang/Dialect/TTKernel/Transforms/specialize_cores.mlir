// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttkernel-specialize-cores)' | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttkernel-specialize-cores,canonicalize,cse)' | FileCheck %s --check-prefix=FOLDED

// Summary: per-core specialization is a single module pass run at the TTKernel
// level. It clones kernels whose `scf.if` conditions or `scf.for` bounds depend
// on core coordinates, replaces coordinate reads with constants, and tags each
// clone with `ttl.core_coord`. Downstream canonicalization resolves the
// coordinate-dependent control flow. Coordinate-only data uses remain in one
// whole-grid kernel.

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

// -- Test 5: coordinate-dependent loop bounds require specialization. -------
// A table-selected upper bound becomes constant in every per-core clone.

// CHECK-NOT:   func.func @kloop()
// CHECK-LABEL: func.func @kloop_c0_0
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 0]]
// CHECK-NOT:     my_logical_y_
// CHECK-LABEL: func.func @kloop_c0_1
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 1]]
// CHECK-NOT:     my_logical_y_
// CHECK-NOT:   func.func @klower()
// CHECK-LABEL: func.func @klower_c0_0
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 0]]
// CHECK-LABEL: func.func @klower_c0_1
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 1]]
// CHECK-NOT:   func.func @kstep()
// CHECK-LABEL: func.func @kstep_c0_0
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 0]]
// CHECK-LABEL: func.func @kstep_c0_1
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 1]]
// CHECK-NOT:   func.func @kregion()
// CHECK-LABEL: func.func @kregion_c0_0
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 0]]
// CHECK-NOT:     my_logical_y_
// CHECK-LABEL: func.func @kregion_c0_1
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 1]]
// CHECK-NOT:     my_logical_y_

// FOLDED-LABEL: func.func @kloop_c0_0
// FOLDED-NEXT:    %[[LOOP_ZERO_0:.*]] = arith.constant 0 : index
// FOLDED-NEXT:    call @consume(%[[LOOP_ZERO_0]]) : (index) -> ()
// FOLDED-NEXT:    return
// FOLDED-LABEL: func.func @kloop_c0_1
// FOLDED-NOT:     ttkernel.experimental.constant_table_lookup
// FOLDED-DAG:     %[[LOOP_ZERO_1:.*]] = arith.constant 0 : index
// FOLDED-DAG:     %[[LOOP_ONE_1:.*]] = arith.constant 1 : index
// FOLDED-DAG:     %[[LOOP_TWO_1:.*]] = arith.constant 2 : index
// FOLDED:         scf.for %{{.*}} = %[[LOOP_ZERO_1]] to %[[LOOP_TWO_1]] step %[[LOOP_ONE_1]]
// FOLDED-LABEL: func.func @klower_c0_0
// FOLDED-DAG:     %[[LOWER_ZERO_0:.*]] = arith.constant 0 : index
// FOLDED-DAG:     %[[LOWER_ONE_0:.*]] = arith.constant 1 : index
// FOLDED-DAG:     %[[LOWER_THREE_0:.*]] = arith.constant 3 : index
// FOLDED:         scf.for %{{.*}} = %[[LOWER_ZERO_0]] to %[[LOWER_THREE_0]] step %[[LOWER_ONE_0]]
// FOLDED-LABEL: func.func @klower_c0_1
// FOLDED-DAG:     %[[LOWER_ONE_1:.*]] = arith.constant 1 : index
// FOLDED-DAG:     %[[LOWER_THREE_1:.*]] = arith.constant 3 : index
// FOLDED:         scf.for %{{.*}} = %[[LOWER_ONE_1]] to %[[LOWER_THREE_1]] step %[[LOWER_ONE_1]]
// FOLDED-LABEL: func.func @kstep_c0_0
// FOLDED-DAG:     %[[STEP_ZERO_0:.*]] = arith.constant 0 : index
// FOLDED-DAG:     %[[STEP_ONE_0:.*]] = arith.constant 1 : index
// FOLDED-DAG:     %[[STEP_FOUR_0:.*]] = arith.constant 4 : index
// FOLDED:         scf.for %{{.*}} = %[[STEP_ZERO_0]] to %[[STEP_FOUR_0]] step %[[STEP_ONE_0]]
// FOLDED-LABEL: func.func @kstep_c0_1
// FOLDED-DAG:     %[[STEP_ZERO_1:.*]] = arith.constant 0 : index
// FOLDED-DAG:     %[[STEP_TWO_1:.*]] = arith.constant 2 : index
// FOLDED-DAG:     %[[STEP_FOUR_1:.*]] = arith.constant 4 : index
// FOLDED:         scf.for %{{.*}} = %[[STEP_ZERO_1]] to %[[STEP_FOUR_1]] step %[[STEP_TWO_1]]

module attributes {ttl.launch_grid = [1 : i64, 2 : i64]} {
  func.func private @consume(index)

  func.func @kloop() {
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    %core_y = "ttkernel.my_logical_y_"() : () -> index
    %upper = ttkernel.experimental.constant_table_lookup %core_y, [1, 2] : index
    scf.for %record = %lower to %upper step %step {
      func.call @consume(%record) : (index) -> ()
    }
    return
  }

  func.func @klower() {
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    %core_y = "ttkernel.my_logical_y_"() : () -> index
    %lower = ttkernel.experimental.constant_table_lookup %core_y, [0, 1] : index
    scf.for %record = %lower to %upper step %step {
      func.call @consume(%record) : (index) -> ()
    }
    return
  }

  func.func @kstep() {
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %core_y = "ttkernel.my_logical_y_"() : () -> index
    %step = ttkernel.experimental.constant_table_lookup %core_y, [1, 2] : index
    scf.for %record = %lower to %upper step %step {
      func.call @consume(%record) : (index) -> ()
    }
    return
  }

  func.func @kregion(%select_coord : i1) {
    %lower = arith.constant 0 : index
    %one = arith.constant 1 : index
    %core_y = "ttkernel.my_logical_y_"() : () -> index
    %upper = scf.if %select_coord -> (index) {
      scf.yield %core_y : index
    } else {
      scf.yield %one : index
    }
    scf.for %record = %lower to %upper step %one {
      func.call @consume(%record) : (index) -> ()
    }
    return
  }
}

// -----

// -- Test 6: unrelated region coordinate use does not specialize. -----------
// A coordinate used inside an scf.if region but not yielded into the loop
// bound is outside the bound's backward slice.

// CHECK-LABEL: func.func @kindependent_region
// CHECK-NOT:     ttl.core_coord
// CHECK:         my_logical_y_
// CHECK-NOT:   func.func @kindependent_region_c

module attributes {ttl.launch_grid = [1 : i64, 2 : i64]} {
  func.func private @consume(index)

  func.func @kindependent_region(%select_value : i1) {
    %lower = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %upper = scf.if %select_value -> (index) {
      %core_y = "ttkernel.my_logical_y_"() : () -> index
      func.call @consume(%core_y) : (index) -> ()
      scf.yield %one : index
    } else {
      scf.yield %two : index
    }
    scf.for %record = %lower to %upper step %one {
      func.call @consume(%record) : (index) -> ()
    }
    return
  }
}

// -----

// -- Test 7: single-core launch grid is a legitimate no-op. ------------------
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

// -- Test 8: functions with symbol uses are skipped, others still clone. -----
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
