// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-specialize-plan))' | FileCheck %s --check-prefix=PLAN
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-specialize-plan),ttl-specialize-cores)' | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-specialize-plan),ttl-specialize-cores,canonicalize,cse)' | FileCheck %s --check-prefix=FOLDED

// Summary: per-core specialization runs in two phases. Phase A
// (ttl-specialize-plan, a parallel func pass) annotates each kernel carrying a
// `ttl.operation_grid` with a `ttl.specialize_plan` describing how to clone it,
// marking every `scf.if` that branches on `ttl.core_x` / `ttl.core_y` with a
// `ttl.specialize_branch` id. Phase B (ttl-specialize-cores, a module pass)
// materializes one clone per coordinate group, forcing each marked branch to
// the group's outcome and tagging the clone with `ttl.core_coord`. Downstream
// canonicalize/cse fold the now-constant conditions and delete dead branches.
// Only branch conditions are specialized; coordinate reads are left intact.

// -- Test 1: coordinate used only as data -> no branch, no specialization. ----
// The coordinates feed an addi that reaches the return (a data use) and drive
// no control flow, so Phase A records no plan and the function is left as a
// single whole-grid kernel with its coordinate reads intact.

// PLAN-LABEL: func.func @kdata
// PLAN-NOT:   ttl.specialize_plan

// CHECK-LABEL: func.func @kdata
// CHECK-NOT:     ttl.core_coord
// CHECK:         ttl.core_x
// CHECK-NOT:   func.func @kdata_c

module {
  func.func @kdata() -> index attributes {ttl.operation_grid = [2 : i64, 1 : i64]} {
    %x = ttl.core_x : index
    %y = ttl.core_y : index
    %s = arith.addi %x, %y : index
    return %s : index
  }
}

// -----

// -- Test 2: coordinate drives a predicate -> control-flow de-dup. -----------
// core_y feeds only an scf.if condition, so Phase A partitions the 1x4 grid
// into two groups: y in {1, 2, 3} (predicate false) and y == 0 (predicate
// true). The false group is emitted first (sorted signature). Phase B forces
// each clone's branch to its group outcome; canonicalize/cse then collapse it.

// PLAN-LABEL: func.func @k2
// PLAN-SAME:    ttl.specialize_plan = [{coords = array<i64: 0, 1, 0, 2, 0, 3>, taken = array<i1: false>}, {coords = array<i64: 0, 0>, taken = array<i1: true>}]
// PLAN:         ttl.specialize_branch = 0 : i64

// CHECK-LABEL: func.func @k2_c0_1
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 1], [0, 2], [0, 3]]
// CHECK:         %[[F:.*]] = arith.constant false
// CHECK:         scf.if %[[F]]

// CHECK-LABEL: func.func @k2_c0_0
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 0]]
// CHECK:         %[[T:.*]] = arith.constant true
// CHECK:         scf.if %[[T]]

// The planning markers must not survive Phase B.
// CHECK-NOT:   ttl.specialize_plan
// CHECK-NOT:   ttl.specialize_branch

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
// constant folding of the yielded values. After Phase B forces each clone's
// branch condition constant, canonicalize deletes the untaken region (and its
// now-unused op) and cse cleans up.
//
// Grid is 1x3, so the corner (y == 0) is one clone and the interior nodes
// (y in {1, 2}) de-duplicate into a second clone; the interior clone is
// emitted first. Only the taken branch's op survives, and no scf.if remains.

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

// -----

// -- Test 4: pipe participants are never specialized. ------------------------
// The module uses a pipe, so even though core_y drives an scf.if, Phase A
// conservatively records no plan (cloning a pipe endpoint deadlocks at
// runtime). The kernel is left as a single whole-grid binary.

// PLAN-LABEL: func.func @kpipe
// PLAN-NOT:   ttl.specialize_plan
// PLAN-NOT:   ttl.specialize_branch

// CHECK-LABEL: func.func @kpipe
// CHECK-NOT:   ttl.core_coord
// CHECK-NOT:   func.func @kpipe_c

module {
  func.func @kpipe() -> index attributes {ttl.operation_grid = [1 : i64, 4 : i64]} {
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %c9 = arith.constant 9 : index
    %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
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
