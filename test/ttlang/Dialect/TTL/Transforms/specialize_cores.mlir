// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-specialize-cores)' | FileCheck %s
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-specialize-cores,canonicalize,cse)' | FileCheck %s --check-prefix=FOLDED

// Summary: ttl-specialize-cores clones a kernel function carrying a
// `ttl.operation_grid` attribute once per launch coordinate, tags each clone
// with `ttl.core_coord`, and const-folds `ttl.core_x` / `ttl.core_y` to the
// concrete coordinate. The second RUN shows that once the coordinates are
// constants, upstream canonicalize/cse fold the coordinate-dependent
// arithmetic per clone.

// The 2x1 grid produces exactly two specialized clones and no leftover
// template function or coordinate ops.

// CHECK-NOT: ttl.operation_grid
// CHECK-NOT: ttl.core_x
// CHECK-NOT: ttl.core_y

// CHECK-LABEL: func.func @k_c0_0
// CHECK-SAME:    ttl.core_coord = [0, 0]
// CHECK-DAG:     arith.constant 0 : index

// CHECK-LABEL: func.func @k_c1_0
// CHECK-SAME:    ttl.core_coord = [1, 0]
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
