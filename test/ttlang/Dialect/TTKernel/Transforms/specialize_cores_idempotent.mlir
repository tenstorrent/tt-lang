// Summary: ttkernel-specialize-cores is idempotent. The first run clones a
// branching kernel per launch coordinate and replaces coordinate reads with
// constants, so clones no longer branch on my_logical_*. A second run must be
// a no-op: same clone set, no double-suffixed symbols like @k_c0_0_c0_0, and
// re-running on the once-specialized IR must be a bit-identical no-op.

// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttkernel-specialize-cores)' | FileCheck %s
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttkernel-specialize-cores,ttkernel-specialize-cores)' | FileCheck %s
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttkernel-specialize-cores)' -o %t.once
// RUN: ttlang-opt %t.once -pass-pipeline='builtin.module(ttkernel-specialize-cores)' | diff -u %t.once -

// CHECK-NOT:   func.func @k()
// CHECK-LABEL: func.func @k_c0_0
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 0]]
// CHECK-LABEL: func.func @k_c1_0
// CHECK-SAME:    ttl.core_coord = {{\[\[}}1, 0]]
// CHECK-LABEL: func.func @k_c0_1
// CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 1]]
// CHECK-LABEL: func.func @k_c1_1
// CHECK-SAME:    ttl.core_coord = {{\[\[}}1, 1]]
// CHECK-NOT:   func.func @k_c{{.*}}_c

module attributes {ttl.launch_grid = [2 : i64, 2 : i64]} {
  func.func @k() -> index {
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %c9 = arith.constant 9 : index
    %x = "ttkernel.my_logical_x_"() : () -> index
    %pred = arith.cmpi eq, %x, %c0 : index
    %r = scf.if %pred -> (index) {
      scf.yield %c7 : index
    } else {
      scf.yield %c9 : index
    }
    return %r : index
  }
}
