// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttkernel-specialize-cores)' --verify-diagnostics
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttkernel-specialize-cores)' 2>/dev/null | FileCheck %s

// Skipping one function does not block the others. `@used` branches on a
// coordinate but has a caller, so it is left un-specialized (warning, not
// fatal). `@free` branches too but has no caller, so it is still cloned per
// launch coordinate. The 1x2 grid yields @free_c0_0 and @free_c0_1; @used and
// its caller are untouched.

// CHECK-DAG: func.func @used()
// CHECK-DAG: func.func @caller()
// CHECK-DAG: func.func @free_c0_0
// CHECK-DAG: func.func @free_c0_1
// CHECK-NOT: func.func @used_c

module attributes {ttl.launch_grid = [1 : i64, 2 : i64]} {
  // expected-warning @+1 {{not specializing}}
  func.func @used() -> index {
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
  func.func @caller() -> index {
    %v = func.call @used() : () -> index
    return %v : index
  }
  func.func @free() -> index {
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
