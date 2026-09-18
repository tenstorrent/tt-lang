// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttkernel-specialize-cores)'

// Negative cases for ttkernel-specialize-cores. The pass is opt-in
// (--ttl-specialize-cores), so a launch grid it cannot use, or a function it
// cannot safely clone, must be a diagnostic rather than a silent no-op.

// -- Missing launch grid. -----------------------------------------------------
// A kernel branches on the coordinate but the module carries no
// `ttl.launch_grid`, so the pass has no per-core extent to specialize over.

// expected-error @+1 {{requires a `ttl.launch_grid`}}
module {
  func.func @no_grid() -> index {
    %c0 = arith.constant 0 : index
    %y = "ttkernel.my_logical_y_"() : () -> index
    %pred = arith.cmpi eq, %y, %c0 : index
    %r = scf.if %pred -> (index) {
      scf.yield %c0 : index
    } else {
      scf.yield %y : index
    }
    return %r : index
  }
}

// -----

// -- Malformed launch grid: wrong length. -------------------------------------

// expected-error @+1 {{length-2 array of positive}}
module attributes {ttl.launch_grid = [2 : i64]} {
  func.func @bad_len() -> index {
    %c0 = arith.constant 0 : index
    %y = "ttkernel.my_logical_y_"() : () -> index
    %pred = arith.cmpi eq, %y, %c0 : index
    %r = scf.if %pred -> (index) {
      scf.yield %c0 : index
    } else {
      scf.yield %y : index
    }
    return %r : index
  }
}

// -----

// -- Malformed launch grid: non-positive extent. ------------------------------

// expected-error @+1 {{length-2 array of positive}}
module attributes {ttl.launch_grid = [2 : i64, 0 : i64]} {
  func.func @zero_extent() -> index {
    %c0 = arith.constant 0 : index
    %y = "ttkernel.my_logical_y_"() : () -> index
    %pred = arith.cmpi eq, %y, %c0 : index
    %r = scf.if %pred -> (index) {
      scf.yield %c0 : index
    } else {
      scf.yield %y : index
    }
    return %r : index
  }
}

// -----

// -- Function with symbol uses is skipped, not fatal. -------------------------
// Cloning renames the function and erases the original, leaving any caller
// dangling. Since inter-function SymbolRefAttr fixups are not performed, the
// pass leaves such a function un-specialized (a warning, not an error) so
// unrelated functions still get specialized. If it wrongly cloned and erased
// @helper, @caller's call would dangle and surface here as an unexpected error.

module attributes {ttl.launch_grid = [1 : i64, 2 : i64]} {
  // expected-warning @+1 {{not specializing}}
  func.func @helper() -> index {
    %c0 = arith.constant 0 : index
    %y = "ttkernel.my_logical_y_"() : () -> index
    %pred = arith.cmpi eq, %y, %c0 : index
    %r = scf.if %pred -> (index) {
      scf.yield %c0 : index
    } else {
      scf.yield %y : index
    }
    return %r : index
  }
  func.func @caller() -> index {
    %v = func.call @helper() : () -> index
    return %v : index
  }
}
