// RUN: ttlang-launch-node-domain-test %s | FileCheck %s

// Summary: Prints launch-node lattice values for exact, empty, and bounded
// unknown execution domains, including a bound covering the complete grid.

// CHECK:      entry = {(0,0), (0,1), (1,0), (1,1)}
// CHECK-NEXT: x_zero = {(0,0), (0,1)}
// CHECK-NEXT: x_nonzero = {(1,0), (1,1)}
// CHECK-NEXT: joined = {(0,0), (0,1), (1,0), (1,1)}
// CHECK-NEXT: empty = {}
// CHECK-NEXT: selected_nonzero = {(1,0), (1,1)}
// CHECK-NEXT: bounded_unknown = <unknown> within {(0,0), (0,1)}
// CHECK-NEXT: full_bound_unknown = <unknown> within {(0,0), (0,1), (1,0), (1,1)}
// CHECK-NEXT: undeclared_pipe = <unknown> within {(0,0), (0,1), (1,0), (1,1)}
// CHECK-NOT:  =

module attributes {ttl.launch_grid = [2 : i64, 2 : i64]} {
  func.func @domains(%runtime: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    "test.observe"() {test.label = "entry"} : () -> ()
    %core_x = ttl.core_x : index
    %core_y = ttl.core_y : index
    %c0 = arith.constant 0 : index
    %is_x_zero = arith.cmpi eq, %core_x, %c0 : index
    scf.if %is_x_zero {
      "test.observe"() {test.label = "x_zero"} : () -> ()
    } else {
      "test.observe"() {test.label = "x_nonzero"} : () -> ()
    }
    "test.observe"() {test.label = "joined"} : () -> ()

    %c3 = arith.constant 3 : index
    %is_outside_grid = arith.cmpi eq, %core_x, %c3 : index
    scf.if %is_outside_grid {
      "test.observe"() {test.label = "empty"} : () -> ()
    }

    %selected_coordinate = scf.if %is_x_zero -> (index) {
      scf.yield %c0 : index
    } else {
      scf.yield %core_x : index
    }
    %selected_is_zero = arith.cmpi eq, %selected_coordinate, %c0 : index
    %selected_is_nonzero = emitc.logical_not %selected_is_zero : i1
    scf.if %selected_is_nonzero {
      "test.observe"() {test.label = "selected_nonzero"} : () -> ()
    }

    %runtime_coordinate = arith.addi %core_y, %runtime : index
    %runtime_selected = arith.cmpi eq, %runtime_coordinate, %c0 : index
    scf.if %is_x_zero {
      scf.if %runtime_selected {
        "test.observe"() {test.label = "bounded_unknown"} : () -> ()
      }
    }

    %unresolved = "test.coordinate_predicate"(%core_x) : (index) -> i1
    scf.if %unresolved {
      "test.observe"() {test.label = "full_bound_unknown"} : () -> ()
    }

    // An undeclared PipeNet predicate has no role domain. The standalone
    // analysis reports an unknown domain instead of asserting.
    %undeclared = ttl.is_src {pipe_net_id = 7 : i64}
    scf.if %undeclared {
      "test.observe"() {test.label = "undeclared_pipe"} : () -> ()
    }
    func.return
  }
}
