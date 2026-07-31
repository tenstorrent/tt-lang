// RUN: ttlang-launch-node-domain-test %s | FileCheck %s

// Summary: Prints exact launch-node lattice values for full, narrowed, empty,
// joined, and unknown execution domains.

// CHECK:      entry = {(0,0), (0,1), (1,0), (1,1)}
// CHECK-NEXT: x_zero = {(0,0), (0,1)}
// CHECK-NEXT: x_nonzero = {(1,0), (1,1)}
// CHECK-NEXT: joined = {(0,0), (0,1), (1,0), (1,1)}
// CHECK-NEXT: empty = {}
// CHECK-NEXT: unknown = <unknown>
// CHECK-NEXT: undeclared_pipe = <unknown>
// CHECK-NOT:  =

module attributes {ttl.launch_grid = [2 : i64, 2 : i64]} {
  func.func @domains() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    "test.observe"() {test.label = "entry"} : () -> ()
    %core_x = ttl.core_x : index
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

    %unresolved = "test.coordinate_predicate"(%core_x) : (index) -> i1
    scf.if %unresolved {
      "test.observe"() {test.label = "unknown"} : () -> ()
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
