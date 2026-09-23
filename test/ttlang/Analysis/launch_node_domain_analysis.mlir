// RUN: ttlang-launch-node-domain-test %s | FileCheck %s

// Summary: Prints launch-node lattice values for exact, empty, and bounded
// unknown execution domains and conditional-execution equivalence results.

// CHECK:      entry = {(0,0), (0,1), (1,0), (1,1)}
// CHECK-NEXT: x_zero = {(0,0), (0,1)}
// CHECK-NEXT: x_nonzero = {(1,0), (1,1)}
// CHECK-NEXT: joined = {(0,0), (0,1), (1,0), (1,1)}
// CHECK-NEXT: empty = {}
// CHECK-NEXT: bounded_unknown = <unknown> within {(0,0), (0,1)}
// CHECK-NEXT: full_bound_unknown = <unknown> within {(0,0), (0,1), (1,0), (1,1)}
// CHECK-NEXT: undeclared_pipe = <unknown> within {(0,0), (0,1), (1,0), (1,1)}
// CHECK-NEXT: constant_false_else = {(0,0), (0,1), (1,0), (1,1)}
// CHECK-NEXT: constant_true_then = {(0,0), (0,1), (1,0), (1,1)}
// CHECK-NEXT: false_or_coordinate_then = {(0,0), (0,1)}
// CHECK-NEXT: false_or_coordinate_else = {(1,0), (1,1)}
// CHECK-NEXT: true_and_coordinate_then = {(0,0), (0,1)}
// CHECK-NEXT: true_and_coordinate_else = {(1,0), (1,1)}
// CHECK-NEXT: coordinate_and_false_else = {(0,0), (0,1), (1,0), (1,1)}
// CHECK-NEXT: coordinate_or_true_then = {(0,0), (0,1), (1,0), (1,1)}
// CHECK-NEXT: runtime_or_false_then = {(0,0), (0,1), (1,0), (1,1)}
// CHECK-NEXT: runtime_or_false_else = {(0,0), (0,1), (1,0), (1,1)}
// CHECK-NEXT: destination_count_loop = {(1,0)}
// CHECK-NEXT: graph_destination_count_loop = {(0,0), (0,1), (1,0), (1,1)}
// CHECK-NEXT: kernel_argument_condition = equivalent
// CHECK-NEXT: helper_argument_condition = not-equivalent
// CHECK-NOT:  =

#destination_records = #ttl.pipenet_records<
    net 0 name "destination_count" pipes [
  <srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0,
   dstEndX = 1, dstEndY = 0>
]>

#device_domain = #ttl.device_domain<
    components = <name = "device", extent = [2]>>
#graph_destination_records = #ttl.pipenet_records<
    net 1 name "graph_destination_count" pipes [
  #ttl.pipe_record<
      srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0,
      dstEndX = 1, dstEndY = 0,
      deviceTransfer = <
        domain = #device_domain,
        edge = <source = <coordinates = [0]>,
                destination = <coordinates = [1]>>>>
]>

#closed_form_records = #ttl.pipenet_records<net 2 mappings
  <graph = <domain = <components = <name = "device", extent = [3]>>,
    kind = all_to_all, componentName = "device", properties = {}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>,
         <srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 0,
          dstEndX = 1, dstEndY = 0>]>>

module attributes {
  ttl.launch_grid = [2 : i64, 2 : i64],
  test.closed_form_records = #closed_form_records
} {
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

    %false = arith.constant false
    %true = arith.constant true
    scf.if %false {
    } else {
      "test.observe"() {test.label = "constant_false_else"} : () -> ()
    }
    scf.if %true {
      "test.observe"() {test.label = "constant_true_then"} : () -> ()
    } else {
    }

    // Composed node-membership guards retain their constant initializer until
    // after launch-domain verification.
    %false_or_coordinate = arith.ori %false, %is_x_zero : i1
    scf.if %false_or_coordinate {
      "test.observe"() {test.label = "false_or_coordinate_then"} : () -> ()
    } else {
      "test.observe"() {test.label = "false_or_coordinate_else"} : () -> ()
    }
    %true_and_coordinate = arith.andi %true, %is_x_zero : i1
    scf.if %true_and_coordinate {
      "test.observe"() {test.label = "true_and_coordinate_then"} : () -> ()
    } else {
      "test.observe"() {test.label = "true_and_coordinate_else"} : () -> ()
    }
    %coordinate_and_false = arith.andi %is_x_zero, %false : i1
    scf.if %coordinate_and_false {
    } else {
      "test.observe"() {test.label = "coordinate_and_false_else"} : () -> ()
    }
    %coordinate_or_true = arith.ori %is_x_zero, %true : i1
    scf.if %coordinate_or_true {
      "test.observe"() {test.label = "coordinate_or_true_then"} : () -> ()
    } else {
    }

    // A runtime predicate can still select either branch on any launch node.
    %runtime_condition = arith.cmpi eq, %runtime, %c0 : index
    %runtime_or_false = arith.ori %runtime_condition, %false : i1
    scf.if %runtime_or_false {
      "test.observe"() {test.label = "runtime_or_false_then"} : () -> ()
    } else {
      "test.observe"() {test.label = "runtime_or_false_else"} : () -> ()
    }

    // A local count removes nodes with no matching destination record.
    %destination_count = ttl.pipenet_destination_count {
        pipe_net_id = 0 : i64, records = #destination_records} : index
    %c1 = arith.constant 1 : index
    scf.for %iteration = %c0 to %destination_count step %c1 {
      "test.observe"() {test.label = "destination_count_loop"} : () -> ()
    }

    // Device identity is unavailable to this node-only analysis. Retaining the
    // incoming domain prevents an unproven device match from removing nodes.
    %graph_destination_count = ttl.pipenet_destination_count {
        pipe_net_id = 1 : i64, records = #graph_destination_records} : index
    scf.for %iteration = %c0 to %graph_destination_count step %c1 {
      "test.observe"() {test.label = "graph_destination_count_loop"} : () -> ()
    }
    func.return
  }

  // One kernel entry argument has the same runtime value throughout one
  // launched kernel instance.
  func.func @kernel_argument_condition(%condition: i1)
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    scf.if %condition {
      "test.observe"() {test.conditional_pair = "kernel_argument_condition"}
          : () -> ()
    }
    scf.if %condition {
      "test.observe"() {test.conditional_pair = "kernel_argument_condition"}
          : () -> ()
    }
    func.return
  }

  // A helper argument can receive different values at separate call sites, so
  // SSA identity inside the helper does not prove one runtime condition.
  func.func private @helper_argument_condition(%condition: i1) {
    scf.if %condition {
      "test.observe"() {test.conditional_pair = "helper_argument_condition"}
          : () -> ()
    }
    scf.if %condition {
      "test.observe"() {test.conditional_pair = "helper_argument_condition"}
          : () -> ()
    }
    func.return
  }
}
