// RUN: ttlang-launch-node-domain-test %s | FileCheck %s

// Summary: Prints launch-node lattice values for exact, empty, and bounded
// unknown execution domains and conditional-execution equivalence results.

// CHECK:      entry = {(0,0), (0,1), (1,0), (1,1)}
// CHECK-NEXT: x_zero = {(0,0), (0,1)}
// CHECK-NEXT: x_nonzero = {(1,0), (1,1)}
// CHECK-NEXT: not_x_zero = {(1,0), (1,1)}
// CHECK-NEXT: scf_if_result = {(0,0), (0,1)}
// CHECK-NEXT: nested_scf_if_result = {(0,0), (0,1)}
// CHECK-NEXT: large_integer_expression = {(0,0), (0,1)}
// CHECK-NEXT: joined = {(0,0), (0,1), (1,0), (1,1)}
// CHECK-NEXT: empty = {}
// CHECK-NEXT: bounded_unknown = <unknown> within {(0,0), (0,1)}
// CHECK-NEXT: full_bound_unknown = <unknown> within {(0,0), (0,1), (1,0), (1,1)}
// CHECK-NEXT: undeclared_pipe = <unknown> within {(0,0), (0,1), (1,0), (1,1)}
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
    %not_x_zero = emitc.logical_not %is_x_zero : i1
    scf.if %not_x_zero {
      "test.observe"() {test.label = "not_x_zero"} : () -> ()
    }

    // Coordinate expressions emitted by the frontend use scf.if results.
    // Evaluate the selected yield before interpreting the outer predicate.
    %c1_selected = arith.constant 1 : index
    %selected_x = scf.if %is_x_zero -> (index) {
      scf.yield %c1_selected : index
    } else {
      scf.yield %c0 : index
    }
    %selected_first_column = arith.cmpi eq, %selected_x, %c1_selected : index
    scf.if %selected_first_column {
      "test.observe"() {test.label = "scf_if_result"} : () -> ()
    }

    %c2_selected = arith.constant 2 : index
    %first_pair:2 = scf.if %is_x_zero -> (index, index) {
      scf.yield %c2_selected, %c0 : index, index
    } else {
      scf.yield %core_x, %core_y : index, index
    }
    %first_pair_selected = arith.cmpi eq, %first_pair#0, %c2_selected : index
    %second_pair:2 = scf.if %first_pair_selected -> (index, index) {
      scf.yield %first_pair#1, %first_pair#0 : index, index
    } else {
      scf.yield %c0, %c0 : index, index
    }
    %nested_pair_selected = arith.cmpi eq, %second_pair#1, %c2_selected : index
    scf.if %nested_pair_selected {
      "test.observe"() {test.label = "nested_scf_if_result"} : () -> ()
    }

    // This long expression forces evaluator cache growth and verifies that
    // replacement folding remains exact across reallocation.
    %x_outside_grid = arith.cmpi eq, %core_x, %c2_selected : index
    %selected_value0 = arith.select %x_outside_grid, %c0, %core_x : index
    %selected_value1 = arith.select %x_outside_grid, %c0, %selected_value0 : index
    %selected_value2 = arith.select %x_outside_grid, %c0, %selected_value1 : index
    %selected_value3 = arith.select %x_outside_grid, %c0, %selected_value2 : index
    %selected_value4 = arith.select %x_outside_grid, %c0, %selected_value3 : index
    %selected_value5 = arith.select %x_outside_grid, %c0, %selected_value4 : index
    %selected_value6 = arith.select %x_outside_grid, %c0, %selected_value5 : index
    %selected_value7 = arith.select %x_outside_grid, %c0, %selected_value6 : index
    %selected_value8 = arith.select %x_outside_grid, %c0, %selected_value7 : index
    %selected_value9 = arith.select %x_outside_grid, %c0, %selected_value8 : index
    %selected_value10 = arith.select %x_outside_grid, %c0, %selected_value9 : index
    %selected_value11 = arith.select %x_outside_grid, %c0, %selected_value10 : index
    %selected_value12 = arith.select %x_outside_grid, %c0, %selected_value11 : index
    %selected_value13 = arith.select %x_outside_grid, %c0, %selected_value12 : index
    %selected_value14 = arith.select %x_outside_grid, %c0, %selected_value13 : index
    %selected_value15 = arith.select %x_outside_grid, %c0, %selected_value14 : index
    %selected_value16 = arith.select %x_outside_grid, %c0, %selected_value15 : index
    %selected_value17 = arith.select %x_outside_grid, %c0, %selected_value16 : index
    %selected_value18 = arith.select %x_outside_grid, %c0, %selected_value17 : index
    %selected_value19 = arith.select %x_outside_grid, %c0, %selected_value18 : index
    %selected_value20 = arith.select %x_outside_grid, %c0, %selected_value19 : index
    %selected_value21 = arith.select %x_outside_grid, %c0, %selected_value20 : index
    %selected_value22 = arith.select %x_outside_grid, %c0, %selected_value21 : index
    %selected_value23 = arith.select %x_outside_grid, %c0, %selected_value22 : index
    %selected_value24 = arith.select %x_outside_grid, %c0, %selected_value23 : index
    %selected_value25 = arith.select %x_outside_grid, %c0, %selected_value24 : index
    %selected_value26 = arith.select %x_outside_grid, %c0, %selected_value25 : index
    %selected_value27 = arith.select %x_outside_grid, %c0, %selected_value26 : index
    %selected_value28 = arith.select %x_outside_grid, %c0, %selected_value27 : index
    %selected_value29 = arith.select %x_outside_grid, %c0, %selected_value28 : index
    %selected_value30 = arith.select %x_outside_grid, %c0, %selected_value29 : index
    %selected_value31 = arith.select %x_outside_grid, %c0, %selected_value30 : index
    %selected_value32 = arith.select %x_outside_grid, %c0, %selected_value31 : index
    %selected_value33 = arith.select %x_outside_grid, %c0, %selected_value32 : index
    %selected_value34 = arith.select %x_outside_grid, %c0, %selected_value33 : index
    %selected_value35 = arith.select %x_outside_grid, %c0, %selected_value34 : index
    %selected_value36 = arith.select %x_outside_grid, %c0, %selected_value35 : index
    %selected_value37 = arith.select %x_outside_grid, %c0, %selected_value36 : index
    %selected_value38 = arith.select %x_outside_grid, %c0, %selected_value37 : index
    %selected_value39 = arith.select %x_outside_grid, %c0, %selected_value38 : index
    %selected_value40 = arith.select %x_outside_grid, %c0, %selected_value39 : index
    %selected_value41 = arith.select %x_outside_grid, %c0, %selected_value40 : index
    %selected_value42 = arith.select %x_outside_grid, %c0, %selected_value41 : index
    %selected_value43 = arith.select %x_outside_grid, %c0, %selected_value42 : index
    %large_expression_is_zero = arith.cmpi eq, %selected_value43, %c0 : index
    scf.if %large_expression_is_zero {
      "test.observe"() {test.label = "large_integer_expression"} : () -> ()
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
