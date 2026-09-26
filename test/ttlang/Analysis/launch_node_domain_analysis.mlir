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
// CHECK-NEXT: commuted_conjunction = equivalent
// CHECK-NEXT: de_morgan = equivalent
// CHECK-NEXT: commuted_exclusive_or = equivalent
// CHECK-NEXT: complement = not-equivalent
// CHECK-NEXT: helper_argument_leaf = not-equivalent
// CHECK-NEXT: pair_sum_within_budget = equivalent
// CHECK-NEXT: pair_sum_beyond_budget = not-equivalent
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

  // Conjunctions of dispatch conditions with operands in either order name
  // one condition.
  func.func @commuted_conjunction()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %zero = arith.constant 0 : i32
    %first = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %second = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %first_set = arith.cmpi ne, %first, %zero : i32
    %second_set = arith.cmpi ne, %second, %zero : i32
    %both = arith.andi %first_set, %second_set : i1
    %both_again = arith.andi %second_set, %first_set : i1
    scf.if %both {
      "test.observe"() {test.conditional_pair = "commuted_conjunction"}
          : () -> ()
    }
    scf.if %both_again {
      "test.observe"() {test.conditional_pair = "commuted_conjunction"}
          : () -> ()
    }
    func.return
  }

  // not (a and b) and (not a) or (not b) are one condition.
  func.func @de_morgan()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %zero = arith.constant 0 : i32
    %first = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %first_set = arith.cmpi ne, %first, %zero : i32
    %second = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %second_set = arith.cmpi ne, %second, %zero : i32
    %false = arith.constant false
    %both = arith.andi %first_set, %second_set : i1
    %not_both = arith.cmpi eq, %both, %false : i1
    %first_clear = arith.cmpi eq, %first, %zero : i32
    %second_clear = arith.cmpi eq, %second, %zero : i32
    %either_clear = arith.ori %first_clear, %second_clear : i1
    scf.if %not_both {
      "test.observe"() {test.conditional_pair = "de_morgan"}
          : () -> ()
    }
    scf.if %either_clear {
      "test.observe"() {test.conditional_pair = "de_morgan"}
          : () -> ()
    }
    func.return
  }

  func.func @commuted_exclusive_or()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %zero = arith.constant 0 : i32
    %first = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %first_set = arith.cmpi ne, %first, %zero : i32
    %second = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %second_set = arith.cmpi ne, %second, %zero : i32
    %differ = arith.xori %first_set, %second_set : i1
    %differ_again = arith.xori %second_set, %first_set : i1
    scf.if %differ {
      "test.observe"() {test.conditional_pair = "commuted_exclusive_or"}
          : () -> ()
    }
    scf.if %differ_again {
      "test.observe"() {test.conditional_pair = "commuted_exclusive_or"}
          : () -> ()
    }
    func.return
  }

  func.func @complement()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %zero = arith.constant 0 : i32
    %first = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %first_set = arith.cmpi ne, %first, %zero : i32
    %first_clear = arith.cmpi eq, %first, %zero : i32
    scf.if %first_set {
      "test.observe"() {test.conditional_pair = "complement"}
          : () -> ()
    }
    scf.if %first_clear {
      "test.observe"() {test.conditional_pair = "complement"}
          : () -> ()
    }
    func.return
  }

  // A helper argument is not a dispatch condition, so an expression over it is
  // never proven, even against the same expression with commuted operands.
  func.func private @helper_argument_leaf(%condition: i1) {
    %zero = arith.constant 0 : i32
    %first = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %first_set = arith.cmpi ne, %first, %zero : i32
    %either = arith.ori %first_set, %condition : i1
    %either_again = arith.ori %condition, %first_set : i1
    scf.if %either {
      "test.observe"() {test.conditional_pair = "helper_argument_leaf"}
          : () -> ()
    }
    scf.if %either_again {
      "test.observe"() {test.conditional_pair = "helper_argument_leaf"}
          : () -> ()
    }
    func.return
  }

  // A sum of pairwise conjunctions whose first-seen variable order puts every
  // x before every y. Its decision diagram has about 2^(n+1) nodes: n = 8 fits
  // the node budget, n = 17 exceeds it and is never proven.
  func.func @pair_sum_within_budget()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %zero = arith.constant 0 : i32
    %x0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %x0_set = arith.cmpi ne, %x0, %zero : i32
    %x1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %x1_set = arith.cmpi ne, %x1, %zero : i32
    %x2 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<2, i32>, header = "predicate.hpp"} : () -> i32
    %x2_set = arith.cmpi ne, %x2, %zero : i32
    %x3 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<3, i32>, header = "predicate.hpp"} : () -> i32
    %x3_set = arith.cmpi ne, %x3, %zero : i32
    %x4 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<4, i32>, header = "predicate.hpp"} : () -> i32
    %x4_set = arith.cmpi ne, %x4, %zero : i32
    %x5 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<5, i32>, header = "predicate.hpp"} : () -> i32
    %x5_set = arith.cmpi ne, %x5, %zero : i32
    %x6 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<6, i32>, header = "predicate.hpp"} : () -> i32
    %x6_set = arith.cmpi ne, %x6, %zero : i32
    %x7 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<7, i32>, header = "predicate.hpp"} : () -> i32
    %x7_set = arith.cmpi ne, %x7, %zero : i32
    %y0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<8, i32>, header = "predicate.hpp"} : () -> i32
    %y0_set = arith.cmpi ne, %y0, %zero : i32
    %y1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<9, i32>, header = "predicate.hpp"} : () -> i32
    %y1_set = arith.cmpi ne, %y1, %zero : i32
    %y2 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<10, i32>, header = "predicate.hpp"} : () -> i32
    %y2_set = arith.cmpi ne, %y2, %zero : i32
    %y3 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<11, i32>, header = "predicate.hpp"} : () -> i32
    %y3_set = arith.cmpi ne, %y3, %zero : i32
    %y4 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<12, i32>, header = "predicate.hpp"} : () -> i32
    %y4_set = arith.cmpi ne, %y4, %zero : i32
    %y5 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<13, i32>, header = "predicate.hpp"} : () -> i32
    %y5_set = arith.cmpi ne, %y5, %zero : i32
    %y6 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<14, i32>, header = "predicate.hpp"} : () -> i32
    %y6_set = arith.cmpi ne, %y6, %zero : i32
    %y7 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<15, i32>, header = "predicate.hpp"} : () -> i32
    %y7_set = arith.cmpi ne, %y7, %zero : i32
    %false = arith.constant false
    %all_x6 = arith.andi %x6_set, %x7_set : i1
    %all_x5 = arith.andi %x5_set, %all_x6 : i1
    %all_x4 = arith.andi %x4_set, %all_x5 : i1
    %all_x3 = arith.andi %x3_set, %all_x4 : i1
    %all_x2 = arith.andi %x2_set, %all_x3 : i1
    %all_x1 = arith.andi %x1_set, %all_x2 : i1
    %all_x0 = arith.andi %x0_set, %all_x1 : i1
    %ordered = arith.andi %false, %all_x0 : i1
    %term0 = arith.andi %x0_set, %y0_set : i1
    %term_again0 = arith.andi %y0_set, %x0_set : i1
    %term1 = arith.andi %x1_set, %y1_set : i1
    %term_again1 = arith.andi %y1_set, %x1_set : i1
    %term2 = arith.andi %x2_set, %y2_set : i1
    %term_again2 = arith.andi %y2_set, %x2_set : i1
    %term3 = arith.andi %x3_set, %y3_set : i1
    %term_again3 = arith.andi %y3_set, %x3_set : i1
    %term4 = arith.andi %x4_set, %y4_set : i1
    %term_again4 = arith.andi %y4_set, %x4_set : i1
    %term5 = arith.andi %x5_set, %y5_set : i1
    %term_again5 = arith.andi %y5_set, %x5_set : i1
    %term6 = arith.andi %x6_set, %y6_set : i1
    %term_again6 = arith.andi %y6_set, %x6_set : i1
    %term7 = arith.andi %x7_set, %y7_set : i1
    %term_again7 = arith.andi %y7_set, %x7_set : i1
    %sum0 = arith.ori %ordered, %term0 : i1
    %sum1 = arith.ori %sum0, %term1 : i1
    %sum2 = arith.ori %sum1, %term2 : i1
    %sum3 = arith.ori %sum2, %term3 : i1
    %sum4 = arith.ori %sum3, %term4 : i1
    %sum5 = arith.ori %sum4, %term5 : i1
    %sum6 = arith.ori %sum5, %term6 : i1
    %sum7 = arith.ori %sum6, %term7 : i1
    %sum_again1 = arith.ori %term_again1, %term_again0 : i1
    %sum_again2 = arith.ori %term_again2, %sum_again1 : i1
    %sum_again3 = arith.ori %term_again3, %sum_again2 : i1
    %sum_again4 = arith.ori %term_again4, %sum_again3 : i1
    %sum_again5 = arith.ori %term_again5, %sum_again4 : i1
    %sum_again6 = arith.ori %term_again6, %sum_again5 : i1
    %sum_again7 = arith.ori %term_again7, %sum_again6 : i1
    scf.if %sum7 {
      "test.observe"() {test.conditional_pair = "pair_sum_within_budget"}
          : () -> ()
    }
    scf.if %sum_again7 {
      "test.observe"() {test.conditional_pair = "pair_sum_within_budget"}
          : () -> ()
    }
    func.return
  }

  func.func @pair_sum_beyond_budget()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %zero = arith.constant 0 : i32
    %x0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %x0_set = arith.cmpi ne, %x0, %zero : i32
    %x1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %x1_set = arith.cmpi ne, %x1, %zero : i32
    %x2 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<2, i32>, header = "predicate.hpp"} : () -> i32
    %x2_set = arith.cmpi ne, %x2, %zero : i32
    %x3 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<3, i32>, header = "predicate.hpp"} : () -> i32
    %x3_set = arith.cmpi ne, %x3, %zero : i32
    %x4 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<4, i32>, header = "predicate.hpp"} : () -> i32
    %x4_set = arith.cmpi ne, %x4, %zero : i32
    %x5 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<5, i32>, header = "predicate.hpp"} : () -> i32
    %x5_set = arith.cmpi ne, %x5, %zero : i32
    %x6 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<6, i32>, header = "predicate.hpp"} : () -> i32
    %x6_set = arith.cmpi ne, %x6, %zero : i32
    %x7 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<7, i32>, header = "predicate.hpp"} : () -> i32
    %x7_set = arith.cmpi ne, %x7, %zero : i32
    %x8 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<8, i32>, header = "predicate.hpp"} : () -> i32
    %x8_set = arith.cmpi ne, %x8, %zero : i32
    %x9 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<9, i32>, header = "predicate.hpp"} : () -> i32
    %x9_set = arith.cmpi ne, %x9, %zero : i32
    %x10 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<10, i32>, header = "predicate.hpp"} : () -> i32
    %x10_set = arith.cmpi ne, %x10, %zero : i32
    %x11 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<11, i32>, header = "predicate.hpp"} : () -> i32
    %x11_set = arith.cmpi ne, %x11, %zero : i32
    %x12 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<12, i32>, header = "predicate.hpp"} : () -> i32
    %x12_set = arith.cmpi ne, %x12, %zero : i32
    %x13 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<13, i32>, header = "predicate.hpp"} : () -> i32
    %x13_set = arith.cmpi ne, %x13, %zero : i32
    %x14 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<14, i32>, header = "predicate.hpp"} : () -> i32
    %x14_set = arith.cmpi ne, %x14, %zero : i32
    %x15 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<15, i32>, header = "predicate.hpp"} : () -> i32
    %x15_set = arith.cmpi ne, %x15, %zero : i32
    %x16 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<16, i32>, header = "predicate.hpp"} : () -> i32
    %x16_set = arith.cmpi ne, %x16, %zero : i32
    %y0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<17, i32>, header = "predicate.hpp"} : () -> i32
    %y0_set = arith.cmpi ne, %y0, %zero : i32
    %y1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<18, i32>, header = "predicate.hpp"} : () -> i32
    %y1_set = arith.cmpi ne, %y1, %zero : i32
    %y2 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<19, i32>, header = "predicate.hpp"} : () -> i32
    %y2_set = arith.cmpi ne, %y2, %zero : i32
    %y3 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<20, i32>, header = "predicate.hpp"} : () -> i32
    %y3_set = arith.cmpi ne, %y3, %zero : i32
    %y4 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<21, i32>, header = "predicate.hpp"} : () -> i32
    %y4_set = arith.cmpi ne, %y4, %zero : i32
    %y5 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<22, i32>, header = "predicate.hpp"} : () -> i32
    %y5_set = arith.cmpi ne, %y5, %zero : i32
    %y6 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<23, i32>, header = "predicate.hpp"} : () -> i32
    %y6_set = arith.cmpi ne, %y6, %zero : i32
    %y7 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<24, i32>, header = "predicate.hpp"} : () -> i32
    %y7_set = arith.cmpi ne, %y7, %zero : i32
    %y8 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<25, i32>, header = "predicate.hpp"} : () -> i32
    %y8_set = arith.cmpi ne, %y8, %zero : i32
    %y9 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<26, i32>, header = "predicate.hpp"} : () -> i32
    %y9_set = arith.cmpi ne, %y9, %zero : i32
    %y10 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<27, i32>, header = "predicate.hpp"} : () -> i32
    %y10_set = arith.cmpi ne, %y10, %zero : i32
    %y11 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<28, i32>, header = "predicate.hpp"} : () -> i32
    %y11_set = arith.cmpi ne, %y11, %zero : i32
    %y12 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<29, i32>, header = "predicate.hpp"} : () -> i32
    %y12_set = arith.cmpi ne, %y12, %zero : i32
    %y13 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<30, i32>, header = "predicate.hpp"} : () -> i32
    %y13_set = arith.cmpi ne, %y13, %zero : i32
    %y14 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<31, i32>, header = "predicate.hpp"} : () -> i32
    %y14_set = arith.cmpi ne, %y14, %zero : i32
    %y15 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<32, i32>, header = "predicate.hpp"} : () -> i32
    %y15_set = arith.cmpi ne, %y15, %zero : i32
    %y16 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<33, i32>, header = "predicate.hpp"} : () -> i32
    %y16_set = arith.cmpi ne, %y16, %zero : i32
    %false = arith.constant false
    %all_x15 = arith.andi %x15_set, %x16_set : i1
    %all_x14 = arith.andi %x14_set, %all_x15 : i1
    %all_x13 = arith.andi %x13_set, %all_x14 : i1
    %all_x12 = arith.andi %x12_set, %all_x13 : i1
    %all_x11 = arith.andi %x11_set, %all_x12 : i1
    %all_x10 = arith.andi %x10_set, %all_x11 : i1
    %all_x9 = arith.andi %x9_set, %all_x10 : i1
    %all_x8 = arith.andi %x8_set, %all_x9 : i1
    %all_x7 = arith.andi %x7_set, %all_x8 : i1
    %all_x6 = arith.andi %x6_set, %all_x7 : i1
    %all_x5 = arith.andi %x5_set, %all_x6 : i1
    %all_x4 = arith.andi %x4_set, %all_x5 : i1
    %all_x3 = arith.andi %x3_set, %all_x4 : i1
    %all_x2 = arith.andi %x2_set, %all_x3 : i1
    %all_x1 = arith.andi %x1_set, %all_x2 : i1
    %all_x0 = arith.andi %x0_set, %all_x1 : i1
    %ordered = arith.andi %false, %all_x0 : i1
    %term0 = arith.andi %x0_set, %y0_set : i1
    %term_again0 = arith.andi %y0_set, %x0_set : i1
    %term1 = arith.andi %x1_set, %y1_set : i1
    %term_again1 = arith.andi %y1_set, %x1_set : i1
    %term2 = arith.andi %x2_set, %y2_set : i1
    %term_again2 = arith.andi %y2_set, %x2_set : i1
    %term3 = arith.andi %x3_set, %y3_set : i1
    %term_again3 = arith.andi %y3_set, %x3_set : i1
    %term4 = arith.andi %x4_set, %y4_set : i1
    %term_again4 = arith.andi %y4_set, %x4_set : i1
    %term5 = arith.andi %x5_set, %y5_set : i1
    %term_again5 = arith.andi %y5_set, %x5_set : i1
    %term6 = arith.andi %x6_set, %y6_set : i1
    %term_again6 = arith.andi %y6_set, %x6_set : i1
    %term7 = arith.andi %x7_set, %y7_set : i1
    %term_again7 = arith.andi %y7_set, %x7_set : i1
    %term8 = arith.andi %x8_set, %y8_set : i1
    %term_again8 = arith.andi %y8_set, %x8_set : i1
    %term9 = arith.andi %x9_set, %y9_set : i1
    %term_again9 = arith.andi %y9_set, %x9_set : i1
    %term10 = arith.andi %x10_set, %y10_set : i1
    %term_again10 = arith.andi %y10_set, %x10_set : i1
    %term11 = arith.andi %x11_set, %y11_set : i1
    %term_again11 = arith.andi %y11_set, %x11_set : i1
    %term12 = arith.andi %x12_set, %y12_set : i1
    %term_again12 = arith.andi %y12_set, %x12_set : i1
    %term13 = arith.andi %x13_set, %y13_set : i1
    %term_again13 = arith.andi %y13_set, %x13_set : i1
    %term14 = arith.andi %x14_set, %y14_set : i1
    %term_again14 = arith.andi %y14_set, %x14_set : i1
    %term15 = arith.andi %x15_set, %y15_set : i1
    %term_again15 = arith.andi %y15_set, %x15_set : i1
    %term16 = arith.andi %x16_set, %y16_set : i1
    %term_again16 = arith.andi %y16_set, %x16_set : i1
    %sum0 = arith.ori %ordered, %term0 : i1
    %sum1 = arith.ori %sum0, %term1 : i1
    %sum2 = arith.ori %sum1, %term2 : i1
    %sum3 = arith.ori %sum2, %term3 : i1
    %sum4 = arith.ori %sum3, %term4 : i1
    %sum5 = arith.ori %sum4, %term5 : i1
    %sum6 = arith.ori %sum5, %term6 : i1
    %sum7 = arith.ori %sum6, %term7 : i1
    %sum8 = arith.ori %sum7, %term8 : i1
    %sum9 = arith.ori %sum8, %term9 : i1
    %sum10 = arith.ori %sum9, %term10 : i1
    %sum11 = arith.ori %sum10, %term11 : i1
    %sum12 = arith.ori %sum11, %term12 : i1
    %sum13 = arith.ori %sum12, %term13 : i1
    %sum14 = arith.ori %sum13, %term14 : i1
    %sum15 = arith.ori %sum14, %term15 : i1
    %sum16 = arith.ori %sum15, %term16 : i1
    %sum_again1 = arith.ori %term_again1, %term_again0 : i1
    %sum_again2 = arith.ori %term_again2, %sum_again1 : i1
    %sum_again3 = arith.ori %term_again3, %sum_again2 : i1
    %sum_again4 = arith.ori %term_again4, %sum_again3 : i1
    %sum_again5 = arith.ori %term_again5, %sum_again4 : i1
    %sum_again6 = arith.ori %term_again6, %sum_again5 : i1
    %sum_again7 = arith.ori %term_again7, %sum_again6 : i1
    %sum_again8 = arith.ori %term_again8, %sum_again7 : i1
    %sum_again9 = arith.ori %term_again9, %sum_again8 : i1
    %sum_again10 = arith.ori %term_again10, %sum_again9 : i1
    %sum_again11 = arith.ori %term_again11, %sum_again10 : i1
    %sum_again12 = arith.ori %term_again12, %sum_again11 : i1
    %sum_again13 = arith.ori %term_again13, %sum_again12 : i1
    %sum_again14 = arith.ori %term_again14, %sum_again13 : i1
    %sum_again15 = arith.ori %term_again15, %sum_again14 : i1
    %sum_again16 = arith.ori %term_again16, %sum_again15 : i1
    scf.if %sum16 {
      "test.observe"() {test.conditional_pair = "pair_sum_beyond_budget"}
          : () -> ()
    }
    scf.if %sum_again16 {
      "test.observe"() {test.conditional_pair = "pair_sum_beyond_budget"}
          : () -> ()
    }
    func.return
  }
}
