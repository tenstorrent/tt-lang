// RUN: ttlang-value-origin-test %s | FileCheck %s

// This file tests possible source definitions through selects, branches,
// loops, casts, and tensor element updates over finite loop index domains.

// A selected scalar may originate from either selected operand.
func.func @select_origins(%condition: i1) {
  %first = "test.source"() {test.label = "first"} : () -> i32
  %second = "test.source"() {test.label = "second"} : () -> i32
  %selected = arith.select %condition, %first, %second : i32
  "test.query"(%selected) {test.expected_origins = ["first", "second"], test.label = "select"} : (i32) -> ()
  return
}

// CHECK: select = [first, second]

// An scf.if result includes values yielded by both regions.
func.func @region_origins(%condition: i1) {
  %selected = scf.if %condition -> i32 {
    %then = "test.source"() {test.label = "then"} : () -> i32
    scf.yield %then : i32
  } else {
    %else = "test.source"() {test.label = "else"} : () -> i32
    scf.yield %else : i32
  }
  "test.query"(%selected) {test.expected_origins = ["then", "else"], test.label = "region"} : (i32) -> ()
  return
}

// CHECK: region = [else, then]

// A one-input/one-result unrealized cast uses its input's source definition.
func.func @unrealized_cast_origin() {
  %source = "test.source"() {test.label = "cast_source"} : () -> i32
  %cast = builtin.unrealized_conversion_cast %source : i32 to index
  "test.query"(%cast) {test.expected_origins = ["cast_source"], test.label = "unrealized_cast"} : (index) -> ()
  return
}

// CHECK: unrealized_cast = [cast_source]

// A multi-result unrealized cast does not associate an individual result with
// its input, so the result itself remains an origin.
func.func @multi_result_unrealized_cast_origin() {
  %source = "test.source"() {test.label = "multi_cast_source"} : () -> i32
  %casts:2 = "builtin.unrealized_conversion_cast"(%source) {test.label = "multi_cast"} : (i32) -> (i16, i16)
  "test.query"(%casts#0) {test.expected_origins = ["multi_cast"], test.label = "multi_result_cast"} : (i16) -> ()
  return
}

// CHECK: multi_result_cast = [multi_cast]

// Block arguments include values from every incoming cf branch.
func.func @block_origins(%condition: i1) {
  cf.cond_br %condition, ^first, ^second
^first:
  %first = "test.source"() {test.label = "block_first"} : () -> i32
  cf.br ^merge(%first : i32)
^second:
  %second = "test.source"() {test.label = "block_second"} : () -> i32
  cf.br ^merge(%second : i32)
^merge(%merged: i32):
  "test.query"(%merged) {test.expected_origins = ["block_first", "block_second"], test.label = "block"} : (i32) -> ()
  return
}

// CHECK: block = [block_first, block_second]

// Cyclic block control flow terminates and includes the values passed to the
// block argument from the entry edge and backedge.
func.func @cyclic_block_origins(%condition: i1) {
  %initial = "test.source"() {test.label = "cycle_initial"} : () -> i32
  cf.br ^loop(%initial : i32)
^loop(%current: i32):
  "test.query"(%current) {test.expected_origins = ["cycle_initial", "cycle_next"], test.label = "cycle"} : (i32) -> ()
  %next = "test.source"() {test.label = "cycle_next"} : () -> i32
  cf.cond_br %condition, ^loop(%next : i32), ^exit
^exit:
  return
}

// CHECK: cycle = [cycle_initial, cycle_next]

// An scf.for induction variable precedes its iterated arguments but is not a
// predecessor of the iterated value.
func.func @loop_iterated_argument() {
  %zero = arith.constant 0 : index
  %two = arith.constant 2 : index
  %one = arith.constant 1 : index
  %initial = "test.source"() {test.label = "loop_initial"} : () -> i32
  %result = scf.for %index = %zero to %two step %one iter_args(%current = %initial) -> i32 {
    "test.query"(%current) {test.expected_origins = ["loop_initial", "loop_update"], test.label = "loop_argument"} : (i32) -> ()
    %updated = "test.source"() {test.label = "loop_update"} : () -> i32
    scf.yield %updated : i32
  }
  "test.query"(%result) {test.expected_origins = ["loop_update"], test.label = "loop_result"} : (i32) -> ()
  return
}

// CHECK: loop_argument = [loop_initial, loop_update]
// CHECK: loop_result = [loop_update]

// A zero-trip loop result comes from its initial value, not the statically
// present yield operand.
func.func @zero_trip_scalar_loop() {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %initial = "test.source"() {test.label = "zero_scalar_initial"} : () -> i32
  %result = scf.for %index = %zero to %zero step %one iter_args(%current = %initial) -> i32 {
    %updated = "test.source"() {test.label = "zero_scalar_update"} : () -> i32
    scf.yield %updated : i32
  }
  "test.query"(%result) {test.expected_origins = ["zero_scalar_initial"], test.label = "zero_scalar"} : (i32) -> ()
  return
}

// CHECK: zero_scalar = [zero_scalar_initial]

// A dynamically bounded loop result may come from the initial value or the
// value yielded after an executed iteration.
func.func @dynamic_trip_scalar_loop(%upper: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %initial = "test.source"() {test.label = "dynamic_scalar_initial"} : () -> i32
  %result = scf.for %index = %zero to %upper step %one iter_args(%current = %initial) -> i32 {
    %updated = "test.source"() {test.label = "dynamic_scalar_update"} : () -> i32
    scf.yield %updated : i32
  }
  "test.query"(%result) {test.expected_origins = ["dynamic_scalar_initial", "dynamic_scalar_update"], test.label = "dynamic_scalar"} : (i32) -> ()
  return
}

// CHECK: dynamic_scalar = [dynamic_scalar_initial, dynamic_scalar_update]

// An scf.while argument may come from the initial operand or a preceding
// iteration even though this loop does not expose LoopLike init operands.
func.func @while_loop_carried_origins(%condition: i1) {
  %initial = "test.source"() {test.label = "while_initial"} : () -> i32
  %result = scf.while (%current = %initial) : (i32) -> i32 {
    "test.query"(%current) {test.expected_origins = ["while_initial", "while_update"], test.label = "while_argument"} : (i32) -> ()
    scf.condition(%condition) %current : i32
  } do {
  ^bb0(%current: i32):
    %updated = "test.source"() {test.label = "while_update"} : () -> i32
    scf.yield %updated : i32
  }
  "test.query"(%result) {test.expected_origins = ["while_initial", "while_update"], test.label = "while_result"} : (i32) -> ()
  return
}

// CHECK: while_argument = [while_initial, while_update]
// CHECK: while_result = [while_initial, while_update]

// Tensor elements follow scf.while initial and backedge values.
func.func @while_tensor_element(%condition: i1) {
  %zero = arith.constant 0 : index
  %base = "test.tensor_source"() {test.label = "while_tensor_base"} : () -> tensor<2xi32>
  %result:2 = scf.while (%tensor = %base, %index = %zero)
      : (tensor<2xi32>, index) -> (tensor<2xi32>, index) {
    scf.condition(%condition) %tensor, %index : tensor<2xi32>, index
  } do {
  ^bb0(%tensor: tensor<2xi32>, %index: index):
    %inserted = "test.source"() {test.label = "while_tensor_insert"} : () -> i32
    %updated = tensor.insert %inserted into %tensor[%index] : tensor<2xi32>
    scf.yield %updated, %index : tensor<2xi32>, index
  }
  %value = tensor.extract %result#0[%zero] : tensor<2xi32>
  "test.query"(%value) {test.expected_origins = ["while_tensor_base", "while_tensor_insert"], test.label = "while_tensor"} : (i32) -> ()
  return
}

// CHECK: while_tensor = [while_tensor_base, while_tensor_insert]

// Equal constant indices select the inserted scalar, not prior tensor data.
func.func @equal_constant_index() {
  %zero = arith.constant 0 : index
  %base = "test.tensor_source"() {test.label = "equal_base"} : () -> tensor<2xi32>
  %inserted = "test.source"() {test.label = "equal_insert"} : () -> i32
  %updated = tensor.insert %inserted into %base[%zero] : tensor<2xi32>
  %value = tensor.extract %updated[%zero] : tensor<2xi32>
  "test.query"(%value) {test.expected_origins = ["equal_insert"], test.label = "equal_index"} : (i32) -> ()
  return
}

// CHECK: equal_index = [equal_insert]

// Distinct constant indices retain the prior tensor element.
func.func @distinct_constant_index() {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %base = "test.tensor_source"() {test.label = "distinct_base"} : () -> tensor<2xi32>
  %inserted = "test.source"() {test.label = "distinct_insert"} : () -> i32
  %updated = tensor.insert %inserted into %base[%zero] : tensor<2xi32>
  %value = tensor.extract %updated[%one] : tensor<2xi32>
  "test.query"(%value) {test.expected_origins = ["distinct_base"], test.label = "distinct_index"} : (i32) -> ()
  return
}

// CHECK: distinct_index = [distinct_base]

// Insert chains retain the definition for the extracted element rather than the
// most recent update to a different element.
func.func @insert_chain() {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %base = "test.tensor_source"() {test.label = "chain_base"} : () -> tensor<2xi32>
  %first = "test.source"() {test.label = "chain_first"} : () -> i32
  %second = "test.source"() {test.label = "chain_second"} : () -> i32
  %first_update = tensor.insert %first into %base[%zero] : tensor<2xi32>
  %second_update = tensor.insert %second into %first_update[%one] : tensor<2xi32>
  %value = tensor.extract %second_update[%zero] : tensor<2xi32>
  "test.query"(%value) {test.expected_origins = ["chain_first"], test.label = "insert_chain"} : (i32) -> ()
  return
}

// CHECK: insert_chain = [chain_first]

// Multi-dimensional indices are compared as tuples, so matching coordinate
// sets in a different order do not identify the inserted element.
func.func @multidimensional_index_tuple() {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %base = "test.tensor_source"() {test.label = "tuple_base"} : () -> tensor<2x2xi32>
  %inserted = "test.source"() {test.label = "tuple_insert"} : () -> i32
  %updated = tensor.insert %inserted into %base[%zero, %one] : tensor<2x2xi32>
  %value = tensor.extract %updated[%one, %zero] : tensor<2x2xi32>
  "test.query"(%value) {test.expected_origins = ["tuple_base"], test.label = "index_tuple"} : (i32) -> ()
  return
}

// CHECK: index_tuple = [tuple_base]

// Tensor casts preserve element definitions and indices.
func.func @tensor_cast() {
  %zero = arith.constant 0 : index
  %base = "test.tensor_source"() {test.label = "cast_base"} : () -> tensor<?xi32>
  %inserted = "test.source"() {test.label = "cast_insert"} : () -> i32
  %updated = tensor.insert %inserted into %base[%zero] : tensor<?xi32>
  %cast = tensor.cast %updated : tensor<?xi32> to tensor<1xi32>
  %value = tensor.extract %cast[%zero] : tensor<1xi32>
  "test.query"(%value) {test.expected_origins = ["cast_insert"], test.label = "tensor_cast"} : (i32) -> ()
  return
}

// CHECK: tensor_cast = [cast_insert]

// An unrealized cast does not define how tensor elements correspond, so its
// tensor result remains an unresolved origin.
func.func @unrealized_tensor_cast() {
  %zero = arith.constant 0 : index
  %base = "test.tensor_source"() {test.label = "unrealized_tensor_base"} : () -> tensor<2xi32>
  %inserted = "test.source"() {test.label = "unrealized_tensor_insert"} : () -> i32
  %updated = tensor.insert %inserted into %base[%zero] : tensor<2xi32>
  %cast = "builtin.unrealized_conversion_cast"(%updated) {test.label = "unrealized_tensor_cast"} : (tensor<2xi32>) -> tensor<?xi32>
  %value = tensor.extract %cast[%zero] : tensor<?xi32>
  "test.query"(%value) {test.expected_origins = ["unrealized_tensor_cast"], test.label = "unrealized_tensor"} : (i32) -> ()
  return
}

// CHECK: unrealized_tensor = [unrealized_tensor_cast]

// Unresolved index equality retains both possible element definitions.
func.func @dynamic_index(%read_index: index, %write_index: index) {
  %base = "test.tensor_source"() {test.label = "dynamic_base"} : () -> tensor<?xi32>
  %inserted = "test.source"() {test.label = "dynamic_insert"} : () -> i32
  %updated = tensor.insert %inserted into %base[%write_index] : tensor<?xi32>
  %value = tensor.extract %updated[%read_index] : tensor<?xi32>
  "test.query"(%value) {test.expected_origins = ["dynamic_base", "dynamic_insert"], test.label = "dynamic_index"} : (i32) -> ()
  return
}

// CHECK: dynamic_index = [dynamic_base, dynamic_insert]

// Reusing one dynamic SSA index for a direct insertion and extraction proves
// that the inserted scalar defines the extracted element.
func.func @same_dynamic_index(%index: index) {
  %base = "test.tensor_source"() {test.label = "same_dynamic_base"} : () -> tensor<?xi32>
  %inserted = "test.source"() {test.label = "same_dynamic_insert"} : () -> i32
  %updated = tensor.insert %inserted into %base[%index] : tensor<?xi32>
  %value = tensor.extract %updated[%index] : tensor<?xi32>
  "test.query"(%value) {test.expected_origins = ["same_dynamic_insert"], test.label = "same_dynamic"} : (i32) -> ()
  return
}

// CHECK: same_dynamic = [same_dynamic_insert]

// Equal finite writer and reader domains prove every extracted element was
// written by the loop-carried tensor update.
func.func @complete_loop_coverage() {
  %zero = arith.constant 0 : index
  %four = arith.constant 4 : index
  %one = arith.constant 1 : index
  %base = "test.tensor_source"() {test.label = "complete_base"} : () -> tensor<4xi32>
  %updated = scf.for %write_index = %zero to %four step %one iter_args(%tensor = %base) -> tensor<4xi32> {
    %inserted = "test.source"() {test.label = "complete_insert"} : () -> i32
    %next = tensor.insert %inserted into %tensor[%write_index] : tensor<4xi32>
    scf.yield %next : tensor<4xi32>
  }
  scf.for %read_index = %zero to %four step %one {
    %value = tensor.extract %updated[%read_index] : tensor<4xi32>
    "test.query"(%value) {test.expected_origins = ["complete_insert"], test.label = "complete_loop"} : (i32) -> ()
  }
  return
}

// CHECK: complete_loop = [complete_insert]

// Partial writer coverage retains both the inserted scalar and initial tensor.
func.func @partial_loop_coverage() {
  %zero = arith.constant 0 : index
  %two = arith.constant 2 : index
  %four = arith.constant 4 : index
  %one = arith.constant 1 : index
  %base = "test.tensor_source"() {test.label = "partial_base"} : () -> tensor<4xi32>
  %updated = scf.for %write_index = %zero to %two step %one iter_args(%tensor = %base) -> tensor<4xi32> {
    %inserted = "test.source"() {test.label = "partial_insert"} : () -> i32
    %next = tensor.insert %inserted into %tensor[%write_index] : tensor<4xi32>
    scf.yield %next : tensor<4xi32>
  }
  scf.for %read_index = %zero to %four step %one {
    %value = tensor.extract %updated[%read_index] : tensor<4xi32>
    "test.query"(%value) {test.expected_origins = ["partial_base", "partial_insert"], test.label = "partial_loop"} : (i32) -> ()
  }
  return
}

// CHECK: partial_loop = [partial_base, partial_insert]

// A zero-trip writer loop leaves the initial tensor as the only origin.
func.func @zero_trip_writer() {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %base = "test.tensor_source"() {test.label = "zero_base"} : () -> tensor<1xi32>
  %updated = scf.for %write_index = %zero to %zero step %one iter_args(%tensor = %base) -> tensor<1xi32> {
    %inserted = "test.source"() {test.label = "zero_insert"} : () -> i32
    %next = tensor.insert %inserted into %tensor[%write_index] : tensor<1xi32>
    scf.yield %next : tensor<1xi32>
  }
  %value = tensor.extract %updated[%zero] : tensor<1xi32>
  "test.query"(%value) {test.expected_origins = ["zero_base"], test.label = "zero_trip"} : (i32) -> ()
  return
}

// CHECK: zero_trip = [zero_base]

// Equal constant indices do not imply a write when the loop has zero trips.
func.func @zero_trip_equal_indices() {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %base = "test.tensor_source"() {test.label = "zero_equal_base"} : () -> tensor<1xi32>
  %updated = scf.for %write_index = %zero to %zero step %one iter_args(%tensor = %base) -> tensor<1xi32> {
    %inserted = "test.source"() {test.label = "zero_equal_insert"} : () -> i32
    %next = tensor.insert %inserted into %tensor[%zero] : tensor<1xi32>
    scf.yield %next : tensor<1xi32>
  }
  %value = tensor.extract %updated[%zero] : tensor<1xi32>
  "test.query"(%value) {test.expected_origins = ["zero_equal_base"], test.label = "zero_equal"} : (i32) -> ()
  return
}

// CHECK: zero_equal = [zero_equal_base]

// Multiple writes in one recurrence retain every possible final definition.
func.func @multiple_writes_in_recurrence() {
  %zero = arith.constant 0 : index
  %four = arith.constant 4 : index
  %one = arith.constant 1 : index
  %base = "test.tensor_source"() {test.label = "multiple_base"} : () -> tensor<4xi32>
  %updated = scf.for %index = %zero to %four step %one iter_args(%tensor = %base) -> tensor<4xi32> {
    %fixed = "test.source"() {test.label = "multiple_fixed"} : () -> i32
    %fixed_update = tensor.insert %fixed into %tensor[%zero] : tensor<4xi32>
    %varying = "test.source"() {test.label = "multiple_varying"} : () -> i32
    %varying_update = tensor.insert %varying into %fixed_update[%index] : tensor<4xi32>
    scf.yield %varying_update : tensor<4xi32>
  }
  %value = tensor.extract %updated[%zero] : tensor<4xi32>
  "test.query"(%value) {test.expected_origins = ["multiple_fixed", "multiple_varying"], test.label = "multiple_writes"} : (i32) -> ()
  return
}

// CHECK: multiple_writes = [multiple_fixed, multiple_varying]

// A loop-body block argument has no dynamic origin when the loop is known not
// to execute.
func.func @zero_trip_body_argument() {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %initial = "test.source"() {test.label = "zero_body_initial"} : () -> i32
  %result = scf.for %index = %zero to %zero step %one iter_args(%current = %initial) -> i32 {
    "test.query"(%current) {test.expected_origins = [], test.label = "zero_body_argument"} : (i32) -> ()
    scf.yield %current : i32
  }
  return
}

// CHECK: zero_body_argument = []

// A statically empty nested loop cannot establish aggregate coverage for an
// enclosing recurrence.
func.func @nested_zero_trip_write() {
  %zero = arith.constant 0 : index
  %four = arith.constant 4 : index
  %one = arith.constant 1 : index
  %base = "test.tensor_source"() {test.label = "nested_zero_base"} : () -> tensor<4xi32>
  %updated = scf.for %outer = %zero to %four step %one iter_args(%outer_tensor = %base) -> tensor<4xi32> {
    %inner_result = scf.for %inner = %zero to %zero step %one iter_args(%inner_tensor = %outer_tensor) -> tensor<4xi32> {
      %inserted = "test.source"() {test.label = "nested_zero_insert"} : () -> i32
      %next = tensor.insert %inserted into %inner_tensor[%outer] : tensor<4xi32>
      scf.yield %next : tensor<4xi32>
    }
    scf.yield %inner_result : tensor<4xi32>
  }
  %value = tensor.extract %updated[%zero] : tensor<4xi32>
  "test.query"(%value) {test.expected_origins = ["nested_zero_base"], test.label = "nested_zero"} : (i32) -> ()
  return
}

// CHECK: nested_zero = [nested_zero_base]

// Exhausting the enumeration limit conservatively retains the inserted value
// and the tensor's previous contents.
func.func @enumeration_limit() {
  %zero = arith.constant 0 : index
  %four = arith.constant 4 : index
  %one = arith.constant 1 : index
  %base = "test.tensor_source"() {test.label = "limit_base"} : () -> tensor<4xi32>
  %updated = scf.for %write_index = %zero to %four step %one iter_args(%tensor = %base) -> tensor<4xi32> {
    %write_expression = arith.addi %write_index, %zero : index
    %inserted = "test.source"() {test.label = "limit_insert"} : () -> i32
    %next = tensor.insert %inserted into %tensor[%write_expression] : tensor<4xi32>
    scf.yield %next : tensor<4xi32>
  }
  %value = tensor.extract %updated[%zero] : tensor<4xi32>
  "test.query"(%value) {test.expected_origins = ["limit_base", "limit_insert"], test.label = "limit", test.max_enumerated_index_tuples = 1 : i64} : (i32) -> ()
  return
}

// CHECK: limit = [limit_base, limit_insert]

// Exhausting the loop-iteration limit makes an otherwise disjoint access
// relation unknown.
func.func @loop_iteration_limit() {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %base = "test.tensor_source"() {test.label = "iteration_limit_base"} : () -> tensor<2xi32>
  scf.for %index = %zero to %two step %one {
    %opposite_index = arith.subi %one, %index : index
    %inserted = "test.source"() {test.label = "iteration_limit_insert"} : () -> i32
    %updated = tensor.insert %inserted into %base[%index] : tensor<2xi32>
    %value = tensor.extract %updated[%opposite_index] : tensor<2xi32>
    "test.query"(%value) {test.expected_origins = ["iteration_limit_base", "iteration_limit_insert"], test.label = "iteration_limit", test.max_enumerated_loop_iterations = 1 : i64} : (i32) -> ()
  }
  return
}

// CHECK: iteration_limit = [iteration_limit_base, iteration_limit_insert]

// Shared loop induction variables preserve pointwise correlation instead of
// comparing only the aggregate sets of index values.
func.func @correlated_indices() {
  %zero = arith.constant 0 : index
  %four = arith.constant 4 : index
  %one = arith.constant 1 : index
  %three = arith.constant 3 : index
  %base = "test.tensor_source"() {test.label = "correlated_base"} : () -> tensor<4xi32>
  scf.for %index = %zero to %four step %one {
    %reverse = arith.subi %three, %index : index
    %inserted = "test.source"() {test.label = "correlated_insert"} : () -> i32
    %updated = tensor.insert %inserted into %base[%index] : tensor<4xi32>
    %value = tensor.extract %updated[%reverse] : tensor<4xi32>
    "test.query"(%value) {test.expected_origins = ["correlated_base"], test.label = "correlated"} : (i32) -> ()
  }
  return
}

// CHECK: correlated = [correlated_base]

// A multi-dimensional LoopLike operation enumerates all induction-variable
// assignments while preserving pointwise index correlation.
func.func @multi_iv_loop() {
  %one = arith.constant 1 : index
  %base = "test.tensor_source"() {test.label = "multi_iv_base"} : () -> tensor<2x2xi32>
  scf.forall (%row, %column) in (2, 2) {
    %reverse_column = arith.subi %one, %column : index
    %inserted = "test.source"() {test.label = "multi_iv_insert"} : () -> i32
    %updated = tensor.insert %inserted into %base[%row, %column] : tensor<2x2xi32>
    %value = tensor.extract %updated[%row, %reverse_column] : tensor<2x2xi32>
    "test.query"(%value) {test.expected_origins = ["multi_iv_base"], test.label = "multi_iv"} : (i32) -> ()
  }
  return
}

// CHECK: multi_iv = [multi_iv_base]

// A read may observe an insertion from a prior loop iteration even when the
// current iteration writes a different index.
func.func @prior_iteration_index() {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %three = arith.constant 3 : index
  %base = "test.tensor_source"() {test.label = "prior_base"} : () -> tensor<4xi32>
  scf.for %index = %one to %three step %one iter_args(%tensor = %base) -> tensor<4xi32> {
    %inserted = "test.source"() {test.label = "prior_insert"} : () -> i32
    %updated = tensor.insert %inserted into %tensor[%index] : tensor<4xi32>
    %previous_index = arith.subi %index, %one : index
    %value = tensor.extract %updated[%previous_index] : tensor<4xi32>
    "test.query"(%value) {test.expected_origins = ["prior_base", "prior_insert"], test.label = "prior_iteration"} : (i32) -> ()
    scf.yield %updated : tensor<4xi32>
  }
  return
}

// CHECK: prior_iteration = [prior_base, prior_insert]

// A prior-iteration read retains every update in the recurrence because
// aggregate index equality does not prove temporal availability.
func.func @prior_iteration_multiple_updates() {
  %zero = arith.constant 0 : index
  %two = arith.constant 2 : index
  %one = arith.constant 1 : index
  %base = "test.tensor_source"() {test.label = "temporal_base"} : () -> tensor<4xi32>
  scf.for %index = %zero to %two step %one iter_args(%tensor = %base) -> tensor<4xi32> {
    %value = tensor.extract %tensor[%index] : tensor<4xi32>
    "test.query"(%value) {test.expected_origins = ["temporal_base", "temporal_first", "temporal_second"], test.label = "temporal_order"} : (i32) -> ()
    %next_index = arith.addi %index, %one : index
    %first = "test.source"() {test.label = "temporal_first"} : () -> i32
    %first_update = tensor.insert %first into %tensor[%next_index] : tensor<4xi32>
    %second = "test.source"() {test.label = "temporal_second"} : () -> i32
    %second_update = tensor.insert %second into %first_update[%index] : tensor<4xi32>
    scf.yield %second_update : tensor<4xi32>
  }
  return
}

// CHECK: temporal_order = [temporal_base, temporal_first, temporal_second]
