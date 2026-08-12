// Tests typed dispatch-condition identity in conditional DFB lifecycles.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s

// Separately evaluated conditions in producer and consumer logical kernels
// prove two complete lifecycles. The acknowledgment orders the second source
// after the first source reaches terminal completion.
// CHECK-LABEL: func.func @cross_kernel_producer
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 2 : index}
// CHECK-LABEL: func.func @cross_kernel_consumer
// CHECK-SAME: ttl.base_cta_index = 2 : i32

module {
  func.func @cross_kernel_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 3 : i32,
                  ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %acknowledgment = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %zero = arith.constant 0 : i64
    %first_value = ttl.opaque_call "predicate" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "predicate.hpp"} : () -> i64
    %first_active = arith.cmpi ne, %first_value, %zero : i64
    scf.if %first_active {
      %slot = ttl.cb_reserve %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    %ack = ttl.cb_wait %acknowledgment : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %acknowledgment : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_value = ttl.opaque_call "predicate" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "predicate.hpp"} : () -> i64
    %second_active = arith.cmpi ne, %second_value, %zero : i64
    scf.if %second_active {
      %slot = ttl.cb_reserve %second : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    return
  }

  func.func @cross_kernel_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %acknowledgment = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %zero = arith.constant 0 : i64
    %first_value = ttl.opaque_call "predicate" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "predicate.hpp"} : () -> i64
    %first_active = arith.cmpi ne, %first_value, %zero : i64
    scf.if %first_active {
      %slot = ttl.cb_wait %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    %ack = ttl.cb_reserve %acknowledgment : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %acknowledgment : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_value = ttl.opaque_call "predicate" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "predicate.hpp"} : () -> i64
    %second_active = arith.cmpi ne, %second_value, %zero : i64
    scf.if %second_active {
      %slot = ttl.cb_wait %second : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %second : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    return
  }
}

// -----

// Different typed identities do not establish equal execution.
// CHECK-LABEL: func.func @different_identities
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @different_identities()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %zero = arith.constant 0 : i64
    %producer_value = ttl.opaque_call "predicate" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "predicate.hpp"} : () -> i64
    %consumer_value = ttl.opaque_call "predicate" () {condition_result = #ttl.dispatch_condition<1, i64>, header = "predicate.hpp"} : () -> i64
    %producer_active = arith.cmpi ne, %producer_value, %zero : i64
    %consumer_active = arith.cmpi ne, %consumer_value, %zero : i64
    scf.if %producer_active {
      ttl.opaque_call "produce" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    scf.if %consumer_active {
      ttl.opaque_call "consume" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// Equal identity with opposite truth polarity remains conservative.
// CHECK-LABEL: func.func @opposite_polarity
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @opposite_polarity()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %zero = arith.constant 0 : i64
    %producer_value = ttl.opaque_call "producer_predicate" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "predicate.hpp"} : () -> i64
    %consumer_value = ttl.opaque_call "consumer_predicate" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "predicate.hpp"} : () -> i64
    %producer_active = arith.cmpi ne, %producer_value, %zero : i64
    %consumer_active = arith.cmpi eq, %consumer_value, %zero : i64
    scf.if %producer_active {
      ttl.opaque_call "produce" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    scf.if %consumer_active {
      ttl.opaque_call "consume" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// Equal independently evaluated identities at every nesting level prove the
// same structured execution condition.
// CHECK-LABEL: func.func @matching_nested_identities
// CHECK-SAME: ttl.base_cta_index = 1 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @matching_nested_identities()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %zero = arith.constant 0 : i64
    %producer_outer_value = ttl.opaque_call "outer" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "predicate.hpp"} : () -> i64
    %producer_inner_value = ttl.opaque_call "inner" () {condition_result = #ttl.dispatch_condition<1, i64>, header = "predicate.hpp"} : () -> i64
    %producer_outer = arith.cmpi ne, %producer_outer_value, %zero : i64
    %producer_inner = arith.cmpi ne, %producer_inner_value, %zero : i64
    scf.if %producer_outer {
      scf.if %producer_inner {
        ttl.opaque_call "produce" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>] () {header = "effects.hpp"} : () -> ()
      }
    }
    %consumer_outer_value = ttl.opaque_call "outer_again" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "other.hpp"} : () -> i64
    %consumer_inner_value = ttl.opaque_call "inner_again" () {condition_result = #ttl.dispatch_condition<1, i64>, header = "other.hpp"} : () -> i64
    %consumer_outer = arith.cmpi ne, %consumer_outer_value, %zero : i64
    %consumer_inner = arith.cmpi ne, %consumer_inner_value, %zero : i64
    scf.if %consumer_outer {
      scf.if %consumer_inner {
        ttl.opaque_call "consume" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
      }
    }
    ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// Missing identity in one nested frame prevents a partial proof.
// CHECK-LABEL: func.func @partial_nested_identity
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @partial_nested_identity()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %zero = arith.constant 0 : i64
    %producer_outer_value = ttl.opaque_call "outer" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "predicate.hpp"} : () -> i64
    %producer_inner_value = ttl.opaque_call "inner" () {condition_result = #ttl.dispatch_condition<1, i64>, header = "predicate.hpp"} : () -> i64
    %producer_outer = arith.cmpi ne, %producer_outer_value, %zero : i64
    %producer_inner = arith.cmpi ne, %producer_inner_value, %zero : i64
    scf.if %producer_outer {
      scf.if %producer_inner {
        ttl.opaque_call "produce" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>] () {header = "effects.hpp"} : () -> ()
      }
    }
    %consumer_outer_value = ttl.opaque_call "outer" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "predicate.hpp"} : () -> i64
    %consumer_inner_value = ttl.opaque_call "inner" () {header = "predicate.hpp"} : () -> i64
    %consumer_outer = arith.cmpi ne, %consumer_outer_value, %zero : i64
    %consumer_inner = arith.cmpi ne, %consumer_inner_value, %zero : i64
    scf.if %consumer_outer {
      scf.if %consumer_inner {
        ttl.opaque_call "consume" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
      }
    }
    ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// Matching structural boolean expressions preserve both leaf identities.
// CHECK-LABEL: func.func @matching_boolean_expressions
// CHECK-SAME: ttl.base_cta_index = 1 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @matching_boolean_expressions()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %zero = arith.constant 0 : i64
    %producer_lhs_value = ttl.opaque_call "lhs" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "predicate.hpp"} : () -> i64
    %producer_rhs_value = ttl.opaque_call "rhs" () {condition_result = #ttl.dispatch_condition<1, i64>, header = "predicate.hpp"} : () -> i64
    %producer_lhs = arith.cmpi ne, %producer_lhs_value, %zero : i64
    %producer_rhs = arith.cmpi ne, %producer_rhs_value, %zero : i64
    %producer_active = arith.andi %producer_lhs, %producer_rhs : i1
    scf.if %producer_active {
      ttl.opaque_call "produce" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    %consumer_lhs_value = ttl.opaque_call "lhs_again" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "predicate.hpp"} : () -> i64
    %consumer_rhs_value = ttl.opaque_call "rhs_again" () {condition_result = #ttl.dispatch_condition<1, i64>, header = "predicate.hpp"} : () -> i64
    %consumer_lhs = arith.cmpi ne, %consumer_lhs_value, %zero : i64
    %consumer_rhs = arith.cmpi ne, %consumer_rhs_value, %zero : i64
    %consumer_active = arith.andi %consumer_lhs, %consumer_rhs : i1
    scf.if %consumer_active {
      ttl.opaque_call "consume" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// Dispatch identity preserves truth, not the particular nonzero carrier value.
// Bitwise arithmetic on an i64 result therefore cannot establish equality.
// CHECK-LABEL: func.func @integer_bitwise_expression
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @integer_bitwise_expression()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %zero = arith.constant 0 : i64
    %mask = arith.constant 1 : i64
    %producer_value = ttl.opaque_call "producer_predicate" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "predicate.hpp"} : () -> i64
    %consumer_value = ttl.opaque_call "consumer_predicate" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "predicate.hpp"} : () -> i64
    %producer_masked = arith.andi %producer_value, %mask : i64
    %consumer_masked = arith.andi %consumer_value, %mask : i64
    %producer_active = arith.cmpi ne, %producer_masked, %zero : i64
    %consumer_active = arith.cmpi ne, %consumer_masked, %zero : i64
    scf.if %producer_active {
      ttl.opaque_call "produce" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    scf.if %consumer_active {
      ttl.opaque_call "consume" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }
}
