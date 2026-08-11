// Tests physical DFB reuse for balanced runtime-conditional lifecycles.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices 2>&1 | FileCheck %s --check-prefix=REPORT

// REPORT: DFB logical_id=0 bounded=0 compiler_created=0 conditionally_bounded=1
// REPORT-SAME: domain=unknown
// REPORT: diagnostic_nodes quiescence=none domain_assumption=unknown-possible may_be_active=1 conditional_execution=1 node_count=1 nodes={(0,0)}
// REPORT: diagnostic_nodes quiescence=none domain_assumption=unknown-possible may_be_active=1 conditional_execution=1 node_count=1 nodes={(1,0)}
// REPORT: DFB logical_id=1 bounded=0 compiler_created=0 conditionally_bounded=1
// REPORT-SAME: domain=unknown
// REPORT: DFB assignment: logical DFB 0 -> physical index 0 (bounded)
// REPORT-NEXT: DFB assignment: logical DFB 1 -> physical index 0 (bounded)

// Separate regions controlled by one opaque predicate preserve one 0-or-1
// transaction for each DFB and permit sequential physical-index reuse.
// CHECK-LABEL: func.func @same_ssa_condition
// CHECK-SAME: ttl.base_cta_index = 1 : i32
// CHECK: %[[FIRST:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: %[[SECOND:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @same_ssa_condition()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %condition = ttl.opaque_call "predicate" () {header = "predicate.hpp"} : () -> i1
    scf.if %condition {
      ttl.opaque_call "first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    scf.if %condition {
      ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    return
  }
}

// -----

// A dynamic launch-node condition retains possible membership on every node.
// Complete lifecycles and ordering on every possible node permit reuse.
// CHECK-LABEL: func.func @unknown_domain_same_ssa_condition
// CHECK-SAME: ttl.base_cta_index = 1 : i32
// CHECK: %[[FIRST:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: %[[SECOND:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 1 : index}

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unknown_domain_same_ssa_condition(%offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %sum = arith.addi %core_x, %offset : index
    %c0 = arith.constant 0 : index
    %condition = arith.cmpi eq, %sum, %c0 : index
    scf.if %condition {
      ttl.opaque_call "first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    scf.if %condition {
      ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    return
  }
}

// -----

// Separately evaluated opaque predicates do not prove that one DFB's producer
// and consumer effects execute together.
// CHECK-LABEL: func.func @recomputed_condition
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @recomputed_condition()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %producer_condition = ttl.opaque_call "predicate" () {header = "predicate.hpp"} : () -> i1
    %consumer_condition = ttl.opaque_call "predicate" () {header = "predicate.hpp"} : () -> i1
    scf.if %producer_condition {
      ttl.opaque_call "produce_first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    scf.if %consumer_condition {
      ttl.opaque_call "consume_first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    scf.if %producer_condition {
      ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    return
  }
}

// -----

// The conditional proof does not compare an unknown launch domain with an
// exact launch domain.
// CHECK-LABEL: func.func @unknown_and_exact_domains
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unknown_and_exact_domains(%offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %conditional = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %exact = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %sum = arith.addi %core_x, %offset : index
    %c0 = arith.constant 0 : index
    %condition = arith.cmpi eq, %sum, %c0 : index
    scf.if %condition {
      ttl.opaque_call "conditional" dfb_dependencies(%conditional : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    ttl.opaque_call "exact" dfb_dependencies(%exact : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// Opposite branch polarity does not prove that producer and consumer effects
// execute together.
// CHECK-LABEL: func.func @opposite_branch_polarity
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @opposite_branch_polarity()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %condition = ttl.opaque_call "predicate" () {header = "predicate.hpp"} : () -> i1
    scf.if %condition {
      ttl.opaque_call "produce_first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>] () {header = "effects.hpp"} : () -> ()
    } else {
      ttl.opaque_call "consume_first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    scf.if %condition {
      ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    return
  }
}

// -----

// A complete conditional transaction is quiescent before a later exact-count
// transaction whether its condition is false or true.
// CHECK-LABEL: func.func @conditional_and_exact_domains
// CHECK-SAME: ttl.base_cta_index = 1 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @conditional_and_exact_domains()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %conditional = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %exact = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %condition = ttl.opaque_call "predicate" () {header = "predicate.hpp"} : () -> i1
    scf.if %condition {
      ttl.opaque_call "conditional" dfb_dependencies(%conditional : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    ttl.opaque_call "exact" dfb_dependencies(%exact : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// Producer and consumer effects with different nested conditions do not form
// one 0-or-1 transaction.
// CHECK-LABEL: func.func @different_nested_conditions
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @different_nested_conditions(%outer: i1, %inner: i1)
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    scf.if %outer {
      scf.if %inner {
        ttl.opaque_call "produce_first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>] () {header = "effects.hpp"} : () -> ()
      }
      ttl.opaque_call "consume_first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    scf.if %outer {
      ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    return
  }
}

// -----

// An opaque DFB access controlled by a separately evaluated condition is not
// covered by the transaction's conditional completion frontier.
// CHECK-LABEL: func.func @mismatched_opaque_access_condition
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @mismatched_opaque_access_condition()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %transaction_condition = ttl.opaque_call "predicate" () {header = "predicate.hpp"} : () -> i1
    %access_condition = ttl.opaque_call "predicate" () {header = "predicate.hpp"} : () -> i1
    scf.if %transaction_condition {
      ttl.opaque_call "first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    scf.if %access_condition {
      ttl.opaque_call "use_first" (%first) {header = "effects.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
    }
    scf.if %transaction_condition {
      ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    return
  }
}

// -----

// Equal unresolved loop bounds do not prove a single 0-or-1 transaction.
// CHECK-LABEL: func.func @unresolved_loop_count
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @unresolved_loop_count(%upper_bound: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %iteration = %c0 to %upper_bound step %c1 {
      ttl.opaque_call "first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    scf.for %iteration = %c0 to %upper_bound step %c1 {
      ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    return
  }
}

// -----

// Multiple runtime-conditional transactions remain unsupported.
// CHECK-LABEL: func.func @repeated_conditional_transactions
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @repeated_conditional_transactions()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %condition = ttl.opaque_call "predicate" () {header = "predicate.hpp"} : () -> i1
    scf.if %condition {
      ttl.opaque_call "first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>, #ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    scf.if %condition {
      ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    return
  }
}
