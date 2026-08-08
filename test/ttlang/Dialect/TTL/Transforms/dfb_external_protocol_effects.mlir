// Verify physical-index reuse from complete external protocol summaries and
// conservative allocation for incomplete, mismatched, or unknown access.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s

// Ordered repeated transactions with matching tile counts prove that both
// sequential DFB lifetimes are bounded and permit index reuse.
// CHECK-LABEL: func.func @complete_effects
// CHECK-SAME: ttl.base_cta_index = 1 : i32
// CHECK: %[[FIRST:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: %[[SECOND:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 1 : index}
// CHECK-NEXT: ttl.opaque_call "first" dfb_dependencies(%[[FIRST]] : !ttl.cb<{{.*}}>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>, #ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] ()
// CHECK-NEXT: ttl.opaque_call "second" dfb_dependencies(%[[SECOND]] : !ttl.cb<{{.*}}>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>, #ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] ()

module {
  func.func @complete_effects() attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    ttl.opaque_call "first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>, #ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>, #ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// Concrete production followed by a complete external consumer summary forms
// a bounded lifecycle and permits sequential physical-index reuse.
// CHECK-LABEL: func.func @concrete_producer_external_consumer
// CHECK-SAME: ttl.base_cta_index = 1 : i32
// CHECK: %[[FIRST:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: %[[SECOND:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @concrete_producer_external_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %first_slot = ttl.cb_reserve %first {num_tiles = 1 : i64} : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %first {num_tiles = 1 : i64} : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    ttl.opaque_call "drain_first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    %second_slot = ttl.cb_reserve %second {num_tiles = 1 : i64} : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %second {num_tiles = 1 : i64} : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    ttl.opaque_call "drain_second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// An omitted pop leaves the first lifecycle incomplete and prevents reuse.
// CHECK-LABEL: func.func @incomplete_effects
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @incomplete_effects() attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    ttl.opaque_call "first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>] () {header = "effects.hpp"} : () -> ()
    ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// Mismatched transaction counts cannot prove a bounded lifecycle.
// CHECK-LABEL: func.func @mismatched_effects
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @mismatched_effects() attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    ttl.opaque_call "first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 2>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// Complete lifecycles with different transaction sizes cannot share a
// physical index because their pointer progressions differ.
// CHECK-LABEL: func.func @different_transaction_sizes
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @different_transaction_sizes() attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    ttl.opaque_call "first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 2>, #ttl.dfb_protocol_effect<push, 0, 2>, #ttl.dfb_protocol_effect<wait, 0, 2>, #ttl.dfb_protocol_effect<pop, 0, 2>] () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// An effect on one aliased dependency occurrence does not summarize another
// occurrence, so the omitted occurrence keeps the first lifecycle unbounded.
// CHECK-LABEL: func.func @aliased_dependency_occurrence
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @aliased_dependency_occurrence() attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    ttl.opaque_call "first" dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] (%first, %first) {header = "effects.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>, !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
    ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// Unknown access overlaps every user-managed DFB in the allocation scope and
// disables reuse even when the listed summaries are complete.
// CHECK-LABEL: func.func @unknown_access
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @unknown_access() attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    ttl.opaque_call "first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp", unknown_dfb_access} : () -> ()
    ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }
}
