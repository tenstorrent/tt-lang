// Tests conservative rejection of unsupported cumulative DFB queue state.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s

// CHECK: lifecycle_completion=mismatched-transaction {{.*}} kernel=@total_mismatch

module {
  func.func @total_mismatch()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>
    %second = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>
    ttl.opaque_call "first"
        dfb_dependencies(%first : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 8>,
                     #ttl.dfb_protocol_effect<push, 0, 8>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>]
        () {header = "effects.hpp"} : () -> ()
    ttl.opaque_call "second"
        dfb_dependencies(%second : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 4>,
                     #ttl.dfb_protocol_effect<push, 0, 4>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>]
        () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// CHECK: lifecycle_completion=mismatched-transaction {{.*}} kernel=@pop_exceeds_wait

module {
  func.func @pop_exceeds_wait()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "first"
        dfb_dependencies(%first : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 8>,
                     #ttl.dfb_protocol_effect<push, 0, 8>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 8>]
        () {header = "effects.hpp"} : () -> ()
    ttl.opaque_call "second"
        dfb_dependencies(%second : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 4>,
                     #ttl.dfb_protocol_effect<push, 0, 4>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>]
        () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// CHECK: lifecycle_completion=mismatched-transaction {{.*}} kernel=@push_exceeds_reserve

module {
  func.func @push_exceeds_reserve()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "first"
        dfb_dependencies(%first : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 4>,
                     #ttl.dfb_protocol_effect<push, 0, 8>,
                     #ttl.dfb_protocol_effect<wait, 0, 8>,
                     #ttl.dfb_protocol_effect<pop, 0, 8>]
        () {header = "effects.hpp"} : () -> ()
    ttl.opaque_call "second"
        dfb_dependencies(%second : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 4>,
                     #ttl.dfb_protocol_effect<push, 0, 4>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>]
        () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// CHECK: lifecycle_completion=incomplete-use-order {{.*}} kernel=@unpublished_wait

module {
  func.func @unpublished_wait()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>
    %second = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>
    ttl.opaque_call "producer"
        dfb_dependencies(%first : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 4>,
                     #ttl.dfb_protocol_effect<push, 0, 4>]
        () {header = "effects.hpp"} : () -> ()
    ttl.opaque_call "consumer"
        dfb_dependencies(%first : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 8>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>]
        () {header = "effects.hpp"} : () -> ()
    ttl.opaque_call "second"
        dfb_dependencies(%second : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 4>,
                     #ttl.dfb_protocol_effect<push, 0, 4>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>]
        () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// Waiting for the second publication before the pop that enables its reserve
// would introduce a cycle, so no synchronization relation is proved.

// CHECK: lifecycle_completion=incomplete-use-order {{.*}} kernel=@cyclic_capacity_producer

module {
  func.func @cyclic_capacity_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "producer"
        dfb_dependencies(%dfb : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 8>,
                     #ttl.dfb_protocol_effect<push, 0, 4>,
                     #ttl.dfb_protocol_effect<reserve, 0, 8>,
                     #ttl.dfb_protocol_effect<push, 0, 4>]
        () {header = "effects.hpp"} : () -> ()
    return
  }

  func.func @cyclic_capacity_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 1 : i32,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "consumer"
        dfb_dependencies(%dfb : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 8>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>]
        () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// CHECK: lifecycle_completion=mismatched-transaction {{.*}} kernel=@cursor_crossing

module {
  func.func @cursor_crossing()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>
    %second = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>
    ttl.opaque_call "first"
        dfb_dependencies(%first : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 8>,
                     #ttl.dfb_protocol_effect<push, 0, 8>,
                     #ttl.dfb_protocol_effect<reserve, 0, 8>,
                     #ttl.dfb_protocol_effect<push, 0, 8>,
                     #ttl.dfb_protocol_effect<wait, 0, 8>,
                     #ttl.dfb_protocol_effect<pop, 0, 8>,
                     #ttl.dfb_protocol_effect<wait, 0, 8>,
                     #ttl.dfb_protocol_effect<pop, 0, 8>]
        () {header = "effects.hpp"} : () -> ()
    ttl.opaque_call "second"
        dfb_dependencies(%second : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 4>,
                     #ttl.dfb_protocol_effect<push, 0, 4>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>]
        () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// A cursor sequence that is legal from offset zero may cross the physical
// boundary when an automatically aliased lifecycle repeats it from offset
// eight. An explicit allocation group may prove a different complete handoff.

// CHECK: DFB logical_id=0 bounded=1
// CHECK: DFB logical_id=1 bounded=1
// CHECK: DFB conflict lhs=0 rhs=1 reason=transaction-mismatch
// CHECK: Total DFB count: 2

module {
  func.func @unsafe_automatic_cursor_repetition()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>
    %second = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>
    ttl.opaque_call "first"
        dfb_dependencies(%first : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 8>,
                     #ttl.dfb_protocol_effect<push, 0, 8>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>]
        () {header = "effects.hpp"} : () -> ()
    ttl.opaque_call "second"
        dfb_dependencies(%second : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 8>,
                     #ttl.dfb_protocol_effect<push, 0, 8>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>]
        () {header = "effects.hpp"} : () -> ()
    return
  }
}
