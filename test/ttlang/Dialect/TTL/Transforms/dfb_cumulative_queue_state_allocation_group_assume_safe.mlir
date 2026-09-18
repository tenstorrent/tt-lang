// Tests unsafe allocation-group handoff with independent DFB ring cursors.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true unsafe-assume-allocation-groups=true})' 2>%t.warning | FileCheck %s
// RUN: FileCheck %s --check-prefix=WARNING < %t.warning

// The first member leaves both cursors at offset eight in a 12-tile physical
// DFB. The second member's eight-tile write would cross the ring boundary,
// although its two four-tile reads would not. Unsafe policy therefore requires
// a declared epoch reset before the second member.

// CHECK: module attributes {ttl.assumed_dfb_allocation_groups = [{allocation_group = #ttl.dfb_allocation_group<0>, assumptions = [{lhs = 1 : i64, reason = "epoch-reset"}], members = [0, 1]}]
// CHECK-LABEL: func.func @cumulative_cursor_epoch_reset
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 3} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 0, block_count = 3} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}

// WARNING: warning: unsafe DFB allocation-group policy accepted #ttl.dfb_allocation_group<0> members=[0, 1] without compiler proof: epoch-reset(1)

module {
  func.func @cumulative_cursor_epoch_reset()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 3>
    %second = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
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
