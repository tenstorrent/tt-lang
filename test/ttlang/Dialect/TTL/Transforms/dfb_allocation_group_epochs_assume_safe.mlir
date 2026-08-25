// Tests unsafe handoff between interleaved allocation-group epochs.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true unsafe-assume-allocation-groups=true})' 2>%t.warning | FileCheck %s
// RUN: FileCheck %s --check-prefix=WARNING < %t.warning

// A0 resets at offset one. B0 then leaves offset two in the three-tile
// envelope, so A1's two-tile transaction requires an assumed epoch reset.

// CHECK: module attributes {ttl.assumed_dfb_allocation_groups = [{allocation_group = #ttl.dfb_allocation_group<0>, assumptions = [{lhs = 0 : i64, reason = "epoch-reset"}], members = [0, 1]}]
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 3} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
// WARNING: warning: unsafe DFB allocation-group policy accepted #ttl.dfb_allocation_group<0> members=[0, 1] without compiler proof: epoch-reset(0)

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @interleaved_epoch_reset()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "interleaved_epoch_reset">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %a = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %b = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.opaque_call "a0"
        dfb_dependencies(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                     #ttl.dfb_protocol_effect<push, 0, 1>,
                     #ttl.dfb_protocol_effect<wait, 0, 1>,
                     #ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "effects.hpp"} : () -> ()
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "interleaved_epoch_reset">, <kind = data_movement, identity = "reader", operation = "interleaved_epoch_reset">, <kind = data_movement, identity = "writer", operation = "interleaved_epoch_reset">]>(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    ttl.opaque_call "b0"
        dfb_dependencies(%b : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 2>,
                     #ttl.dfb_protocol_effect<push, 0, 2>,
                     #ttl.dfb_protocol_effect<wait, 0, 2>,
                     #ttl.dfb_protocol_effect<pop, 0, 2>]
        () {header = "effects.hpp"} : () -> ()
    ttl.opaque_call "a1"
        dfb_dependencies(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 2>,
                     #ttl.dfb_protocol_effect<push, 0, 2>,
                     #ttl.dfb_protocol_effect<wait, 0, 2>,
                     #ttl.dfb_protocol_effect<pop, 0, 2>]
        () {header = "effects.hpp"} : () -> ()
    return
  }

  func.func @reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "interleaved_epoch_reset">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %a = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "interleaved_epoch_reset">, <kind = data_movement, identity = "reader", operation = "interleaved_epoch_reset">, <kind = data_movement, identity = "writer", operation = "interleaved_epoch_reset">]>(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }

  func.func @writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "interleaved_epoch_reset">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %a = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "interleaved_epoch_reset">, <kind = data_movement, identity = "reader", operation = "interleaved_epoch_reset">, <kind = data_movement, identity = "writer", operation = "interleaved_epoch_reset">]>(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }
}
