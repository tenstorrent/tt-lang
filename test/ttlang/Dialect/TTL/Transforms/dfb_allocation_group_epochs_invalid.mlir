// Tests rejected reset-epoch allocation-group contracts.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})'

// Without the reset between B0 and A1, B's complete lifetime overlaps A1.

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @missing_reset_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "producer", operation = "missing_reset">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %a = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %b = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.opaque_call "produce_a0" dfb_dependencies(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>] () {header = "effects.hpp"} : () -> ()
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "consumer", operation = "missing_reset">, <kind = data_movement, identity = "producer", operation = "missing_reset">, <kind = data_movement, identity = "writer", operation = "missing_reset">]>(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<0> members=[0, 1] cannot alias logical DFBs 0 and 1: concurrent-lifetime}}
    ttl.opaque_call "produce_b" dfb_dependencies(%b : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>] () {header = "effects.hpp"} : () -> ()
    ttl.opaque_call "produce_a1" dfb_dependencies(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }

  func.func @missing_reset_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "consumer", operation = "missing_reset">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %a = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %b = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.opaque_call "consume_a0" dfb_dependencies(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "consumer", operation = "missing_reset">, <kind = data_movement, identity = "producer", operation = "missing_reset">, <kind = data_movement, identity = "writer", operation = "missing_reset">]>(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    ttl.opaque_call "consume_a1" dfb_dependencies(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    ttl.opaque_call "consume_b" dfb_dependencies(%b : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }

  func.func @missing_reset_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "missing_reset">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %a = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "consumer", operation = "missing_reset">, <kind = data_movement, identity = "producer", operation = "missing_reset">, <kind = data_movement, identity = "writer", operation = "missing_reset">]>(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    return
  }
}

// -----

// A noncanonical A1 epoch cannot hand the physical index from NOC0 to NOC1.
// Resets of the control DFB establish ordering without resetting A1.

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @owner_noc0()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "noc0", operation = "owner_handoff">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 3 : i32,
                  ttl.crta_indices = []} {
    %a = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %b = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %control = ttl.bind_cb {cb_index = 2, block_count = 1}
        {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.opaque_call "a0" dfb_dependencies(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "owner_handoff">, <kind = data_movement, identity = "noc0", operation = "owner_handoff">, <kind = data_movement, identity = "noc1", operation = "owner_handoff">]>(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    ttl.opaque_call "a1" dfb_dependencies(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "owner_handoff">, <kind = data_movement, identity = "noc0", operation = "owner_handoff">, <kind = data_movement, identity = "noc1", operation = "owner_handoff">]>(%control : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    return
  }

  func.func @owner_noc1()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "noc1", operation = "owner_handoff">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 3 : i32,
                  ttl.crta_indices = []} {
    %a = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %b = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %control = ttl.bind_cb {cb_index = 2, block_count = 1}
        {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "owner_handoff">, <kind = data_movement, identity = "noc0", operation = "owner_handoff">, <kind = data_movement, identity = "noc1", operation = "owner_handoff">]>(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "owner_handoff">, <kind = data_movement, identity = "noc0", operation = "owner_handoff">, <kind = data_movement, identity = "noc1", operation = "owner_handoff">]>(%control : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<1> members=[0, 1] cannot alias logical DFBs 0 and 1: pointer-owner-mismatch}}
    ttl.opaque_call "b0" dfb_dependencies(%b : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }

  func.func @owner_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "owner_handoff">,
                  ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
    %a = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %control = ttl.bind_cb {cb_index = 2, block_count = 1}
        {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "owner_handoff">, <kind = data_movement, identity = "noc0", operation = "owner_handoff">, <kind = data_movement, identity = "noc1", operation = "owner_handoff">]>(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "owner_handoff">, <kind = data_movement, identity = "noc0", operation = "owner_handoff">, <kind = data_movement, identity = "noc1", operation = "owner_handoff">]>(%control : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    return
  }
}

// -----

// A0 resets at offset one. B0 leaves offset two in the three-tile envelope,
// so A1's two-tile transaction would straddle the physical ring boundary.

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @epoch_envelope_crossing()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "epoch_envelope_crossing">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %a = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<2>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %b = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<2>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.opaque_call "a0" dfb_dependencies(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "epoch_envelope_crossing">, <kind = data_movement, identity = "reader", operation = "epoch_envelope_crossing">, <kind = data_movement, identity = "writer", operation = "epoch_envelope_crossing">]>(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    ttl.opaque_call "b0" dfb_dependencies(%b : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 2>, #ttl.dfb_protocol_effect<push, 0, 2>, #ttl.dfb_protocol_effect<wait, 0, 2>, #ttl.dfb_protocol_effect<pop, 0, 2>] () {header = "effects.hpp"} : () -> ()
    // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<2> members=[0, 1] physical envelope of 3 tiles makes logical DFB 0 epoch 1 cross the ring boundary on launch node (0,0)}}
    ttl.opaque_call "a1" dfb_dependencies(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 2>, #ttl.dfb_protocol_effect<push, 0, 2>, #ttl.dfb_protocol_effect<wait, 0, 2>, #ttl.dfb_protocol_effect<pop, 0, 2>] () {header = "effects.hpp"} : () -> ()
    return
  }

  func.func @epoch_envelope_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "epoch_envelope_crossing">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %a = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<2>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "epoch_envelope_crossing">, <kind = data_movement, identity = "reader", operation = "epoch_envelope_crossing">, <kind = data_movement, identity = "writer", operation = "epoch_envelope_crossing">]>(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }

  func.func @epoch_envelope_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "epoch_envelope_crossing">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %a = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<2>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "epoch_envelope_crossing">, <kind = data_movement, identity = "reader", operation = "epoch_envelope_crossing">, <kind = data_movement, identity = "writer", operation = "epoch_envelope_crossing">]>(%a : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }
}
