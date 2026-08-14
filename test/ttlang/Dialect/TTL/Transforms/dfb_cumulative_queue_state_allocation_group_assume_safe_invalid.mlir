// Tests cumulative queue contradictions that unsafe allocation groups cannot override.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true unsafe-assume-allocation-groups=true})'

// Waiting for the second publication before the pop that returns its capacity
// would deadlock. The rejected required synchronization edge remains
// contradictory evidence rather than becoming an unsafe missing-order assumption.

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @cyclic_capacity_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<0> members=[0, 1] has contradictory cursor order involving logical DFB 0 on launch node (0,0)}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %other = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
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
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %other = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
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
