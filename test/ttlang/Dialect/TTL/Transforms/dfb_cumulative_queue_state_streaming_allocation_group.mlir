// Verifies feasible repeated opaque-call streams may share a typed DFB allocation.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true unsafe-assume-allocation-groups=true})' | FileCheck %s

// Producer and consumer effects within separate opaque calls can require each
// call to make partial progress before the other completes. The finite DFB
// capacity permits this schedule even though operation-completion edges in
// both directions would form a cycle.

// CHECK-LABEL: func.func @streaming_producer
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 3}
// CHECK-LABEL: func.func @streaming_consumer
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 3}

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @streaming_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 8], !ttcore.tile<32x32, bf16>, 3>
    ttl.opaque_call "producer"
        dfb_dependencies(%dfb : !ttl.cb<[1, 8], !ttcore.tile<32x32, bf16>, 3>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 16>,
                     #ttl.dfb_protocol_effect<push, 0, 8>,
                     #ttl.dfb_protocol_effect<reserve, 0, 16>,
                     #ttl.dfb_protocol_effect<push, 0, 8>,
                     #ttl.dfb_protocol_effect<reserve, 0, 16>,
                     #ttl.dfb_protocol_effect<push, 0, 8>,
                     #ttl.dfb_protocol_effect<reserve, 0, 16>,
                     #ttl.dfb_protocol_effect<push, 0, 8>,
                     #ttl.dfb_protocol_effect<reserve, 0, 16>,
                     #ttl.dfb_protocol_effect<push, 0, 8>,
                     #ttl.dfb_protocol_effect<reserve, 0, 16>,
                     #ttl.dfb_protocol_effect<push, 0, 8>]
        () {header = "effects.hpp"} : () -> ()
    return
  }

  func.func @streaming_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 1 : i32,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 8], !ttcore.tile<32x32, bf16>, 3>
    ttl.opaque_call "consumer"
        dfb_dependencies(%dfb : !ttl.cb<[1, 8], !ttcore.tile<32x32, bf16>, 3>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 8>,
                     #ttl.dfb_protocol_effect<pop, 0, 8>,
                     #ttl.dfb_protocol_effect<wait, 0, 8>,
                     #ttl.dfb_protocol_effect<pop, 0, 8>,
                     #ttl.dfb_protocol_effect<wait, 0, 8>,
                     #ttl.dfb_protocol_effect<pop, 0, 8>,
                     #ttl.dfb_protocol_effect<wait, 0, 8>,
                     #ttl.dfb_protocol_effect<pop, 0, 8>,
                     #ttl.dfb_protocol_effect<wait, 0, 8>,
                     #ttl.dfb_protocol_effect<pop, 0, 8>,
                     #ttl.dfb_protocol_effect<wait, 0, 8>,
                     #ttl.dfb_protocol_effect<pop, 0, 8>]
        () {header = "effects.hpp"} : () -> ()
    return
  }
}
