// Verifies feasible repeated opaque-call streams may share a typed DFB allocation.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true unsafe-assume-allocation-groups=true})' | FileCheck %s

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

// -----

// Repeated native producer operations still require exact synchronization
// with repeated effects described by an opaque consumer call.

// CHECK-LABEL: func.func @native_repeated_producer
// CHECK: ttl.bind_cb{cb_index = [[SHARED:[0-9]+]], block_count = 2} {dfb_id = 0 : index}
// CHECK: ttl.bind_cb{cb_index = {{[0-9]+}}, block_count = 2} {dfb_id = 1 : index}
// CHECK: ttl.bind_cb{cb_index = [[SHARED]], block_count = 2} {dfb_id = 2 : index}

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @native_repeated_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32,
                  ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %completion = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 2, block_count = 2}
        {dfb_id = 2 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %transaction = %lower to %upper step %step {
      %reserved = ttl.cb_reserve %first
          : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %first : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    }
    %signal = ttl.cb_wait %completion
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %completion : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    scf.for %transaction = %lower to %upper step %step {
      %reserved = ttl.cb_reserve %second
          : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    }
    return
  }

  func.func @opaque_repeated_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %completion = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 2, block_count = 2}
        {dfb_id = 2 : index}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "consume_first"
        dfb_dependencies(%first, %completion
            : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>,
              !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>,
                     #ttl.dfb_protocol_effect<reserve, 1, 1>,
                     #ttl.dfb_protocol_effect<push, 1, 1>]
        () {header = "effects.hpp"} : () -> ()
    ttl.opaque_call "consume_second"
        dfb_dependencies(%second
            : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>,
                     #ttl.dfb_protocol_effect<wait, 0, 4>,
                     #ttl.dfb_protocol_effect<pop, 0, 4>]
        () {header = "effects.hpp"} : () -> ()
    return
  }
}
