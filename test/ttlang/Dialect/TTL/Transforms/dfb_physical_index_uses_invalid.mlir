// Tests invalid physical DFB-index dataflow through pure integer operations.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)'
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false})'

// Truncating an index to i1 preserves index data, so widening the value again
// cannot hide a custom-function use that omits the source DFB dependency.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @producer() attributes {ttl.noc_index = 0 : i32,
      ttl.kernel_thread = #ttkernel.thread<noc>} {
    %a = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %ack = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %b = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %a_slot = ttl.cb_reserve %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %ack_val = ttl.cb_wait %ack : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_pop %ack : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    %b_slot = ttl.cb_reserve %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }

  func.func @consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %a = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %ack = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %b = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %a_val = ttl.cb_wait %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %ack_slot = ttl.cb_reserve %ack : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_push %ack : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    %b_val = ttl.cb_wait %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %a_index = ttl.get_dfb_id %a
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %narrow = arith.trunci %a_index : i32 to i1
    %wide = arith.extui %narrow : i1 to i32
    // expected-error @below {{'ttl.opaque_call' op custom function consumes the physical index for logical DFB 0 without listing that DFB as a dependency operand}}
    ttl.opaque_call "read_dfb_by_index" (%wide)
        {header = "read_dfb_by_index.hpp"} : (i32) -> ()
    ttl.cb_pop %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// A DFB-index template argument exposes the physical index but does not by
// itself declare that the custom function accesses the referenced storage.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @index_template_without_dependency()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{'ttl.opaque_call' op custom function consumes the physical index for logical DFB 0 without listing that DFB as a dependency operand}}
    ttl.opaque_call "read_dfb_by_index"
        template_args [#ttl.external_template_arg<dfb_index, 0>]
        template_dfbs(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        () {header = "read_dfb_by_index.hpp"} : () -> ()
    return
  }
}
