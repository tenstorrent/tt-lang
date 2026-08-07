// Tests invalid logical DFB declarations for concurrent-kernel liveness.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)'

// One logical DFB must have one exact type across all kernel functions.

func.func @type_mismatch_producer()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

func.func @type_mismatch_consumer()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  // expected-error @below {{logical DFB 0 has inconsistent types across kernel functions}}
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  return
}

// -----

// Default-mode allocation rejects a compiler-created DFB without a consumer
// acquire.

module {
  func.func @producer_only()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-error @below {{'ttl.bind_cb' op compiler-allocated logical DFB has a partial lifecycle: missing ttl.cb_wait}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {ttl.compiler_allocated}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reserved = ttl.cb_reserve %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// A cast does not hide a partial compiler-created lifecycle from validation.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @cast_partial_lifecycle()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{'ttl.bind_cb' op compiler-allocated logical DFB has a partial lifecycle: missing ttl.cb_wait}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {ttl.compiler_allocated}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cast = builtin.unrealized_conversion_cast %dfb
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
        to !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reserved = ttl.cb_reserve %cast
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cast : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// A compiler-created DFB accessed only by a custom function is used and must
// provide a visible lifecycle.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @custom_function_without_lifecycle()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{'ttl.bind_cb' op compiler-allocated logical DFB has a partial lifecycle: missing ttl.cb_reserve}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {ttl.compiler_allocated}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "custom_consume" (%dfb) {header = "custom_consume.hpp"}
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
    return
  }
}

// -----

// A physical index passed to a custom function must have a direct DFB
// dependency operand on the same call.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @custom_function_missing_dependency()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb_index = ttl.get_dfb_id %dfb
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{'ttl.opaque_call' op custom function consumes the physical index for logical DFB 0 without listing that DFB as a dependency operand}}
    ttl.opaque_call "custom_consume" (%dfb_index)
        {header = "custom_consume.hpp"} : (i32) -> ()
    return
  }
}

// -----

// Pure integer operations cannot hide a physical DFB index from the custom
// function dependency check.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @custom_function_laundered_dependency()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb_index = ttl.get_dfb_id %dfb
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %zero = arith.constant 0 : i32
    %laundered = arith.addi %dfb_index, %zero : i32
    // expected-error @below {{'ttl.opaque_call' op custom function consumes the physical index for logical DFB 0 without listing that DFB as a dependency operand}}
    ttl.opaque_call "custom_consume" (%laundered)
        {header = "custom_consume.hpp"} : (i32) -> ()
    return
  }
}

// -----

// Passing a physical DFB index across a function boundary is an unanalyzable
// escape even when the local callee would return the value unchanged.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func private @forward_index(i32) -> i32

  func.func @function_forwarded_dependency()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb_index = ttl.get_dfb_id %dfb
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{'func.call' op physical index for logical DFB 0 escapes through an unsupported operation}}
    %forwarded = func.call @forward_index(%dfb_index) : (i32) -> i32
    ttl.opaque_call "custom_consume" (%forwarded)
        {header = "custom_consume.hpp"} : (i32) -> ()
    return
  }
}

// -----

// Default-mode allocation rejects a compiler-created DFB without a producer
// acquire.

module {
  func.func @consumer_only()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-error @below {{'ttl.bind_cb' op compiler-allocated logical DFB has a partial lifecycle: missing ttl.cb_reserve}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {ttl.compiler_allocated}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %available = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// An unbounded lifetime conflicts with every other lifetime. Thirty-three
// unbounded logical DFBs must be rejected rather than unsafely compacted.

// expected-error @below {{DFB allocation needs 33 unspilled physical indices but hardware supports at most 32}}
module {
  func.func @unbounded_over_capacity()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 33 : i32, ttl.crta_indices = []} {
    %dfb0 = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb1 = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb2 = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb3 = ttl.bind_cb {cb_index = 3, block_count = 2} {dfb_id = 3 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb4 = ttl.bind_cb {cb_index = 4, block_count = 2} {dfb_id = 4 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb5 = ttl.bind_cb {cb_index = 5, block_count = 2} {dfb_id = 5 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb6 = ttl.bind_cb {cb_index = 6, block_count = 2} {dfb_id = 6 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb7 = ttl.bind_cb {cb_index = 7, block_count = 2} {dfb_id = 7 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb8 = ttl.bind_cb {cb_index = 8, block_count = 2} {dfb_id = 8 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb9 = ttl.bind_cb {cb_index = 9, block_count = 2} {dfb_id = 9 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb10 = ttl.bind_cb {cb_index = 10, block_count = 2} {dfb_id = 10 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb11 = ttl.bind_cb {cb_index = 11, block_count = 2} {dfb_id = 11 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb12 = ttl.bind_cb {cb_index = 12, block_count = 2} {dfb_id = 12 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb13 = ttl.bind_cb {cb_index = 13, block_count = 2} {dfb_id = 13 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb14 = ttl.bind_cb {cb_index = 14, block_count = 2} {dfb_id = 14 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb15 = ttl.bind_cb {cb_index = 15, block_count = 2} {dfb_id = 15 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb16 = ttl.bind_cb {cb_index = 16, block_count = 2} {dfb_id = 16 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb17 = ttl.bind_cb {cb_index = 17, block_count = 2} {dfb_id = 17 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb18 = ttl.bind_cb {cb_index = 18, block_count = 2} {dfb_id = 18 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb19 = ttl.bind_cb {cb_index = 19, block_count = 2} {dfb_id = 19 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb20 = ttl.bind_cb {cb_index = 20, block_count = 2} {dfb_id = 20 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb21 = ttl.bind_cb {cb_index = 21, block_count = 2} {dfb_id = 21 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb22 = ttl.bind_cb {cb_index = 22, block_count = 2} {dfb_id = 22 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb23 = ttl.bind_cb {cb_index = 23, block_count = 2} {dfb_id = 23 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb24 = ttl.bind_cb {cb_index = 24, block_count = 2} {dfb_id = 24 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb25 = ttl.bind_cb {cb_index = 25, block_count = 2} {dfb_id = 25 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb26 = ttl.bind_cb {cb_index = 26, block_count = 2} {dfb_id = 26 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb27 = ttl.bind_cb {cb_index = 27, block_count = 2} {dfb_id = 27 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb28 = ttl.bind_cb {cb_index = 28, block_count = 2} {dfb_id = 28 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb29 = ttl.bind_cb {cb_index = 29, block_count = 2} {dfb_id = 29 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb30 = ttl.bind_cb {cb_index = 30, block_count = 2} {dfb_id = 30 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb31 = ttl.bind_cb {cb_index = 31, block_count = 2} {dfb_id = 31 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb32 = ttl.bind_cb {cb_index = 32, block_count = 2} {dfb_id = 32 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
