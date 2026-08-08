// Verify diagnostics for unresolved logical identities, incompatible physical
// assignments, and partial compiler-created lifecycles.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false})'

// Distinct logical DFBs with incompatible types cannot retain one provisional
// physical index.
module {
  func.func @bf16_declaration()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }

  func.func @f32_declaration()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    // expected-error @below {{provisional physical DFB index 0 aliases conflicting logical DFBs 1 and 0}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    return
  }
}

// -----

// User-declared DFBs require an explicit module-wide logical identity.
module {
  func.func @missing_user_identity()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    // expected-error @below {{user-declared DFB requires dfb_id before physical allocation}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Generated identities must not overflow the index attribute domain.
module {
  func.func @exhausted_identity_domain()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %user = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 9223372036854775807 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{logical DFB identifiers leave no space for compiler-created DFBs}}
    %compiler = ttl.bind_cb {cb_index = 1, block_count = 2}
        {ttl.compiler_allocated}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Declarations of one logical DFB require one exact type.
module {
  func.func @type_mismatch_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }

  func.func @type_mismatch_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    // expected-error @below {{logical DFB 0 has inconsistent types across kernel functions}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    return
  }
}

// -----

// One logical DFB requires one physical index in every participating kernel.
module {
  func.func @physical_index_mismatch_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }

  func.func @physical_index_mismatch_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    // expected-error @below {{logical DFB 0 has inconsistent physical indices 0 and 1}}
    %dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// A producer lifecycle requires a reserve before its push.
func.func @missing_reserve()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  // expected-error @below {{'ttl.bind_cb' op compiler-allocated logical DFB has a partial lifecycle: missing ttl.cb_reserve}}
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  return
}

// -----

// A producer lifecycle requires a push after its reserve.
func.func @missing_push()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  // expected-error @below {{'ttl.bind_cb' op compiler-allocated logical DFB has a partial lifecycle: missing ttl.cb_push}}
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %reserve = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %wait = ttl.cb_wait %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  return
}

// -----

// A consumer lifecycle requires a wait before its pop.
func.func @missing_wait()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  // expected-error @below {{'ttl.bind_cb' op compiler-allocated logical DFB has a partial lifecycle: missing ttl.cb_wait}}
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %reserve = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  return
}

// -----

// A consumer lifecycle requires a pop after its wait.
func.func @missing_pop()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  // expected-error @below {{'ttl.bind_cb' op compiler-allocated logical DFB has a partial lifecycle: missing ttl.cb_pop}}
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %reserve = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----
// expected-error @-1 {{cannot determine DFB page size for element type i4}}

// Runtime DFB metadata cannot represent a fractional-byte page.
func.func @sub_byte_page_size()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[128], i4, 2>
  return
}
