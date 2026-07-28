// Tests invalid logical identities and physical DFB allocation metadata.
//
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false})'

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
    // expected-error @below {{physical DFB index 0 has inconsistent CircularBufferType values}}
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
