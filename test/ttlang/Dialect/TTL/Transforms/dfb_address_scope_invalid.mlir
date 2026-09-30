// Tests invalid DFB address scopes.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)'

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @invalid_scope() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.noc_index = 0 : i32
  } {
    // expected-error @below {{address_scope must be 'local' or 'remote_uniform'}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        {address_scope = "operation_uniform", dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}

// -----

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @first_declaration() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.noc_index = 0 : i32
  } {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1}
        {address_scope = "local", dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }

  func.func @inconsistent_declaration() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.noc_index = 1 : i32
  } {
    // expected-error @below {{logical DFB 0 has inconsistent address scopes across kernel functions: expected local but found remote_uniform}}
    %second = ttl.bind_cb {cb_index = 0, block_count = 1}
        {address_scope = "remote_uniform", dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
