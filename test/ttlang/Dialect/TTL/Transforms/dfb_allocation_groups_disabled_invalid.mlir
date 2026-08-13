// Tests allocation groups when automatic user DFB reuse is disabled.
// RUN: ttlang-opt %s --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false})'

module {
  func.func @disabled_reuse()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    // expected-error @below {{DFB allocation groups require user DFB reuse to be enabled}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
