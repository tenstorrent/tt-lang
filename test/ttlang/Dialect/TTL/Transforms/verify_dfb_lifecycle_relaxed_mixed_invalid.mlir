// RUN: env TTL_RELAX_DFB_SPSC=1 ttlang-opt %s --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices,ttl-verify-dfb-spsc,ttl-verify-dfb-lifecycle)'

// A data-movement kernel that performs both roles is checked on its combined
// sequence even when relaxed ownership admits another consumer kernel on the
// same node.
// expected-warning @+1 {{`TTL_RELAX_DFB_SPSC` disables per-launch-node DFB producer, consumer, and wait correspondence checks}}
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @mixed_roles() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // expected-error @below {{logical DFB 0 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
    // expected-note @below {{the single kernel reaches 2 outstanding block(s), exceeding capacity 1}}
    // expected-note @below {{keep reserve/push and wait/pop ordered, and pop consumed blocks before the outstanding count exceeds DFB capacity}}
    %first = ttl.cb_reserve %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.cb_reserve %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %view = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    func.return
  }

  func.func @other_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %view = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    func.return
  }
}
