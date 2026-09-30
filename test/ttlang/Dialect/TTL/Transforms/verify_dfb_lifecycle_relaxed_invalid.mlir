// RUN: env TTL_RELAX_DFB_SPSC=1 ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices,ttl-verify-dfb-spsc,ttl-verify-dfb-lifecycle)'

// Relaxing participant ownership must not suppress a missing producer release.
// expected-warning @+1 {{`TTL_RELAX_DFB_SPSC` disables per-launch-node DFB producer, consumer, and wait correspondence checks}}
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @producer_complete() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_reserve %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }

  func.func @producer_incomplete() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{logical DFB 0 has an incomplete or misordered producer lifecycle on core_x=0, core_y=0}}
    // expected-note @below {{the producer kernel finishes with 1 open reserve block(s) without a push}}
    // expected-note @below {{keep each reserve followed by its matching push before opening more than DFB capacity}}
    %block = ttl.cb_reserve %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }

  func.func @consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }
}

// -----

// Exact counts prove that every visible kernel acts on the node, so relaxed
// ownership does not exempt their totals: two producers reserving once each
// cannot satisfy three waits.
// expected-warning @+1 {{`TTL_RELAX_DFB_SPSC` disables per-launch-node DFB producer, consumer, and wait correspondence checks}}
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @producer_first() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{logical DFB 0 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
    // expected-note @below {{the consumer pops 3 block(s) per launch, but the producer pushes 2 block(s) per launch}}
    // expected-note @below {{keep pops within pushes and unpopped blocks within DFB capacity on every active node}}
    %block = ttl.cb_reserve %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }

  func.func @producer_second() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_reserve %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }

  func.func @consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %third = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }
}
