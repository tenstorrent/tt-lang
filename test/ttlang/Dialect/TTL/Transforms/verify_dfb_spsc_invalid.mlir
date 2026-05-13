// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -ttl-verify-dfb-spsc

// Two kernel threads each call `cb_wait` on cb_index=0; the verifier rejects
// because tt-metal CBs use a single `pages_acked` counter that races when
// more than one thread pops. The diagnostic also attaches a "declared here"
// note pointing at the first `ttl.bind_cb` for this index.

module {
  func.func @two_consumers_compute() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{dataflow buffer cb_index=0 has 2 consumer threads}}
    // expected-note @below {{tt-metal CBs are single-producer single-consumer; allocate one DFB per consumer}}
    %v = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }

  func.func @two_consumers_noc() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-note @below {{also waited on here}}
    %v = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Two kernel threads each call `cb_reserve`; rejected for the symmetric
// reason on the producer side.

module {
  func.func @two_producers_compute() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 3, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{dataflow buffer cb_index=3 has 2 producer threads}}
    // expected-note @below {{tt-metal CBs are single-producer single-consumer; allocate one DFB per producer}}
    %v = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }

  func.func @two_producers_noc() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 3, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-note @below {{also reserved here}}
    %v = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Three threads consume the same DFB; one error site, one note per other
// thread.

module {
  func.func @consumer_a() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 5, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{dataflow buffer cb_index=5 has 3 consumer threads}}
    // expected-note @below {{tt-metal CBs are single-producer single-consumer; allocate one DFB per consumer}}
    %v = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }

  func.func @consumer_b() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 5, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-note @below {{also waited on here}}
    %v = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }

  func.func @consumer_c() attributes {ttl.kernel_thread = #ttkernel.thread<ethernet>} {
    %cb = ttl.bind_cb {cb_index = 5, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-note @below {{also waited on here}}
    %v = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Same `cb_index` violates SPSC on BOTH the producer and consumer sides.
// The pass emits one error per violated role and continues past the first.

module {
  func.func @both_sides_a() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // The pass attaches one "declared here" note per error (so two total
    // here). MLIR's verify-diagnostics matches at most one expected-note
    // per emitted note, so we assert only the producer-side note; the
    // consumer-side note is exercised by the single-role cases above.
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 8, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{dataflow buffer cb_index=8 has 2 producer threads}}
    // expected-note @below {{tt-metal CBs are single-producer single-consumer; allocate one DFB per producer}}
    %r = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-error @below {{dataflow buffer cb_index=8 has 2 consumer threads}}
    // expected-note @below {{tt-metal CBs are single-producer single-consumer; allocate one DFB per consumer}}
    %w = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }

  func.func @both_sides_b() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 8, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-note @below {{also reserved here}}
    %r = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-note @below {{also waited on here}}
    %w = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}
