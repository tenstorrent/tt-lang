// RUN: ttlang-opt %s --split-input-file -ttl-verify-dfb-spsc | FileCheck %s

// Producer in one thread, consumer in another: classic SPSC, accepted.
// CHECK-LABEL: func.func @producer
// CHECK-LABEL: func.func @consumer
module {
  func.func @producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %v = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }

  func.func @consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %v = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Producer and consumer in the same thread: still SPSC (one producer thread,
// one consumer thread); the verifier counts threads, not ops.
// CHECK-LABEL: func.func @produce_and_consume
module {
  func.func @produce_and_consume() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 2, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %r = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %w = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Multiple `cb_wait` calls inside one thread are fine: only the thread set
// matters, not the call count.
// CHECK-LABEL: func.func @consumer_multi_wait
module {
  func.func @producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 1, block_count = 4}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %v = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }

  func.func @consumer_multi_wait() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 1, block_count = 4}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %a = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Ops not tagged with `ttl.kernel_thread` are ignored entirely. This matters
// because helper or host funcs may share a CB declaration without participating
// in the runtime push/pop protocol.
// CHECK-LABEL: func.func @kernel_consumer
// CHECK-LABEL: func.func @untagged_helper
module {
  func.func @kernel_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 4, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %v = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }

  func.func @untagged_helper() {
    %cb = ttl.bind_cb {cb_index = 4, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %v = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Multiple `cb_reserve` calls inside one producer thread are fine: the verifier
// counts threads per role, not ops.
// CHECK-LABEL: func.func @producer_multi_reserve
module {
  func.func @producer_multi_reserve() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 6, block_count = 4}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %a = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }

  func.func @single_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 6, block_count = 4}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %v = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Two distinct CBs each correctly SPSC across the same two threads: the
// verifier disambiguates by `cb_index` and accepts both.
// CHECK-LABEL: func.func @two_cb_producer
// CHECK-LABEL: func.func @two_cb_consumer
module {
  func.func @two_cb_producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb_a = ttl.bind_cb {cb_index = 10, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb_b = ttl.bind_cb {cb_index = 11, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %a = ttl.cb_reserve %cb_a
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b = ttl.cb_reserve %cb_b
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }

  func.func @two_cb_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb_a = ttl.bind_cb {cb_index = 10, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb_b = ttl.bind_cb {cb_index = 11, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %a = ttl.cb_wait %cb_a
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b = ttl.cb_wait %cb_b
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}
