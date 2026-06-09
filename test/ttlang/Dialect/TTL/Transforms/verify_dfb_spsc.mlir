// RUN: ttlang-opt %s --split-input-file -ttl-verify-dfb-spsc | FileCheck %s

// Producer in one thread, consumer in another: classic SPSC, accepted.
// CHECK-LABEL: func.func @producer
// CHECK-LABEL: func.func @consumer
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
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
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
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
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
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
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
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
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
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
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
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

// -----

// Two consumer threads may wait on the same DFB when their launch-node domains
// are disjoint.
// CHECK-LABEL: func.func @consumer_x0
// CHECK-LABEL: func.func @consumer_x1
module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @producer_all_nodes() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 20, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %slot = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }

  func.func @consumer_x0() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 20, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %is_x0 = arith.cmpi eq, %core_x, %zero : index
    scf.if %is_x0 {
      %view = ttl.cb_wait %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    func.return
  }

  func.func @consumer_x1() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 20, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %one = arith.constant 1 : index
    %is_x1 = arith.cmpi eq, %core_x, %one : index
    scf.if %is_x1 {
      %view = ttl.cb_wait %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    func.return
  }
}

// -----

// PipeNet role domains may make two consumer threads disjoint.
// CHECK-LABEL: func.func @consumer_dst
// CHECK-LABEL: func.func @consumer_src
module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @consumer_dst() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %pipe = ttl.create_pipe src(1, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(1, 0) dst(0, 0) to(0, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 21, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_dst %pipe : !ttl.pipe<src(1, 0) dst(0, 0) to(0, 0) net 0> {
      %view = ttl.cb_wait %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    func.return
  }

  func.func @consumer_src() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(1, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(1, 0) dst(0, 0) to(0, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 21, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(1, 0) dst(0, 0) to(0, 0) net 0> {
      %view = ttl.cb_wait %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    func.return
  }
}
