// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices,ttl-verify-dfb-spsc)' | FileCheck %s
// RUN: env TTL_RELAX_DFB_SPSC=1 ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices,ttl-verify-dfb-spsc)' -o /dev/null

// Producer in one thread, consumer in another: classic SPSC, accepted.
// CHECK-LABEL: func.func @producer
// CHECK-LABEL: func.func @consumer
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %v = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }

  func.func @consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %v = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// An opaque external dependency may contain the producer protocol, so producer
// absence cannot be proven without an access contract.
// CHECK-LABEL: func.func @opaque_possible_producer
// CHECK-LABEL: func.func @opaque_consumer
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @opaque_possible_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 34 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "possible_producer" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) () {header = "opaque.hpp"} : () -> ()
    func.return
  }

  func.func @opaque_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 34 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Unknown external DFB access prevents producer-absence proof for user DFBs.
// CHECK-LABEL: func.func @unknown_possible_producer
// CHECK-LABEL: func.func @unknown_consumer
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @unknown_possible_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    ttl.opaque_call "possible_producer" () {header = "opaque.hpp", unknown_dfb_access} : () -> ()
    func.return
  }

  func.func @unknown_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 35 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_wait %dfb
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
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %r = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
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
    %cb = ttl.bind_cb {cb_index = 0, block_count = 4} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %v = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    func.return
  }

  func.func @consumer_multi_wait() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 4} {dfb_id = 1 : index}
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
  func.func @kernel_producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 4 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %slot = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }

  func.func @kernel_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 4 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %v = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }

  func.func @untagged_helper() {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 4 : index}
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
    %cb = ttl.bind_cb {cb_index = 0, block_count = 4} {dfb_id = 6 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %a = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    func.return
  }

  func.func @single_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 4} {dfb_id = 6 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %v = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Two logical DFBs are each SPSC across the same two kernels. Their distinct
// `dfb_id` values keep their participant sets separate.
// CHECK-LABEL: func.func @two_cb_producer
// CHECK-LABEL: func.func @two_cb_consumer
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @two_cb_producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb_a = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 10 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb_b = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 11 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %a = ttl.cb_reserve %cb_a
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b = ttl.cb_reserve %cb_b
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb_a : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_push %cb_b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }

  func.func @two_cb_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb_a = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 10 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb_b = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 11 : index}
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
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 20 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %slot = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }

  func.func @consumer_x0() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 20 : index}
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
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 20 : index}
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

// Hidden producer and consumer actions participate in SPSC verification
// through the shared DFB access interface.
// CHECK-LABEL: func.func @hidden_producer
// CHECK-LABEL: func.func @hidden_consumer
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @hidden_producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 31 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "produce" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>] () {header = "effects.hpp"} : () -> ()
    func.return
  }

  func.func @hidden_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 31 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "consume" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    func.return
  }
}

// -----

// Concrete acquisitions and hidden releases in the same thread remain one
// participant for each SPSC role.
// CHECK-LABEL: func.func @same_thread_hidden_releases
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @consumer_source() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %consumer = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 33 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reserved = ttl.cb_reserve %consumer
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %consumer : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }

  func.func @same_thread_hidden_releases() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %producer = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 32 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %consumer = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 33 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reserved = ttl.cb_reserve %producer
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.opaque_call "push" dfb_dependencies(%producer : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<push, 0, 1>] () {header = "effects.hpp"} : () -> ()
    %waited = ttl.cb_wait %consumer
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.opaque_call "pop" dfb_dependencies(%consumer : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    func.return
  }
}

// -----

// Release-only protocol actions do not require launch-domain analysis.
// CHECK-LABEL: func.func @release_only
module {
  func.func @release_only() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 34 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "push" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<push, 0, 1>] () {header = "effects.hpp"} : () -> ()
    func.return
  }
}

// -----

// Two logical DFBs may share a physical index when each remains SPSC.
// CHECK-LABEL: func.func @first_reused_producer
// CHECK-LABEL: func.func @first_reused_consumer
// CHECK-LABEL: func.func @second_reused_producer
// CHECK-LABEL: func.func @second_reused_consumer
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @first_reused_producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 10 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %slot = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }

  func.func @first_reused_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 10 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %view = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }

  func.func @second_reused_producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 11 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %slot = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }

  func.func @second_reused_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 11 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %view = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Two producer kernels may reserve the same DFB when their launch-node domains
// are disjoint.
// CHECK-LABEL: func.func @producer_x0
// CHECK-LABEL: func.func @producer_x1
module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @producer_x0() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 21 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %is_x0 = arith.cmpi eq, %core_x, %zero : index
    scf.if %is_x0 {
      %slot = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    func.return
  }

  func.func @producer_x1() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 21 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %one = arith.constant 1 : index
    %is_x1 = arith.cmpi eq, %core_x, %one : index
    scf.if %is_x1 {
      %slot = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    func.return
  }
}

// -----

// PipeNet role domains may make two consumer threads disjoint.
// CHECK-LABEL: func.func @consumer_dst
// CHECK-LABEL: func.func @consumer_src
module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 21 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %slot = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }

  func.func @consumer_dst() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %pipe = ttl.create_pipe src(1, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(1, 0) dst(0, 0) to(0, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 21 : index}
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
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 21 : index}
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
