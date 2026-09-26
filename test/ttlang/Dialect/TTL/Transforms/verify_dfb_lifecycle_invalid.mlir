// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices,ttl-verify-dfb-lifecycle)'

// Summary: Negative tests for per-launch-node DFB transaction verification.

// Repeated publications without a consumer are rejected even when the DFB is
// not used by a Pipe.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @producer_only() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 33 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      // expected-error @below {{logical DFB 33 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
      // expected-note @below {{the producer pushes 4 block(s) per launch and the consumer pops 0, leaving 4 outstanding block(s) for capacity 1}}
      // expected-note @below {{keep pops within pushes and unpopped blocks within DFB capacity on every active node}}
      %slot = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    func.return
  }
}

// -----

// A published block is not reusable until its consumer pops it.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 92 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %reserved = ttl.cb_reserve %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    func.return
  }

  func.func @consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 92 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // expected-error @below {{logical DFB 92 has an incomplete or misordered consumer lifecycle on core_x=0, core_y=0}}
    // expected-note @below {{the consumer kernel finishes with 1 open wait block(s) without a pop}}
    // expected-note @below {{keep each wait followed by its matching pop before opening more than DFB capacity}}
    %waited = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Endpoint surplus can fit the DFB while the ordered producer prefix exceeds
// capacity before the consumer transaction executes.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @peak_exceeds_capacity_with_surplus() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 35 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %producer_upper = arith.constant 2 : index
    %consumer_upper = arith.constant 1 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %producer_upper step %step {
      // expected-error @below {{logical DFB 35 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
      // expected-note @below {{the single kernel reaches 2 outstanding block(s), exceeding capacity 1}}
      // expected-note @below {{keep reserve/push and wait/pop ordered, and pop consumed blocks before the outstanding count exceeds DFB capacity}}
      %slot = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    scf.for %iteration = %lower to %consumer_upper step %step {
      %view = ttl.cb_wait %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    func.return
  }
}

// -----

// Equal endpoint counts do not make a sequential producer prefix safe.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @peak_exceeds_capacity_with_equal_counts() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 36 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 5 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      // expected-error @below {{logical DFB 36 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
      // expected-note @below {{the single kernel reaches 5 outstanding block(s), exceeding capacity 1}}
      // expected-note @below {{keep reserve/push and wait/pop ordered, and pop consumed blocks before the outstanding count exceeds DFB capacity}}
      %slot = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    scf.for %iteration = %lower to %upper step %step {
      %view = ttl.cb_wait %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    func.return
  }
}

// -----

// A wait does not release capacity until its matching pop executes.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @reserve_before_prior_wait_is_popped() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 37 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // expected-error @below {{logical DFB 37 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
    // expected-note @below {{the single kernel reaches 2 outstanding block(s), exceeding capacity 1}}
    // expected-note @below {{keep reserve/push and wait/pop ordered, and pop consumed blocks before the outstanding count exceeds DFB capacity}}
    %first = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %first_view = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %second = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second_view = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    func.return
  }
}

// -----

// Repeated producer-side Pipe sends exceed capacity when no consumer returns a
// slot.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unbalanced_pipe_source() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 30 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      scf.for %iteration = %lower to %upper step %step {
        // expected-error @below {{logical DFB 30 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
        // expected-note @below {{the producer pushes 2 block(s) per launch and the consumer pops 0, leaving 2 outstanding block(s) for capacity 1}}
        // expected-note @below {{keep pops within pushes and unpopped blocks within DFB capacity on every active node}}
        %slot = ttl.cb_reserve %cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %send = ttl.copy %cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      }
    }
    func.return
  }
}

// -----

// A distinct consumer cannot release capacity when the producer opens more
// reserve transactions than capacity before publishing any of them.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @grouped_reserve_producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 90 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // expected-error @below {{logical DFB 90 has a producer lifecycle that can block on core_x=0, core_y=0}}
    // expected-note @below {{the producer kernel reaches 2 open reserve block(s), exceeding capacity 1}}
    // expected-note @below {{keep each reserve followed by its matching push before opening more than DFB capacity}}
    %first = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %second = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    func.return
  }

  func.func @interleaved_pop_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 90 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %first = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    func.return
  }
}

// -----

// A distinct producer cannot publish more data when the consumer opens more
// wait transactions than capacity before releasing any of them.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @interleaved_push_producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 91 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %first = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    func.return
  }

  func.func @grouped_wait_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 91 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // expected-error @below {{logical DFB 91 has a consumer lifecycle that can block on core_x=0, core_y=0}}
    // expected-note @below {{the consumer kernel reaches 2 open wait block(s), exceeding capacity 1}}
    // expected-note @below {{keep each wait followed by its matching pop before opening more than DFB capacity}}
    %first = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %second = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    func.return
  }
}

// -----

// Equal launch-node domains do not prove capacity safety. Five producer
// acquisitions and one consumer acquisition exceed capacity on core (0, 0).

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @five_pipe_publications() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 31 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 5 : index
    %step = arith.constant 1 : index
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      scf.for %iteration = %lower to %upper step %step {
        // expected-error @below {{logical DFB 31 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
        // expected-note @below {{the producer pushes 5 block(s) per launch and the consumer pops 1, leaving 4 outstanding block(s) for capacity 2}}
        // expected-note @below {{keep pops within pushes and unpopped blocks within DFB capacity on every active node}}
        %slot = ttl.cb_reserve %cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %send = ttl.copy %cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      }
    }
    func.return
  }

  func.func @one_pipe_consumption() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 31 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %view = ttl.cb_wait %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    func.return
  }
}

// -----

// Multiple producer sites are checked in order when a wait separates them.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @later_reserve_selects_write_pointer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 32 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      // expected-error @below {{logical DFB 32 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
      // expected-note @below {{the single kernel reaches 2 outstanding block(s), exceeding capacity 1}}
      // expected-note @below {{keep reserve/push and wait/pop ordered, and pop consumed blocks before the outstanding count exceeds DFB capacity}}
      %first_slot = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      %consumed_slot = ttl.cb_wait %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      %second_slot = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      %third_slot = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// An `inspect` contract declares that the callee performs no protocol
// actions, so it does not exempt the node from verification.
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @inspecting_overcapacity_producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 38 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.opaque_call "inspect"
        dfb_dependencies(%cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        dfb_accesses [#ttl.dfb_non_transactional_access<inspect, 0>]
        () {header = "inspect.hpp"} : () -> ()
    // expected-error @below {{logical DFB 38 has a producer lifecycle that can block on core_x=0, core_y=0}}
    // expected-note @below {{the producer kernel reaches 2 open reserve block(s), exceeding capacity 1}}
    // expected-note @below {{keep each reserve followed by its matching push before opening more than DFB capacity}}
    %first = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %second = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    func.return
  }

  func.func @consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 38 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %first = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    func.return
  }
}

// -----

// A release without a preceding acquisition is rejected even when the DFB has
// no acquisition anywhere in the module.
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @orphan_push() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 39 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{logical DFB 39 has an incomplete or misordered producer lifecycle on core_x=0, core_y=0}}
    // expected-note @below {{the producer kernel performs a push before a matching reserve}}
    // expected-note @below {{keep each reserve followed by its matching push before opening more than DFB capacity}}
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }
}

// -----

// The nodes an opaque call may execute on are its domain's upper bound. A call
// under a node predicate and a runtime condition excludes only the predicate's
// nodes; the other node is still verified.
module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @producer_with_guarded_external_consumer(%runtime: i1) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 60 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %is_x0 = arith.cmpi eq, %core_x, %zero : index
    %guard = arith.andi %is_x0, %runtime : i1
    scf.if %guard {
      ttl.opaque_call "consume"
          dfb_dependencies(%cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
          () {header = "consume.hpp"} : () -> ()
    }
    scf.for %iteration = %zero to %four step %one {
      // expected-error @below {{logical DFB 60 has capacity-unsafe producer and consumer transactions on core_x=1, core_y=0}}
      // expected-note @below {{the producer pushes 4 block(s) per launch and the consumer pops 0, leaving 4 outstanding block(s) for capacity 1}}
      // expected-note @below {{keep pops within pushes and unpopped blocks within DFB capacity on every active node}}
      %slot = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    func.return
  }
}

// -----

// Opaque-call pushes count toward the totals: three pushed blocks against one
// pop leave two blocks outstanding in a one-block DFB.
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @opaque_two_push_producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 54 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // expected-error @below {{logical DFB 54 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
    // expected-note @below {{the producer pushes 3 block(s) per launch and the consumer pops 1, leaving 2 outstanding block(s) for capacity 1}}
    // expected-note @below {{keep pops within pushes and unpopped blocks within DFB capacity on every active node}}
    ttl.opaque_call "produce_three_times"
        dfb_dependencies(%cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>,
                     #ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>,
                     #ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>]
        () {header = "produce.hpp"} : () -> ()
    func.return
  }

  func.func @one_pop_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 54 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %block = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    func.return
  }
}

// -----

// A synchronized reset starts a new interval; the surplus before it is still
// checked against capacity.
module attributes {ttl.launch_grid = [1 : i64, 1 : i64], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @surplus_then_reset_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    scf.for %iteration = %zero to %four step %one {
      // expected-error @below {{logical DFB 0 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
      // expected-note @below {{the producer pushes 4 block(s) per launch and the consumer pops 0, leaving 4 outstanding block(s) for capacity 1}}
      // expected-note @below {{in the interval before the first synchronized reset}}
      // expected-note @below {{keep pops within pushes and unpopped blocks within DFB capacity on every active node}}
      %slot = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>
    return
  }

  func.func @surplus_then_reset_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>
    return
  }

  func.func @surplus_then_reset_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>
    return
  }
}

// -----

// A synchronized reset at the start of the launch does not exempt the
// interval after it.
module attributes {ttl.launch_grid = [1 : i64, 1 : i64], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @reset_then_surplus_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    scf.for %iteration = %zero to %four step %one {
      // expected-error @below {{logical DFB 0 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
      // expected-note @below {{the producer pushes 4 block(s) per launch and the consumer pops 0, leaving 4 outstanding block(s) for capacity 1}}
      // expected-note @below {{in the interval after synchronized reset 1}}
      // expected-note @below {{keep pops within pushes and unpopped blocks within DFB capacity on every active node}}
      %slot = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    return
  }

  func.func @reset_then_surplus_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    return
  }

  func.func @reset_then_surplus_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    return
  }
}

// -----

// An opaque summary on the same DFB does not hide the kernel's own held
// waits; the totals balance (four pushes, three user pops and one opaque pop).
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @four_pushes() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 55 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    scf.for %iteration = %zero to %four step %one {
      %slot = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    func.return
  }

  func.func @held_waits_beside_opaque_pop() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 55 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{logical DFB 55 has a consumer lifecycle that can block on core_x=0, core_y=0}}
    // expected-note @below {{the consumer kernel reaches 3 open wait block(s), exceeding capacity 2}}
    // expected-note @below {{keep each wait followed by its matching pop before opening more than DFB capacity}}
    %first = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %second = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %third = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "consume"
        dfb_dependencies(%cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "consume.hpp"} : () -> ()
    func.return
  }
}

// -----

// Unconditional waits are compared with the pushes when the pops are
// conditional: the third wait never completes.
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @two_pushes() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 0, block_count = 4} {dfb_id = 56 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    scf.for %iteration = %zero to %two step %one {
      // expected-error @below {{logical DFB 56 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
      // expected-note @below {{the consumer waits for 3 block(s) per launch, but the producer pushes 2 block(s) per launch}}
      // expected-note @below {{keep pops within pushes and unpopped blocks within DFB capacity on every active node}}
      %slot = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    }
    func.return
  }

  func.func @three_waits_conditional_pops(%condition: i1) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 4} {dfb_id = 56 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %three = arith.constant 3 : index
    scf.for %iteration = %zero to %three step %one {
      %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      scf.if %condition {
        ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
      }
    }
    func.return
  }
}

// -----

// A reset under a dispatch condition may or may not execute; the surplus
// before it is checked in the alternative where it does.
module attributes {ttl.launch_grid = [1 : i64, 1 : i64], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %flag = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %zero_i32 = arith.constant 0 : i32
    %active = arith.cmpi ne, %flag, %zero_i32 : i32
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    scf.for %iteration = %zero to %four step %one {
      // expected-error @below {{logical DFB 0 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
      // expected-note @below {{the producer pushes 4 block(s) per launch and the consumer pops 0, leaving 4 outstanding block(s) for capacity 1}}
      // expected-note @below {{in the interval before the first synchronized reset}}
      // expected-note @below {{keep pops within pushes and unpopped blocks within DFB capacity on every active node}}
      %slot = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    scf.if %active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }

  func.func @compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %flag = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %zero_i32 = arith.constant 0 : i32
    %active = arith.cmpi ne, %flag, %zero_i32 : i32
    scf.if %active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }

  func.func @writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %flag = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %zero_i32 = arith.constant 0 : i32
    %active = arith.cmpi ne, %flag, %zero_i32 : i32
    scf.if %active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }
}

// -----

// A reset under a dispatch condition at the start of the launch does not
// exempt the surplus after it.
module attributes {ttl.launch_grid = [1 : i64, 1 : i64], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %flag = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %zero_i32 = arith.constant 0 : i32
    %active = arith.cmpi ne, %flag, %zero_i32 : i32
    scf.if %active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    scf.for %iteration = %zero to %four step %one {
      // expected-error @below {{logical DFB 0 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
      // expected-note @below {{the producer pushes 4 block(s) per launch and the consumer pops 0, leaving 4 outstanding block(s) for capacity 1}}
      // expected-note @below {{in the interval after synchronized reset 1}}
      // expected-note @below {{keep pops within pushes and unpopped blocks within DFB capacity on every active node}}
      %slot = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    return
  }

  func.func @compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %flag = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %zero_i32 = arith.constant 0 : i32
    %active = arith.cmpi ne, %flag, %zero_i32 : i32
    scf.if %active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }

  func.func @writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %flag = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %zero_i32 = arith.constant 0 : i32
    %active = arith.cmpi ne, %flag, %zero_i32 : i32
    scf.if %active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }
}

// -----

// A loop with an unresolved trip count leaves only the counters it changes
// unknown; the double pop after it is still a pop before a matching wait.
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @dynamic_loop_then_double_pop(%trip: index) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 59 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scf.for %iteration = %zero to %trip step %one {
      // expected-error @below {{logical DFB 59 has an incomplete or misordered consumer lifecycle on core_x=0, core_y=0}}
      // expected-note @below {{the consumer kernel performs a pop before a matching wait}}
      // expected-note @below {{keep each wait followed by its matching pop before opening more than DFB capacity}}
      %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    %last = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }
}

// -----

// Two resets under different dispatch conditions: the alternatives that
// executed the same resets correspond across kernels, so the reader's
// surplus before the first reset is compared with the writer's pop before
// it, whatever the writer's second pop does.
module attributes {ttl.launch_grid = [1 : i64, 1 : i64], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %active0 = arith.cmpi ne, %flag0, %zero_i32 : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %active1 = arith.cmpi ne, %flag1, %zero_i32 : i32
    // expected-error @below {{logical DFB 0 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
    // expected-note @below {{the producer pushes 3 block(s) per launch and the consumer pops 1, leaving 2 outstanding block(s) for capacity 1}}
    // expected-note @below {{in the interval before the first synchronized reset}}
    // expected-note @below {{keep pops within pushes and unpopped blocks within DFB capacity on every active node}}
    %slot0 = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %slot1 = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %slot2 = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    scf.if %active0 {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    scf.if %active1 {
      ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }
  func.func @compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %active0 = arith.cmpi ne, %flag0, %zero_i32 : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %active1 = arith.cmpi ne, %flag1, %zero_i32 : i32
    scf.if %active0 {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    scf.if %active1 {
      ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }
  func.func @writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %active0 = arith.cmpi ne, %flag0, %zero_i32 : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %active1 = arith.cmpi ne, %flag1, %zero_i32 : i32
    %block0 = ttl.cb_wait %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    scf.if %active0 {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    %block1 = ttl.cb_wait %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    scf.if %active1 {
      ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }
}

// -----

// A reset and a state-discarding reconfiguration share ordinal 0. Their
// alternatives stay distinct, so the two pushes with neither executed are
// checked.
#recon = #ttl.dfb_reconfiguration<0, participants[#ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">, #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">, #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">], discard_dfb_state = true>
module attributes {ttl.launch_grid = [1 : i64, 1 : i64], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @collision_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %reset_active = arith.cmpi ne, %flag0, %zero_i32 : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %reconfigure_active = arith.cmpi ne, %flag1, %zero_i32 : i32
    // expected-error @below {{logical DFB 0 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
    // expected-note @below {{the producer pushes 2 block(s) per launch and the consumer pops 0, leaving 2 outstanding block(s) for capacity 1}}
    // expected-note @below {{in the interval before the first synchronized reset}}
    // expected-note @below {{keep pops within pushes and unpopped blocks within DFB capacity on every active node}}
    %first = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    scf.if %reset_active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    %second = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    scf.if %reconfigure_active {
      ttl.dfb_reconfiguration #recon
    }
    return
  }

  func.func @collision_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %reset_active = arith.cmpi ne, %flag0, %zero_i32 : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %reconfigure_active = arith.cmpi ne, %flag1, %zero_i32 : i32
    scf.if %reset_active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    scf.if %reconfigure_active {
      ttl.dfb_reconfiguration #recon
    }
    return
  }

  func.func @collision_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %reset_active = arith.cmpi ne, %flag0, %zero_i32 : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %reconfigure_active = arith.cmpi ne, %flag1, %zero_i32 : i32
    scf.if %reset_active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    scf.if %reconfigure_active {
      ttl.dfb_reconfiguration #recon
    }
    return
  }
}

// -----

// Four resets under nested conditions yield sixteen alternatives, so the
// surplus before them is checked.
module attributes {ttl.launch_grid = [1 : i64, 1 : i64], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @nested_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %active0 = arith.cmpi ne, %flag0, %zero_i32 : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %active1 = arith.cmpi ne, %flag1, %zero_i32 : i32
    %flag2 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<2, i32>, header = "predicate.hpp"} : () -> i32
    %active2 = arith.cmpi ne, %flag2, %zero_i32 : i32
    %flag3 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<3, i32>, header = "predicate.hpp"} : () -> i32
    %active3 = arith.cmpi ne, %flag3, %zero_i32 : i32
    %flag4 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<4, i32>, header = "predicate.hpp"} : () -> i32
    %active4 = arith.cmpi ne, %flag4, %zero_i32 : i32
    %flag5 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<5, i32>, header = "predicate.hpp"} : () -> i32
    %active5 = arith.cmpi ne, %flag5, %zero_i32 : i32
    %flag6 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<6, i32>, header = "predicate.hpp"} : () -> i32
    %active6 = arith.cmpi ne, %flag6, %zero_i32 : i32
    %flag7 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<7, i32>, header = "predicate.hpp"} : () -> i32
    %active7 = arith.cmpi ne, %flag7, %zero_i32 : i32
    // expected-error @below {{logical DFB 0 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
    // expected-note @below {{the producer pushes 3 block(s) per launch and the consumer pops 0, leaving 3 outstanding block(s) for capacity 1}}
    // expected-note @below {{in the interval before the first synchronized reset}}
    // expected-note @below {{keep pops within pushes and unpopped blocks within DFB capacity on every active node}}
    %first = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %third = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    scf.if %active0 {
      scf.if %active1 {
        ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      }
    }
    scf.if %active2 {
      scf.if %active3 {
        ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      }
    }
    scf.if %active4 {
      scf.if %active5 {
        ttl.reset_dfbs <2, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      }
    }
    scf.if %active6 {
      scf.if %active7 {
        ttl.reset_dfbs <3, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      }
    }
    return
  }

  func.func @nested_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %active0 = arith.cmpi ne, %flag0, %zero_i32 : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %active1 = arith.cmpi ne, %flag1, %zero_i32 : i32
    %flag2 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<2, i32>, header = "predicate.hpp"} : () -> i32
    %active2 = arith.cmpi ne, %flag2, %zero_i32 : i32
    %flag3 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<3, i32>, header = "predicate.hpp"} : () -> i32
    %active3 = arith.cmpi ne, %flag3, %zero_i32 : i32
    %flag4 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<4, i32>, header = "predicate.hpp"} : () -> i32
    %active4 = arith.cmpi ne, %flag4, %zero_i32 : i32
    %flag5 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<5, i32>, header = "predicate.hpp"} : () -> i32
    %active5 = arith.cmpi ne, %flag5, %zero_i32 : i32
    %flag6 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<6, i32>, header = "predicate.hpp"} : () -> i32
    %active6 = arith.cmpi ne, %flag6, %zero_i32 : i32
    %flag7 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<7, i32>, header = "predicate.hpp"} : () -> i32
    %active7 = arith.cmpi ne, %flag7, %zero_i32 : i32
    scf.if %active0 {
      scf.if %active1 {
        ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      }
    }
    scf.if %active2 {
      scf.if %active3 {
        ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      }
    }
    scf.if %active4 {
      scf.if %active5 {
        ttl.reset_dfbs <2, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      }
    }
    scf.if %active6 {
      scf.if %active7 {
        ttl.reset_dfbs <3, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      }
    }
    return
  }

  func.func @nested_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %active0 = arith.cmpi ne, %flag0, %zero_i32 : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %active1 = arith.cmpi ne, %flag1, %zero_i32 : i32
    %flag2 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<2, i32>, header = "predicate.hpp"} : () -> i32
    %active2 = arith.cmpi ne, %flag2, %zero_i32 : i32
    %flag3 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<3, i32>, header = "predicate.hpp"} : () -> i32
    %active3 = arith.cmpi ne, %flag3, %zero_i32 : i32
    %flag4 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<4, i32>, header = "predicate.hpp"} : () -> i32
    %active4 = arith.cmpi ne, %flag4, %zero_i32 : i32
    %flag5 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<5, i32>, header = "predicate.hpp"} : () -> i32
    %active5 = arith.cmpi ne, %flag5, %zero_i32 : i32
    %flag6 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<6, i32>, header = "predicate.hpp"} : () -> i32
    %active6 = arith.cmpi ne, %flag6, %zero_i32 : i32
    %flag7 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<7, i32>, header = "predicate.hpp"} : () -> i32
    %active7 = arith.cmpi ne, %flag7, %zero_i32 : i32
    scf.if %active0 {
      scf.if %active1 {
        ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      }
    }
    scf.if %active2 {
      scf.if %active3 {
        ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      }
    }
    scf.if %active4 {
      scf.if %active5 {
        ttl.reset_dfbs <2, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      }
    }
    scf.if %active6 {
      scf.if %active7 {
        ttl.reset_dfbs <3, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      }
    }
    return
  }
}

// -----

// Two resets under one dispatch condition: when it is false neither
// executes and the reader's two pushes exceed the capacity.
module attributes {ttl.launch_grid = [1 : i64, 1 : i64], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @same_condition_leak_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    // expected-error @below {{logical DFB 0 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
    // expected-note @below {{the producer pushes 2 block(s) per launch and the consumer pops 0, leaving 2 outstanding block(s) for capacity 1}}
    // expected-note @below {{in the interval before the first synchronized reset}}
    // expected-note @below {{keep pops within pushes and unpopped blocks within DFB capacity on every active node}}
    %first = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %first_active = arith.cmpi ne, %flag0, %zero_i32 : i32
    scf.if %first_active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    %second = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_again = arith.constant 0 : i32
    %second_active = arith.cmpi ne, %flag0, %zero_again : i32
    scf.if %second_active {
      ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }

  func.func @same_condition_leak_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %first_active = arith.cmpi ne, %flag0, %zero_i32 : i32
    scf.if %first_active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    %zero_again = arith.constant 0 : i32
    %second_active = arith.cmpi ne, %flag0, %zero_again : i32
    scf.if %second_active {
      ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }

  func.func @same_condition_leak_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %first_active = arith.cmpi ne, %flag0, %zero_i32 : i32
    scf.if %first_active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    %zero_again = arith.constant 0 : i32
    %second_active = arith.cmpi ne, %flag0, %zero_again : i32
    scf.if %second_active {
      ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }
}

// -----

// The reader pushes inside the outer branch and twice after it; when the
// inner reset is skipped the pushes exceed the capacity, which only the
// combination of both kernels' alternatives for that decision shows.
module attributes {ttl.launch_grid = [1 : i64, 1 : i64], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @nested_leak_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %outer = arith.cmpi ne, %flag0, %zero_i32 : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %inner = arith.cmpi ne, %flag1, %zero_i32 : i32
    scf.if %outer {
      // expected-error @below {{logical DFB 0 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
      // expected-note @below {{the producer pushes 3 block(s) per launch and the consumer pops 1, leaving 2 outstanding block(s) for capacity 1}}
      // expected-note @below {{in the interval before the first synchronized reset}}
      // expected-note @below {{keep pops within pushes and unpopped blocks within DFB capacity on every active node}}
      %first = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      scf.if %inner {
        ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      }
    }
    %second = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %third = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }

  func.func @nested_leak_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %outer = arith.cmpi ne, %flag0, %zero_i32 : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %inner = arith.cmpi ne, %flag1, %zero_i32 : i32
    scf.if %outer {
      scf.if %inner {
        ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      }
    }
    return
  }

  func.func @nested_leak_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %outer = arith.cmpi ne, %flag0, %zero_i32 : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %inner = arith.cmpi ne, %flag1, %zero_i32 : i32
    scf.if %outer {
      scf.if %inner {
        ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      }
    }
    %block = ttl.cb_wait %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}

// -----

// The pops are conditional, so the unconditional waits are compared with
// the pushes in the interval before the reset: the third wait never
// completes.
module attributes {ttl.launch_grid = [1 : i64, 1 : i64], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @segmented_waits_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %stale = ttl.bind_cb {cb_index = 0, block_count = 4} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %pop_active = arith.cmpi ne, %flag0, %zero_i32 : i32
    // expected-error @below {{logical DFB 0 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
    // expected-note @below {{the consumer waits for 3 block(s) per launch, but the producer pushes 2 block(s) per launch}}
    // expected-note @below {{in the interval before the first synchronized reset}}
    // expected-note @below {{keep pops within pushes and unpopped blocks within DFB capacity on every active node}}
    %first = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %second = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>)
    return
  }

  func.func @segmented_waits_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 4} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %pop_active = arith.cmpi ne, %flag0, %zero_i32 : i32
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>)
    return
  }

  func.func @segmented_waits_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 4} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %pop_active = arith.cmpi ne, %flag0, %zero_i32 : i32
    %block0 = ttl.cb_wait %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.if %pop_active {
      ttl.cb_pop %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    }
    %block1 = ttl.cb_wait %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.if %pop_active {
      ttl.cb_pop %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    }
    %block2 = ttl.cb_wait %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.if %pop_active {
      ttl.cb_pop %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    }
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>)
    return
  }
}

// -----

// The reset executes only when both conditions hold; when either fails the
// reader's two pushes exceed the capacity. The writer names the conjunction
// with its operands swapped.
module attributes {ttl.launch_grid = [1 : i64, 1 : i64], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compound_leak_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %first = arith.cmpi ne, %flag0, %zero_i32 : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %second = arith.cmpi ne, %flag1, %zero_i32 : i32
    %both = arith.andi %first, %second : i1
    // expected-error @below {{logical DFB 0 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
    // expected-note @below {{the producer pushes 2 block(s) per launch and the consumer pops 0, leaving 2 outstanding block(s) for capacity 1}}
    // expected-note @below {{in the interval before the first synchronized reset}}
    // expected-note @below {{keep pops within pushes and unpopped blocks within DFB capacity on every active node}}
    %first_block = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    scf.if %both {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    %second_block = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }

  func.func @compound_leak_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %first = arith.cmpi ne, %flag0, %zero_i32 : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %second = arith.cmpi ne, %flag1, %zero_i32 : i32
    %both = arith.andi %first, %second : i1
    scf.if %both {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }

  func.func @compound_leak_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %first = arith.cmpi ne, %flag0, %zero_i32 : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %second = arith.cmpi ne, %flag1, %zero_i32 : i32
    %both = arith.andi %second, %first : i1
    scf.if %both {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }
}
