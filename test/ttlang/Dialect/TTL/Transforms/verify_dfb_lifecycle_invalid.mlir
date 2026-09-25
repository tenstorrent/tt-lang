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
      // expected-note @below {{keep waits within visible publications and outstanding publications within DFB capacity on every active node}}
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
        // expected-note @below {{keep waits within visible publications and outstanding publications within DFB capacity on every active node}}
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
        // expected-note @below {{keep waits within visible publications and outstanding publications within DFB capacity on every active node}}
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
      // expected-note @below {{keep waits within visible publications and outstanding publications within DFB capacity on every active node}}
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
    // expected-note @below {{keep waits within visible publications and outstanding publications within DFB capacity on every active node}}
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
