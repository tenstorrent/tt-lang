// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-auto-sync),ttl-finalize-dfb-indices,ttl-verify-dfb-lifecycle)' | FileCheck %s

// Summary: Positive tests for DFB transaction order and capacity.

// A launch may finish with published blocks when their count fits the DFB
// capacity.
// CHECK-LABEL: func.func @bounded_producer_surplus
// CHECK-LABEL: func.func @full_capacity_publication
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @bounded_producer_surplus() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 40 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %slot = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }

  func.func @full_capacity_publication() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 41 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %published = ttl.cb_reserve %cb {num_tiles = 2 : i64}
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// External DFB dependencies make transaction counts opaque to the module.
// CHECK-LABEL: func.func @external_input_producer
// CHECK-LABEL: func.func @external_input_consumer
// CHECK-LABEL: func.func @external_output_consumer
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @external_input_producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 42 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      %slot = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    func.return
  }

  func.func @external_input_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 42 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.opaque_call "consume" (%cb) {header = "consume.hpp"}
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> ()
    func.return
  }

  func.func @external_output_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 43 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.opaque_call "produce" (%cb) {header = "produce.hpp"}
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> ()
    %view = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// A single kernel can reuse a capacity-one DFB when every loop iteration
// consumes its publication before the next reserve.
// CHECK-LABEL: func.func @interleaved_single_kernel
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @interleaved_single_kernel() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 44 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 5 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      %slot = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      %view = ttl.cb_wait %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    func.return
  }
}

// -----

// Lifecycle operations in a zero-trip loop contribute no transactions.
// CHECK-LABEL: func.func @zero_trip_lifecycle
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @zero_trip_lifecycle() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 46 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 0 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      %slot = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      %view = ttl.cb_wait %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    func.return
  }
}

// -----

// A loop with a runtime trip count is safe when every iteration closes all
// producer and consumer lifecycle counters.
// CHECK-LABEL: func.func @runtime_trip_closed_lifecycle
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @runtime_trip_closed_lifecycle(%upper: index) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 48 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      %slot = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      %view = ttl.cb_wait %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    func.return
  }
}

// -----

// Coordinate-selected `scf.if` results are evaluated as per-node loop bounds.
// CHECK-LABEL: func.func @if_result_bound_producer
// CHECK-LABEL: func.func @if_result_bound_consumer
module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @if_result_bound_producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 47 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %is_x0 = arith.cmpi eq, %core_x, %zero : index
    %count = scf.if %is_x0 -> index {
      scf.yield %two : index
    } else {
      scf.yield %one : index
    }
    scf.for %iteration = %zero to %count step %one {
      %slot = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    func.return
  }

  func.func @if_result_bound_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 47 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %is_x0 = arith.cmpi eq, %core_x, %zero : index
    %count = scf.if %is_x0 -> index {
      scf.yield %two : index
    } else {
      scf.yield %one : index
    }
    scf.for %iteration = %zero to %count step %one {
      %view = ttl.cb_wait %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    func.return
  }
}

// -----

// An unresolved predicate before a complete lifecycle does not make that
// lifecycle's node-specific order unknown.
// CHECK-LABEL: func.func @unknown_aggregate_domain_with_known_sequence
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @unknown_aggregate_domain_with_known_sequence(%condition: i1) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 45 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    scf.if %condition {
    } else {
    }
    %slot = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %view = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    func.return
  }
}

// -----

// A region operation without invocation bounds, such as `ttl.dst_section`, is
// active when the shared execution counts prove it executes once per parent
// execution.
// CHECK-LABEL: func.func @dst_section_producer
// CHECK-LABEL: func.func @dst_section_consumer
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @dst_section_producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 50 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      %slot = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    func.return
  }

  func.func @dst_section_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 50 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    ttl.dst_section {
      scf.for %iteration = %lower to %upper step %step {
        %view = ttl.cb_wait %cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      }
    }
    func.return
  }
}

// -----

// A sequence the execution counts cannot resolve is accepted: the trip count
// is a kernel argument and the body does not close its reserve.
// CHECK-LABEL: func.func @unresolved_trip_count_accepted
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @unresolved_trip_count_accepted(%upper: index) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 51 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      %slot = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    func.return
  }
}

// -----

// An `inspect` contract has no protocol effect. Declared `dfb_effects` count
// their pushes and pops as self-contained transfers.
// CHECK-LABEL: func.func @inspecting_producer
// CHECK-LABEL: func.func @effect_summary_consumer
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @inspecting_producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 52 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.opaque_call "inspect"
        dfb_dependencies(%cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        dfb_accesses [#ttl.dfb_non_transactional_access<inspect, 0>]
        () {header = "inspect.hpp"} : () -> ()
    %slot = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    func.return
  }

  func.func @effect_summary_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 52 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.opaque_call "consume"
        dfb_dependencies(%cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "consume.hpp"} : () -> ()
    func.return
  }
}

// -----

// A compute kernel runs producer effects on PACK and consumer effects on
// UNPACK, so holding the current state while reserving the next fits one slot.
// The same sequence in a data-movement kernel is rejected
// (`reserve_before_prior_wait_is_popped` in the invalid tests).
// CHECK-LABEL: func.func @compute_state_one_slot
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @compute_state_one_slot() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 53 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    %init = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    scf.for %iteration = %lower to %upper step %step {
      %current = ttl.cb_wait %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %next = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    %final = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    func.return
  }
}

// -----

// Multi-block transactions count in blocks: two-block acquisitions fill a
// two-block DFB exactly and a single two-block release closes them.
// CHECK-LABEL: func.func @two_block_producer
// CHECK-LABEL: func.func @two_block_consumer
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @two_block_producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 54 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      %slots = ttl.cb_reserve %cb {num_tiles = 2 : i64}
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x2x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %cb {num_tiles = 2 : i64} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    func.return
  }

  func.func @two_block_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 54 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      %blocks = ttl.cb_wait %cb {num_tiles = 2 : i64}
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x2x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %cb {num_tiles = 2 : i64} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    func.return
  }
}

// -----

// Opaque-call effects state Metal-level actions: a reserve of two blocks is a
// readiness threshold that one pushed block follows, so only the pushes count,
// each as a transfer that acquires and releases its block.
// CHECK-LABEL: func.func @threshold_reserve_producer
// CHECK-LABEL: func.func @two_block_pop_consumer
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @threshold_reserve_producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 53 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.opaque_call "produce_high_water"
        dfb_dependencies(%cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 2>, #ttl.dfb_protocol_effect<push, 0, 1>,
                     #ttl.dfb_protocol_effect<reserve, 0, 2>, #ttl.dfb_protocol_effect<push, 0, 1>]
        () {header = "produce.hpp"} : () -> ()
    func.return
  }

  func.func @two_block_pop_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 53 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %first = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %second = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    func.return
  }
}

// -----

// A synchronized reset restores the DFB to its empty state, so a block
// published before each reset is not outstanding afterwards: every interval
// between resets fits the one-block capacity on its own.
// CHECK-LABEL: func.func @reset_loop_producer
// CHECK-LABEL: func.func @reset_loop_compute
// CHECK-LABEL: func.func @reset_loop_writer
module attributes {ttl.launch_grid = [1 : i64, 1 : i64], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @reset_loop_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    scf.for %iteration = %zero to %four step %one {
      %slot = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>
    }
    return
  }

  func.func @reset_loop_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    scf.for %iteration = %zero to %four step %one {
      ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>
    }
    return
  }

  func.func @reset_loop_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    scf.for %iteration = %zero to %four step %one {
      ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>
    }
    return
  }
}

// -----

// An opaque push or pop closes a user acquisition of its kind that is still
// open, as automatic synchronization pairs them.
// CHECK-LABEL: func.func @user_reserve_opaque_push
// CHECK-LABEL: func.func @user_wait_opaque_pop
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @user_reserve_opaque_push() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 57 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reserved = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.opaque_call "publish"
        dfb_dependencies(%cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<push, 0, 1>]
        () {header = "effects.hpp"} : () -> ()
    func.return
  }

  func.func @user_wait_opaque_pop() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 57 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %waited = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.opaque_call "release"
        dfb_dependencies(%cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "effects.hpp"} : () -> ()
    func.return
  }
}

// -----

// When the pops are conditional, the consumer's exact wait count stands in for
// them, but an opaque wait is a threshold and is not counted.
// CHECK-LABEL: func.func @two_push_producer
// CHECK-LABEL: func.func @threshold_wait_then_user_wait
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @two_push_producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 58 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    scf.for %iteration = %zero to %two step %one {
      %slot = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    func.return
  }

  func.func @threshold_wait_then_user_wait(%condition: i1) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 58 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "consume_one_of_two"
        dfb_dependencies(%cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 2>, #ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "effects.hpp"} : () -> ()
    %waited = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.if %condition {
      ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    } else {
      ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    func.return
  }
}

// -----

// A reset under a dispatch condition: one block is published and consumed in
// every interval whether or not the reset executes.
// CHECK-LABEL: func.func @reader
// CHECK-LABEL: func.func @compute
// CHECK-LABEL: func.func @writer
module attributes {ttl.launch_grid = [1 : i64, 1 : i64], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %flag = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %zero_i32 = arith.constant 0 : i32
    %active = arith.cmpi ne, %flag, %zero_i32 : i32
    %slot = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    scf.if %active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    %later = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
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
    %block = ttl.cb_wait %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    scf.if %active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    %later = ttl.cb_wait %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}

// -----

// An opaque push after a reset under a dispatch condition closes the user
// reserve in the alternative where the reset is skipped and is a transfer of
// its own where the reset executed; both alternatives balance.
// CHECK-LABEL: func.func @reader
// CHECK-LABEL: func.func @compute
// CHECK-LABEL: func.func @writer
module attributes {ttl.launch_grid = [1 : i64, 1 : i64], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %active0 = arith.cmpi ne, %flag0, %zero_i32 : i32
    %slot = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.if %active0 {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    ttl.opaque_call "publish"
        dfb_dependencies(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        dfb_effects [#ttl.dfb_protocol_effect<push, 0, 1>]
        () {header = "effects.hpp"} : () -> ()
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
    scf.if %active0 {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
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
    scf.if %active0 {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    %block0 = ttl.cb_wait %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}

// -----

// Two resets under one dispatch condition decide together: the writer pops
// only when neither reset executes, which matches the reader's single push.
// CHECK-LABEL: func.func @same_condition_reader
// CHECK-LABEL: func.func @same_condition_compute
// CHECK-LABEL: func.func @same_condition_writer
module attributes {ttl.launch_grid = [1 : i64, 1 : i64], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @same_condition_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %first = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
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

  func.func @same_condition_compute()
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

  func.func @same_condition_writer()
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
    } else {
      %block = ttl.cb_wait %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    return
  }
}

// -----

// The two resets are under complementary conditions, so exactly one
// executes and each interval carries one push into the single slot.
// CHECK-LABEL: func.func @complementary_reader
// CHECK-LABEL: func.func @complementary_compute
// CHECK-LABEL: func.func @complementary_writer
module attributes {ttl.launch_grid = [1 : i64, 1 : i64], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @complementary_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %active = arith.cmpi ne, %flag0, %zero_i32 : i32
    %inactive = arith.cmpi eq, %flag0, %zero_i32 : i32
    %first = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    scf.if %active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    scf.if %inactive {
      ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    %second = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }

  func.func @complementary_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %active = arith.cmpi ne, %flag0, %zero_i32 : i32
    %inactive = arith.cmpi eq, %flag0, %zero_i32 : i32
    scf.if %active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    scf.if %inactive {
      ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }

  func.func @complementary_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %active = arith.cmpi ne, %flag0, %zero_i32 : i32
    %inactive = arith.cmpi eq, %flag0, %zero_i32 : i32
    scf.if %active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    scf.if %inactive {
      ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }
}

// -----

// A conjunction and its comparison with zero are complementary, so exactly
// one reset executes.
// CHECK-LABEL: func.func @compound_complement_reader
// CHECK-LABEL: func.func @compound_complement_compute
// CHECK-LABEL: func.func @compound_complement_writer
module attributes {ttl.launch_grid = [1 : i64, 1 : i64], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compound_complement_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %first = arith.cmpi ne, %flag0, %zero_i32 : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %second = arith.cmpi ne, %flag1, %zero_i32 : i32
    %both = arith.andi %first, %second : i1
    %false = arith.constant false
    %neither = arith.cmpi eq, %both, %false : i1
    %first_block = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    scf.if %both {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    scf.if %neither {
      ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    %second_block = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }

  func.func @compound_complement_compute()
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
    %false = arith.constant false
    %neither = arith.cmpi eq, %both, %false : i1
    scf.if %both {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    scf.if %neither {
      ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }

  func.func @compound_complement_writer()
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
    %both = arith.andi %first, %second : i1
    %false = arith.constant false
    %neither = arith.cmpi eq, %both, %false : i1
    scf.if %both {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    scf.if %neither {
      ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }
}

// -----

// A disjunction and the conjunction of the negated operands are
// complementary, so exactly one reset executes.
// CHECK-LABEL: func.func @demorgan_complement_reader
// CHECK-LABEL: func.func @demorgan_complement_compute
// CHECK-LABEL: func.func @demorgan_complement_writer
module attributes {ttl.launch_grid = [1 : i64, 1 : i64], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @demorgan_complement_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %first = arith.cmpi ne, %flag0, %zero_i32 : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %second = arith.cmpi ne, %flag1, %zero_i32 : i32
    %either = arith.ori %first, %second : i1
    %not_first = arith.cmpi eq, %flag0, %zero_i32 : i32
    %not_second = arith.cmpi eq, %flag1, %zero_i32 : i32
    %neither = arith.andi %not_first, %not_second : i1
    %first_block = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    scf.if %either {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    scf.if %neither {
      ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    %second_block = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }

  func.func @demorgan_complement_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %first = arith.cmpi ne, %flag0, %zero_i32 : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %second = arith.cmpi ne, %flag1, %zero_i32 : i32
    %either = arith.ori %first, %second : i1
    %not_first = arith.cmpi eq, %flag0, %zero_i32 : i32
    %not_second = arith.cmpi eq, %flag1, %zero_i32 : i32
    %neither = arith.andi %not_first, %not_second : i1
    scf.if %either {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    scf.if %neither {
      ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }

  func.func @demorgan_complement_writer()
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
    %either = arith.ori %first, %second : i1
    %not_first = arith.cmpi eq, %flag0, %zero_i32 : i32
    %not_second = arith.cmpi eq, %flag1, %zero_i32 : i32
    %neither = arith.andi %not_first, %not_second : i1
    scf.if %either {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    scf.if %neither {
      ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }
}

// -----

// Conjunctions with operands in either order are one condition, so the
// writer pops exactly when neither reset executes.
// CHECK-LABEL: func.func @commuted_conjunction_reader
// CHECK-LABEL: func.func @commuted_conjunction_compute
// CHECK-LABEL: func.func @commuted_conjunction_writer
module attributes {ttl.launch_grid = [1 : i64, 1 : i64], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @commuted_conjunction_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %first = arith.cmpi ne, %flag0, %zero_i32 : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %second = arith.cmpi ne, %flag1, %zero_i32 : i32
    %both = arith.andi %first, %second : i1
    %both_again = arith.andi %second, %first : i1
    %block = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    scf.if %both {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    scf.if %both_again {
      ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }

  func.func @commuted_conjunction_compute()
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
    %both_again = arith.andi %second, %first : i1
    scf.if %both {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    scf.if %both_again {
      ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }

  func.func @commuted_conjunction_writer()
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
    %both = arith.andi %first, %second : i1
    %both_again = arith.andi %second, %first : i1
    scf.if %both {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    scf.if %both_again {
      ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    } else {
      %block = ttl.cb_wait %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    return
  }
}

// -----

// Every branch pushes once and pops once. The branches without the reset
// share one sequence, and their merged alternative executes under either
// branch condition, so it never pairs with the reset branch of another
// kernel.
// CHECK-LABEL: func.func @merged_alternatives_reader
// CHECK-LABEL: func.func @merged_alternatives_compute
// CHECK-LABEL: func.func @merged_alternatives_writer
module attributes {ttl.launch_grid = [1 : i64, 1 : i64], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @merged_alternatives_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %reset_active = arith.cmpi ne, %flag0, %zero_i32 : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %outer = arith.cmpi ne, %flag1, %zero_i32 : i32
    %flag2 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<2, i32>, header = "predicate.hpp"} : () -> i32
    %inner = arith.cmpi ne, %flag2, %zero_i32 : i32
    scf.if %outer {
      %slot_6 = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    } else {
      scf.if %inner {
        %slot_8 = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      } else {
        %slot_9 = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        scf.if %reset_active {
          ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        }
        %slot_10 = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      }
    }
    return
  }

  func.func @merged_alternatives_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %reset_active = arith.cmpi ne, %flag0, %zero_i32 : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %outer = arith.cmpi ne, %flag1, %zero_i32 : i32
    %flag2 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<2, i32>, header = "predicate.hpp"} : () -> i32
    %inner = arith.cmpi ne, %flag2, %zero_i32 : i32
    scf.if %outer {
    } else {
      scf.if %inner {
      } else {
        scf.if %reset_active {
          ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        }
      }
    }
    return
  }

  func.func @merged_alternatives_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero_i32 = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %reset_active = arith.cmpi ne, %flag0, %zero_i32 : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %outer = arith.cmpi ne, %flag1, %zero_i32 : i32
    %flag2 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<2, i32>, header = "predicate.hpp"} : () -> i32
    %inner = arith.cmpi ne, %flag2, %zero_i32 : i32
    scf.if %outer {
      %block_6 = ttl.cb_wait %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    } else {
      scf.if %inner {
        %block_8 = ttl.cb_wait %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_pop %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      } else {
        %block_9 = ttl.cb_wait %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_pop %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        scf.if %reset_active {
          ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        }
        %block_10 = ttl.cb_wait %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_pop %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      }
    }
    return
  }
}

// -----

// A user reservation of one block followed by an opaque push of two blocks
// completes that block and publishes one more, so two blocks are open at
// most and fit the capacity.
// CHECK-LABEL: func.func @user_reserve_wider_opaque_push
// CHECK-LABEL: func.func @two_waits_two_pops
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @user_reserve_wider_opaque_push() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 58 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reserved = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.opaque_call "publish_two"
        dfb_dependencies(%cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 2>, #ttl.dfb_protocol_effect<push, 0, 2>]
        () {header = "effects.hpp"} : () -> ()
    func.return
  }

  func.func @two_waits_two_pops() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 58 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }
}

// -----

// The writer pops through an opaque call on both branches, so no user wait
// can stand in for the conditional pops; the one push left unpopped fits
// the capacity.
// CHECK-LABEL: func.func @opaque_conditional_pop_reader
// CHECK-LABEL: func.func @opaque_conditional_pop_compute
// CHECK-LABEL: func.func @opaque_conditional_pop_writer
module attributes {ttl.launch_grid = [1 : i64, 1 : i64], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @opaque_conditional_pop_reader() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "ow">, ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %d = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero = arith.constant 0 : i32
    %f = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %c = arith.cmpi ne, %f, %zero : i32
    %r0 = ttl.cb_reserve %d : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %d : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %r1 = ttl.cb_reserve %d : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %d : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
  func.func @opaque_conditional_pop_compute() attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "ow">, ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %d = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero = arith.constant 0 : i32
    %f = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %c = arith.cmpi ne, %f, %zero : i32
    return
  }
  func.func @opaque_conditional_pop_writer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "ow">, ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %d = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero = arith.constant 0 : i32
    %f = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %c = arith.cmpi ne, %f, %zero : i32
    scf.if %c {
      ttl.opaque_call "consume" dfb_dependencies(%d : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) dfb_effects [#ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "consume.hpp"} : () -> ()
    } else {
      ttl.opaque_call "consume" dfb_dependencies(%d : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) dfb_effects [#ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "consume.hpp"} : () -> ()
    }
    return
  }
}

// -----

// A reset under a condition that is always true leaves no skipped path, so
// the path condition of the skipped alternative is unsatisfiable and it is
// dropped instead of joining transactions across the reset.
// CHECK-LABEL: func.func @tautology_fold_reader
// CHECK-LABEL: func.func @tautology_fold_compute
// CHECK-LABEL: func.func @tautology_fold_writer
module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @tautology_fold_reader() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "cap">, ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %c0 = arith.cmpi ne, %flag0, %zero : i32
    %z0 = arith.cmpi eq, %flag0, %zero : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %c1 = arith.cmpi ne, %flag1, %zero : i32
    %z1 = arith.cmpi eq, %flag1, %zero : i32
    %e1 = arith.ori %c0, %z0 : i1
    %ra = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    scf.if %e1 {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "cap">, <kind = data_movement, identity = "reader", operation = "cap">, <kind = data_movement, identity = "writer", operation = "cap">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    %rb = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
  func.func @tautology_fold_compute() attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "cap">, ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %c0 = arith.cmpi ne, %flag0, %zero : i32
    %z0 = arith.cmpi eq, %flag0, %zero : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %c1 = arith.cmpi ne, %flag1, %zero : i32
    %z1 = arith.cmpi eq, %flag1, %zero : i32
    %e1 = arith.ori %c0, %z0 : i1
    scf.if %e1 {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "cap">, <kind = data_movement, identity = "reader", operation = "cap">, <kind = data_movement, identity = "writer", operation = "cap">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }
  func.func @tautology_fold_writer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "cap">, ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero = arith.constant 0 : i32
    %flag0 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<0, i32>, header = "predicate.hpp"} : () -> i32
    %c0 = arith.cmpi ne, %flag0, %zero : i32
    %z0 = arith.cmpi eq, %flag0, %zero : i32
    %flag1 = ttl.opaque_call "scalar_predicate" template_args [#ttl.external_template_arg<signed_integer, 1>] () {condition_result = #ttl.dispatch_condition<1, i32>, header = "predicate.hpp"} : () -> i32
    %c1 = arith.cmpi ne, %flag1, %zero : i32
    %z1 = arith.cmpi eq, %flag1, %zero : i32
    %e1 = arith.ori %c0, %z0 : i1
    scf.if %e1 {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "cap">, <kind = data_movement, identity = "reader", operation = "cap">, <kind = data_movement, identity = "writer", operation = "cap">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }
}
