// Tests invalid synchronized dataflow-buffer reset contracts.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})'

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @missing_participant()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{'ttl.reset_dfbs' op synchronized DFB reset is missing a declared logical-kernel participant}}
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }
}

// -----

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @target_mismatch_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }

  func.func @target_mismatch_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32} {
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{'ttl.reset_dfbs' op synchronized DFB reset participants must declare identical target sets}}
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }

  func.func @target_mismatch_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }
}

// -----

// A reset under a dispatch condition executes a runtime-dependent number of
// times, so the DFB state across it cannot be checked.
module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @runtime_condition_reset_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %zero = arith.constant 0 : i64
    %value = ttl.opaque_call "compute_active" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
    %active = arith.cmpi ne, %value, %zero : i64
    scf.if %active {
      // expected-error @below {{'ttl.reset_dfbs' op synchronized DFB reset must execute a compile-time-known number of times on each launch node; it may depend on the launch node but not on runtime values}}
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    }
    return
  }

  func.func @runtime_condition_reset_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %zero = arith.constant 0 : i64
    %value = ttl.opaque_call "reader_active" () {condition_result = #ttl.dispatch_condition<1, i64>, header = "condition.hpp"} : () -> i64
    %active = arith.cmpi ne, %value, %zero : i64
    scf.if %active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    }
    return
  }

  func.func @runtime_condition_reset_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %zero = arith.constant 0 : i64
    %value = ttl.opaque_call "writer_active" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
    %active = arith.cmpi ne, %value, %zero : i64
    scf.if %active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    }
    return
  }
}

// -----

module {
  func.func @empty_reset_target_set()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">} {
    // expected-error @below {{'ttl.reset_dfbs' op requires at least one DFB}}
    "ttl.reset_dfbs"() {reset = #ttl.synchronized_dfb_reset<0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>} : () -> ()
    return
  }
}

// -----

module {
  func.func @duplicate_reset_target()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{'ttl.reset_dfbs' op DFBs must be distinct}}
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb, %dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }
}

// -----

module {
  func.func @invalid_participant_set() attributes {
    // expected-error @below {{synchronized DFB reset participants must contain one compute kernel and two data movement kernels}}
    reset = #ttl.synchronized_dfb_reset<0, participants[<kind = compute>]>
  } {
    return
  }
}

// -----

module attributes {ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  func.func @unsupported_reset_target()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // expected-error @below {{'ttl.reset_dfbs' op is supported only for Blackhole; selected target is #ttcore.arch<wormhole_b0>}}
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    return
  }
}

// -----

// A conditional reset cannot define one reset instance per loop interval.

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @conditional_repeated_reset_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "conditional_repeated_reset">} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    scf.for %iteration = %lower to %upper step %step {
      %value = ttl.opaque_call "compute_active" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
      %active = arith.cmpi ne, %value, %zero : i64
      scf.if %active {
        // expected-error @below {{'ttl.reset_dfbs' op synchronized DFB reset must execute a compile-time-known number of times on each launch node; it may depend on the launch node but not on runtime values}}
        ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "conditional_repeated_reset">, <kind = data_movement, identity = "reader", operation = "conditional_repeated_reset">, <kind = data_movement, identity = "writer", operation = "conditional_repeated_reset">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      }
    }
    return
  }

  func.func @conditional_repeated_reset_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "conditional_repeated_reset">,
                  ttl.noc_index = 0 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    scf.for %iteration = %lower to %upper step %step {
      %value = ttl.opaque_call "reader_active" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
      %active = arith.cmpi ne, %value, %zero : i64
      scf.if %active {
        ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "conditional_repeated_reset">, <kind = data_movement, identity = "reader", operation = "conditional_repeated_reset">, <kind = data_movement, identity = "writer", operation = "conditional_repeated_reset">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      }
    }
    return
  }

  func.func @conditional_repeated_reset_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "conditional_repeated_reset">,
                  ttl.noc_index = 1 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    scf.for %iteration = %lower to %upper step %step {
      %value = ttl.opaque_call "writer_active" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
      %active = arith.cmpi ne, %value, %zero : i64
      scf.if %active {
        ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "conditional_repeated_reset">, <kind = data_movement, identity = "reader", operation = "conditional_repeated_reset">, <kind = data_movement, identity = "writer", operation = "conditional_repeated_reset">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      }
    }
    return
  }
}

// -----

// All participants must execute the same number of reset instances.

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @count_mismatch_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "count_mismatch">} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "count_mismatch">, <kind = data_movement, identity = "reader", operation = "count_mismatch">, <kind = data_movement, identity = "writer", operation = "count_mismatch">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }

  func.func @count_mismatch_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "count_mismatch">,
                  ttl.noc_index = 0 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "count_mismatch">, <kind = data_movement, identity = "reader", operation = "count_mismatch">, <kind = data_movement, identity = "writer", operation = "count_mismatch">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }

  func.func @count_mismatch_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "count_mismatch">,
                  ttl.noc_index = 1 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      // expected-error @below {{'ttl.reset_dfbs' op synchronized DFB reset participants must execute in the same structured iteration sequence}}
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "count_mismatch">, <kind = data_movement, identity = "reader", operation = "count_mismatch">, <kind = data_movement, identity = "writer", operation = "count_mismatch">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }
}

// -----

// Equal total counts do not match different nested iteration sequences.

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @sequence_mismatch_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "sequence_mismatch">} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "sequence_mismatch">, <kind = data_movement, identity = "reader", operation = "sequence_mismatch">, <kind = data_movement, identity = "writer", operation = "sequence_mismatch">]>
    }
    return
  }

  func.func @sequence_mismatch_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "sequence_mismatch">,
                  ttl.noc_index = 0 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "sequence_mismatch">, <kind = data_movement, identity = "reader", operation = "sequence_mismatch">, <kind = data_movement, identity = "writer", operation = "sequence_mismatch">]>
    }
    return
  }

  func.func @sequence_mismatch_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "sequence_mismatch">,
                  ttl.noc_index = 1 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %outer = %lower to %upper step %step {
      scf.for %inner = %lower to %upper step %step {
        // expected-error @below {{'ttl.reset_all_dfbs' op repeated synchronized DFB reset must execute in one loop, not in nested loops}}
        ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "sequence_mismatch">, <kind = data_movement, identity = "reader", operation = "sequence_mismatch">, <kind = data_movement, identity = "writer", operation = "sequence_mismatch">]>
      }
    }
    return
  }
}

// -----

// Parallel iterations do not have an exact sequential reset count.

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @parallel_reset_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "parallel_reset">} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.parallel (%iteration) = (%lower) to (%upper) step (%step) {
      // expected-error @below {{'ttl.reset_all_dfbs' op synchronized DFB reset must execute a compile-time-known number of times on each launch node; it may depend on the launch node but not on runtime values}}
      ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "parallel_reset">, <kind = data_movement, identity = "reader", operation = "parallel_reset">, <kind = data_movement, identity = "writer", operation = "parallel_reset">]>
    }
    return
  }

  func.func @parallel_reset_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "parallel_reset">,
                  ttl.noc_index = 0 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.parallel (%iteration) = (%lower) to (%upper) step (%step) {
      ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "parallel_reset">, <kind = data_movement, identity = "reader", operation = "parallel_reset">, <kind = data_movement, identity = "writer", operation = "parallel_reset">]>
    }
    return
  }

  func.func @parallel_reset_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "parallel_reset">,
                  ttl.noc_index = 1 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.parallel (%iteration) = (%lower) to (%upper) step (%step) {
      ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "parallel_reset">, <kind = data_movement, identity = "reader", operation = "parallel_reset">, <kind = data_movement, identity = "writer", operation = "parallel_reset">]>
    }
    return
  }
}
