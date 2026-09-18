// Summary: Tests synchronized dataflow-buffer reset epochs.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s

// A collective reset terminates a producer-only lifecycle and orders the next
// complete lifecycle in different logical kernels.
// CHECK: DFB logical_id=0 bounded=1
// CHECK: epochs=[{accesses=[0, 1],transactions=[1],write_cursor_runs=[1],read_cursor_runs=[],write_owner=(0,0):noc0:write,read_owner=unknown,entry_reconfiguration=initial,active_configurations=[initial],terminal_reset=0,terminal_reconfiguration=none,terminal_state=canonical}]
// CHECK: DFB logical_id=1 bounded=1
// CHECK: Total DFB count: 1
// CHECK: DFB assignment: logical DFB 0 -> physical index 0 storage index 0 (bounded)
// CHECK: DFB assignment: logical DFB 1 -> physical index 0 storage index 0 (bounded)

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @unconditional_reset_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %current = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %old_slot = ttl.cb_reserve %old : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %old : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    %current_slot = ttl.cb_reserve %current : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %current : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @unconditional_reset_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %current = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    %current_slot = ttl.cb_wait %current : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %current : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @unconditional_reset_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    return
  }
}

// -----

// One selected reset declaration may execute once per iteration of equivalent
// immutable sequential loops in all three participants.

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @repeated_selected_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "repeated_selected">} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "repeated_selected">, <kind = data_movement, identity = "reader", operation = "repeated_selected">, <kind = data_movement, identity = "writer", operation = "repeated_selected">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }

  func.func @repeated_selected_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "repeated_selected">,
                  ttl.noc_index = 0 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    affine.for %iteration = 0 to 4 {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "repeated_selected">, <kind = data_movement, identity = "reader", operation = "repeated_selected">, <kind = data_movement, identity = "writer", operation = "repeated_selected">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }

  func.func @repeated_selected_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "repeated_selected">,
                  ttl.noc_index = 1 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "repeated_selected">, <kind = data_movement, identity = "reader", operation = "repeated_selected">, <kind = data_movement, identity = "writer", operation = "repeated_selected">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }
}

// -----

// Repeated all-interface resets use the same participant-loop contract.

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @repeated_all_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "repeated_all">} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "repeated_all">, <kind = data_movement, identity = "reader", operation = "repeated_all">, <kind = data_movement, identity = "writer", operation = "repeated_all">]>
    }
    return
  }

  func.func @repeated_all_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "repeated_all">,
                  ttl.noc_index = 0 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "repeated_all">, <kind = data_movement, identity = "reader", operation = "repeated_all">, <kind = data_movement, identity = "writer", operation = "repeated_all">]>
    }
    return
  }

  func.func @repeated_all_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "repeated_all">,
                  ttl.noc_index = 1 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "repeated_all">, <kind = data_movement, identity = "reader", operation = "repeated_all">, <kind = data_movement, identity = "writer", operation = "repeated_all">]>
    }
    return
  }
}

// -----

// A selected reset partitions tensor-backed interface state. Payload bytes
// remain allocated, but reset occupancy makes pre-reset payload unavailable.
// CHECK: DFB logical_id=0 bounded=0
// CHECK: lifecycle_completion=missing-protocol-effect
// CHECK: epochs=[{accesses=[0, 1],transactions=[1]

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @tensor_crossing_reset_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 3}
        {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 6144>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %slot = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    return
  }

  func.func @tensor_crossing_reset_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 3}
        {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 6144>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    %slot = ttl.cb_wait %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @tensor_crossing_reset_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 3}
        {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 6144>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    return
  }
}

// -----

// Four two-tile transactions fit contiguously in a nine-tile DFB. The reset
// canonicalizes their safe nonzero terminal pointer offset.
// CHECK: DFB logical_id=0 bounded=1
// CHECK: transactions=[2, 2, 2, 2]
// CHECK-SAME: terminal_reset=0,terminal_reconfiguration=none,terminal_state=canonical

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @safe_nondividing_run_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 9} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %transaction = %lower to %upper step %step {
      ttl.opaque_call "produce_two" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 2>, #ttl.dfb_protocol_effect<push, 0, 2>] () {header = "producer.hpp"} : () -> ()
    }
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>)
    return
  }

  func.func @safe_nondividing_run_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 9} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %transaction = %lower to %upper step %step {
      ttl.opaque_call "consume_two" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 2>, #ttl.dfb_protocol_effect<pop, 0, 2>] () {header = "consumer.hpp"} : () -> ()
    }
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>)
    return
  }

  func.func @safe_nondividing_run_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 9} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>)
    return
  }
}

// -----

// A fifth two-tile acquire would start at offset eight and cross the end of a
// nine-tile DFB. A later reset cannot make that transaction contiguous.
// CHECK: DFB logical_id=0 bounded=0
// CHECK: lifecycle_completion=mismatched-transaction

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @straddling_nondividing_run_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 9} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>
    %lower = arith.constant 0 : index
    %upper = arith.constant 5 : index
    %step = arith.constant 1 : index
    scf.for %transaction = %lower to %upper step %step {
      ttl.opaque_call "produce_two" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 2>, #ttl.dfb_protocol_effect<push, 0, 2>] () {header = "producer.hpp"} : () -> ()
    }
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>)
    return
  }

  func.func @straddling_nondividing_run_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 9} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>
    %lower = arith.constant 0 : index
    %upper = arith.constant 5 : index
    %step = arith.constant 1 : index
    scf.for %transaction = %lower to %upper step %step {
      ttl.opaque_call "consume_two" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 2>, #ttl.dfb_protocol_effect<pop, 0, 2>] () {header = "consumer.hpp"} : () -> ()
    }
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>)
    return
  }

  func.func @straddling_nondividing_run_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 9} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>)
    return
  }
}

// -----

// A conditional reset cannot order unconditional accesses across logical
// kernels because the synchronization does not execute on the disabled branch.
// CHECK: DFB logical_id=0 bounded=0
// CHECK: lifecycle_completion=unsupported-control-flow
// CHECK: DFB logical_id=1 bounded=0
// CHECK: lifecycle_completion=unsupported-control-flow
// CHECK: Total DFB count: 2

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @independent_conditional_reset_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %following = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %old_slot = ttl.cb_reserve %old : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %old : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %zero = arith.constant 0 : i64
    %value = ttl.opaque_call "active" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
    %active = arith.cmpi ne, %value, %zero : i64
    scf.if %active {
      ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>
    }
    %following_slot = ttl.cb_reserve %following : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %following : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @independent_conditional_reset_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %following = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %old_slot = ttl.cb_wait %old : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %old : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %zero = arith.constant 0 : i64
    %value = ttl.opaque_call "active_again" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
    %active = arith.cmpi ne, %value, %zero : i64
    scf.if %active {
      ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>
    }
    %following_slot = ttl.cb_wait %following : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %following : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @independent_conditional_reset_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %zero = arith.constant 0 : i64
    %value = ttl.opaque_call "active_for_writer" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
    %active = arith.cmpi ne, %value, %zero : i64
    scf.if %active {
      ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>
    }
    return
  }
}

// -----

// A reset makes a complete two-tile transaction canonical even when its
// pointer movement does not divide the nine-tile descriptor capacity.
// CHECK: DFB logical_id=0 bounded=1
// CHECK: epochs=[{accesses=[0, 1, 2, 3],transactions=[2],write_owner=(0,0):noc0:write,read_owner=(0,0):unpack:read,entry_reconfiguration=initial,active_configurations=[initial],terminal_reset=0,terminal_reconfiguration=none,terminal_state=canonical}]
// CHECK: DFB logical_id=1 bounded=1
// CHECK: Total DFB count: 1

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @nondividing_reset_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 9} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>
    %current = ttl.bind_cb {cb_index = 1, block_count = 9} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>
    ttl.opaque_call "produce_two" dfb_dependencies(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 2>, #ttl.dfb_protocol_effect<push, 0, 2>] () {header = "producer.hpp"} : () -> ()
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>)
    %current_slot = ttl.cb_reserve %current : <[1, 1], !ttcore.tile<32x32, bf16>, 9> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %current : <[1, 1], !ttcore.tile<32x32, bf16>, 9>
    return
  }

  func.func @nondividing_reset_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 9} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>
    %current = ttl.bind_cb {cb_index = 1, block_count = 9} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>
    ttl.opaque_call "consume_two" dfb_dependencies(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 2>, #ttl.dfb_protocol_effect<pop, 0, 2>] () {header = "consumer.hpp"} : () -> ()
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>)
    %current_slot = ttl.cb_wait %current : <[1, 1], !ttcore.tile<32x32, bf16>, 9> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %current : <[1, 1], !ttcore.tile<32x32, bf16>, 9>
    return
  }

  func.func @nondividing_reset_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 9} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>)
    return
  }
}

// -----

// An all-local reset partitions nested logical DFB lifecycles that cannot be
// named at the reset call site.
// CHECK: DFB logical_id=0 bounded=1
// CHECK: epochs=[{accesses=[0, 1, 2, 3],transactions=[2],write_owner=(0,0):noc0:write,read_owner=(0,0):unpack:read,entry_reconfiguration=initial,active_configurations=[initial],terminal_reset=0,terminal_reconfiguration=none,terminal_state=canonical}]
// CHECK: DFB logical_id=1 bounded=1
// CHECK: Total DFB count: 1

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @all_dfbs_reset_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 9} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>
    %current = ttl.bind_cb {cb_index = 1, block_count = 9} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>
    ttl.opaque_call "produce_two" dfb_dependencies(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 2>, #ttl.dfb_protocol_effect<push, 0, 2>] () {header = "producer.hpp"} : () -> ()
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>
    %current_slot = ttl.cb_reserve %current : <[1, 1], !ttcore.tile<32x32, bf16>, 9> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %current : <[1, 1], !ttcore.tile<32x32, bf16>, 9>
    return
  }

  func.func @all_dfbs_reset_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 9} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>
    %current = ttl.bind_cb {cb_index = 1, block_count = 9} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>
    ttl.opaque_call "consume_two" dfb_dependencies(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 9>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 2>, #ttl.dfb_protocol_effect<pop, 0, 2>] () {header = "consumer.hpp"} : () -> ()
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>
    %current_slot = ttl.cb_wait %current : <[1, 1], !ttcore.tile<32x32, bf16>, 9> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %current : <[1, 1], !ttcore.tile<32x32, bf16>, 9>
    return
  }

  func.func @all_dfbs_reset_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>
    return
  }
}

// -----

// A payload access after reset belongs to a new epoch and cannot consume the
// preceding epoch's produced data.
// CHECK: DFB logical_id=0 bounded=0
// CHECK: lifecycle_completion=incomplete-use-order{{.*}}evidence=ttl.cb_push

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @payload_crosses_reset()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 1 : i32,
                  ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %slot = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    ttl.opaque_call "late_payload_use" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) () {header = "use.hpp"} : () -> ()
    return
  }

  func.func @payload_crosses_reset_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    return
  }

  func.func @payload_crosses_reset_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 1 : i32,
                  ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    return
  }
}

// -----

// Access in a logical kernel outside the participant set is unordered with
// the reset and leaves the complete lifecycle conservative.
// CHECK: DFB logical_id=0 bounded=0
// CHECK: lifecycle_completion=incomplete-use-order

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @concurrent_access_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 1 : i32,
                  ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %slot = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    return
  }

  func.func @concurrent_access_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "outside_compute", operation = "outside_test">,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %slot = ttl.cb_wait %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @concurrent_access_reset_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    return
  }

  func.func @concurrent_access_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 1 : i32,
                  ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    return
  }
}

// -----

// Multiple ordered resets partition one logical DFB into multiple producer
// epochs. The final reset establishes canonical state for the next lifecycle.
// CHECK: DFB logical_id=0 bounded=1
// CHECK: epochs=[{accesses=[0, 1],transactions=[1],write_cursor_runs=[1],read_cursor_runs=[],write_owner=(0,0):noc0:write,read_owner=unknown,entry_reconfiguration=initial,active_configurations=[initial],terminal_reset=0,terminal_reconfiguration=none,terminal_state=canonical}, {accesses=[2, 3],transactions=[1],write_cursor_runs=[1],read_cursor_runs=[],write_owner=(0,0):noc0:write,read_owner=unknown,entry_reconfiguration=initial,active_configurations=[initial],terminal_reset=1,terminal_reconfiguration=none,terminal_state=canonical}]
// CHECK: DFB logical_id=1 bounded=1
// CHECK: Total DFB count: 1

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @multiple_reset_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %epochs = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %following = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %first = ttl.cb_reserve %epochs : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %epochs : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%epochs : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    %second = ttl.cb_reserve %epochs : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %epochs : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%epochs : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    %next = ttl.cb_reserve %following : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %following : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @multiple_reset_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %epochs = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %following = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%epochs : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%epochs : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    %next = ttl.cb_wait %following : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %following : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @multiple_reset_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %epochs = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%epochs : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    ttl.reset_dfbs <1, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%epochs : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    return
  }
}

// -----

// Equal typed condition identities prove that every participant and every
// preceding payload effect executes in the same conditional reset instance.
// CHECK: DFB logical_id=0 bounded=1
// CHECK: conditional_execution=1
// CHECK: terminal_reset=0,terminal_reconfiguration=none,terminal_state=canonical
// CHECK: DFB logical_id=1 bounded=1
// CHECK: Total DFB count: 1

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @conditional_reset_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %following = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %zero = arith.constant 0 : i64
    %produce_value = ttl.opaque_call "active" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
    %produce_active = arith.cmpi ne, %produce_value, %zero : i64
    scf.if %produce_active {
      %slot = ttl.cb_reserve %old : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %old : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    }
    %reset_value = ttl.opaque_call "active_again" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
    %reset_active = arith.cmpi ne, %reset_value, %zero : i64
    scf.if %reset_active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    }
    %next = ttl.cb_reserve %following : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %following : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @conditional_reset_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %following = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %zero = arith.constant 0 : i64
    %reset_value = ttl.opaque_call "active_for_compute" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
    %reset_active = arith.cmpi ne, %reset_value, %zero : i64
    scf.if %reset_active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    }
    %next = ttl.cb_wait %following : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %following : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @conditional_reset_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %zero = arith.constant 0 : i64
    %reset_value = ttl.opaque_call "active_for_writer" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
    %reset_active = arith.cmpi ne, %reset_value, %zero : i64
    scf.if %reset_active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    }
    return
  }
}

// -----

// Equal nested condition identities order accesses and the reset within one
// structured conditional operation in each participant.
// CHECK: DFB logical_id=0 bounded=1
// CHECK: conditional_execution=1
// CHECK: terminal_reset=0,terminal_reconfiguration=none,terminal_state=canonical
// CHECK: DFB logical_id=1 bounded=1
// CHECK: Total DFB count: 1

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @nested_conditional_reset_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %following = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %zero = arith.constant 0 : i64
    %outer_value = ttl.opaque_call "outer_active" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
    %outer_active = arith.cmpi ne, %outer_value, %zero : i64
    scf.if %outer_active {
      %inner_value = ttl.opaque_call "inner_active" () {condition_result = #ttl.dispatch_condition<1, i64>, header = "condition.hpp"} : () -> i64
      %inner_active = arith.cmpi ne, %inner_value, %zero : i64
      scf.if %inner_active {
        %slot = ttl.cb_reserve %old : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_push %old : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
        ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
      }
    }
    %next = ttl.cb_reserve %following : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %following : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @nested_conditional_reset_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %following = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %zero = arith.constant 0 : i64
    %outer_value = ttl.opaque_call "outer_for_compute" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
    %outer_active = arith.cmpi ne, %outer_value, %zero : i64
    scf.if %outer_active {
      %inner_value = ttl.opaque_call "inner_for_compute" () {condition_result = #ttl.dispatch_condition<1, i64>, header = "condition.hpp"} : () -> i64
      %inner_active = arith.cmpi ne, %inner_value, %zero : i64
      scf.if %inner_active {
        ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
      }
    }
    %next = ttl.cb_wait %following : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %following : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @nested_conditional_reset_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %zero = arith.constant 0 : i64
    %outer_value = ttl.opaque_call "outer_for_writer" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
    %outer_active = arith.cmpi ne, %outer_value, %zero : i64
    scf.if %outer_active {
      %inner_value = ttl.opaque_call "inner_for_writer" () {condition_result = #ttl.dispatch_condition<1, i64>, header = "condition.hpp"} : () -> i64
      %inner_active = arith.cmpi ne, %inner_value, %zero : i64
      scf.if %inner_active {
        ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
      }
    }
    return
  }
}

// -----

// Opposite reset polarity cannot prove one dynamic reset instance.
// CHECK: DFB logical_id=0 bounded=0
// CHECK: lifecycle_completion=unsupported-control-flow
// CHECK: DFB logical_id=1 bounded=1
// CHECK: Total DFB count: 2

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @opposite_reset_polarity_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %following = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %zero = arith.constant 0 : i64
    %produce_value = ttl.opaque_call "active" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
    %produce_active = arith.cmpi ne, %produce_value, %zero : i64
    scf.if %produce_active {
      %slot = ttl.cb_reserve %old : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %old : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    }
    %reset_value = ttl.opaque_call "inactive" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
    %reset_inactive = arith.cmpi eq, %reset_value, %zero : i64
    scf.if %reset_inactive {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    }
    %next = ttl.cb_reserve %following : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %following : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @opposite_reset_polarity_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %following = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %zero = arith.constant 0 : i64
    %reset_value = ttl.opaque_call "inactive_for_compute" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
    %reset_inactive = arith.cmpi eq, %reset_value, %zero : i64
    scf.if %reset_inactive {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    }
    %next = ttl.cb_wait %following : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %following : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @opposite_reset_polarity_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %zero = arith.constant 0 : i64
    %reset_value = ttl.opaque_call "inactive_for_writer" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
    %reset_inactive = arith.cmpi eq, %reset_value, %zero : i64
    scf.if %reset_inactive {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    }
    return
  }
}

// -----

// Each iteration completes one two-tile transaction before the same reset.
// Per-iteration cursor normalization permits the allocation group to reuse one
// three-tile physical allocation.
// CHECK: DFB allocation group #ttl.dfb_allocation_group<0> members=[0, 1] envelope_bytes=6144 handoff=proven
// CHECK: DFB logical_id=0 bounded=1
// CHECK: epochs=[{executions=4,accesses=[0, 1, 2, 3],transactions=[2]
// CHECK: DFB logical_id=1 bounded=1
// CHECK: Total DFB count: 1
// CHECK: DFB assignment: logical DFB 0 -> physical index 0 storage index 0 allocation_group=#ttl.dfb_allocation_group<0> (bounded)
// CHECK: DFB assignment: logical DFB 1 -> physical index 0 storage index 0 allocation_group=#ttl.dfb_allocation_group<0> (bounded)

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @repeated_reset_lifetime_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "repeated_lifetime">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 1}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
    %current = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      %old_block = ttl.cb_wait %old : <[1, 2], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %old : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "repeated_lifetime">, <kind = data_movement, identity = "reader", operation = "repeated_lifetime">, <kind = data_movement, identity = "writer", operation = "repeated_lifetime">]>(%old : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
    }
    %current_block = ttl.cb_wait %current : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %current : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @repeated_reset_lifetime_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "repeated_lifetime">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 1}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
    %current = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      %old_block = ttl.cb_reserve %old : <[1, 2], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %old : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "repeated_lifetime">, <kind = data_movement, identity = "reader", operation = "repeated_lifetime">, <kind = data_movement, identity = "writer", operation = "repeated_lifetime">]>(%old : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
    }
    %current_block = ttl.cb_reserve %current : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %current : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @repeated_reset_lifetime_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "repeated_lifetime">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 1}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "repeated_lifetime">, <kind = data_movement, identity = "reader", operation = "repeated_lifetime">, <kind = data_movement, identity = "writer", operation = "repeated_lifetime">]>(%old : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
    }
    return
  }
}
