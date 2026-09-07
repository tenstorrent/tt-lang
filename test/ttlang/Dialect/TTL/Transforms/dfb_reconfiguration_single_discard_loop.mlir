// Verifies one repeated state-discarding DFB reconfiguration.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#reconfigure = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer], discard_dfb_state = true>

// One call at the end of each iteration resets unread data and reinstalls the
// initial descriptor before the next iteration.
// CHECK: ttl.dfb_reconfiguration_plan = {boundary_ordinals = array<i64: 0>
// CHECK-SAME: entry_reconfiguration = 0 : i64

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute_repeated_discard() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %data = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.opaque_call "wait_without_pop" dfb_dependencies(
          %data : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>]
          () {header = "access.hpp"} : () -> ()
      ttl.dfb_reconfiguration #reconfigure
    }
    return
  }

  func.func @read_repeated_discard() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %data = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.opaque_call "publish" dfb_dependencies(
          %data : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                       #ttl.dfb_protocol_effect<push, 0, 1>]
          () {header = "access.hpp"} : () -> ()
      ttl.dfb_reconfiguration #reconfigure
    }
    return
  }

  func.func @write_repeated_discard() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #reconfigure
    }
    return
  }
}
