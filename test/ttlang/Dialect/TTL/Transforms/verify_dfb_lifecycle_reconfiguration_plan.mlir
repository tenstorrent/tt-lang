// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-verify-dfb-lifecycle)'

// Summary: A state-discarding reconfiguration restores exactly the DFB
// descriptors that the finalized `ttl.dfb_reconfiguration_plan` installs at
// its boundary. Both modules below are finalized; they differ only in whether
// the plan reinstalls DFB 0 at boundary 1.

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary0 = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer], discard_dfb_state = true>
#boundary1 = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer], discard_dfb_state = true>

// The plan reinstalls DFB 0 at boundary 1 in every iteration, so each
// iteration publishes one block into an empty DFB.
module attributes {
    ttl.dfb_allocations = [{allocation_nodes = [[0, 0]], block_count = 1 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32, page_size = 2048 : i32, storage_index = 0 : i32}],
    ttl.dfb_reconfiguration_plan = {boundary_ordinals = array<i64: 0, 1>, dfbs = [{dfb_index = 0 : i32, configurations = [
        {block_count = 1 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32, page_size = 2048 : i32, storage_segments = [{nodes = [[0, 0]]}]},
        {block_count = 1 : i32, element_type = !ttcore.tile<32x32, bf16>, entry_reconfiguration = 1 : i64, num_tiles = 1 : i32, page_size = 2048 : i32, storage_segments = [{nodes = [[0, 0]]}]}]}]},
    ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #compute} {
    %partial = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %iteration = %c0 to %c4 step %c1 {
      ttl.opaque_call "produce_partial" dfb_dependencies(%partial : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
          dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>]
          () {header = "partial.hpp"} : () -> ()
      ttl.dfb_reconfiguration #boundary0
      ttl.dfb_reconfiguration #boundary1
    }
    return
  }

  func.func @read() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #reader, ttl.noc_index = 0 : i32} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %iteration = %c0 to %c4 step %c1 {
      ttl.dfb_reconfiguration #boundary0
      ttl.dfb_reconfiguration #boundary1
    }
    return
  }

  func.func @write() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #writer, ttl.noc_index = 1 : i32} {
    %partial = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %iteration = %c0 to %c4 step %c1 {
      ttl.opaque_call "read_partial" dfb_dependencies(%partial : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
          dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>]
          () {header = "partial.hpp"} : () -> ()
      ttl.dfb_reconfiguration #boundary0
      ttl.dfb_reconfiguration #boundary1
    }
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary0 = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer], discard_dfb_state = true>
#boundary1 = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer], discard_dfb_state = true>

// The plan keeps DFB 0's initial descriptor at both boundaries, so neither
// state-discarding reconfiguration resets it. The four published blocks are
// never popped and exceed the capacity of one.
module attributes {
    ttl.dfb_allocations = [{allocation_nodes = [[0, 0]], block_count = 1 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32, page_size = 2048 : i32, storage_index = 0 : i32}],
    ttl.dfb_reconfiguration_plan = {boundary_ordinals = array<i64: 0, 1>, dfbs = [{dfb_index = 0 : i32, configurations = [
        {block_count = 1 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32, page_size = 2048 : i32, storage_segments = [{nodes = [[0, 0]]}]}]}]},
    ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #compute} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %partial = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %iteration = %c0 to %c4 step %c1 {
      // expected-error @+3 {{logical DFB 0 has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0}}
      // expected-note @+2 {{the producer pushes 4 block(s) per launch and the consumer pops 0, leaving 4 outstanding block(s) for capacity 1}}
      // expected-note @+1 {{keep pops within pushes and unpopped blocks within DFB capacity on every active node}}
      ttl.opaque_call "produce_partial" dfb_dependencies(%partial : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
          dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>]
          () {header = "partial.hpp"} : () -> ()
      ttl.dfb_reconfiguration #boundary0
      ttl.dfb_reconfiguration #boundary1
    }
    return
  }

  func.func @read() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #reader, ttl.noc_index = 0 : i32} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %iteration = %c0 to %c4 step %c1 {
      ttl.dfb_reconfiguration #boundary0
      ttl.dfb_reconfiguration #boundary1
    }
    return
  }

  func.func @write() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #writer, ttl.noc_index = 1 : i32} {
    %partial = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %iteration = %c0 to %c4 step %c1 {
      ttl.opaque_call "read_partial" dfb_dependencies(%partial : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
          dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>]
          () {header = "partial.hpp"} : () -> ()
      ttl.dfb_reconfiguration #boundary0
      ttl.dfb_reconfiguration #boundary1
    }
    return
  }
}
