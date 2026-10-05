// Under repeated reconfiguration, node (1, 0) accesses the DFB before the first
// boundary and its lifetime is not proven. In later iterations those accesses
// run under the configuration the last boundary installs, so the node keeps its
// descriptor there as well as in the initial configuration.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s

// CHECK: ttl.dfb_reconfiguration_plan
// CHECK-SAME: {block_count = 2 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32, page_size = 2048 : i32, storage_segments = [{nodes = {{\[\[}}1, 0{{\]\]}}}]}
// CHECK-SAME: {block_count = 2 : i32, element_type = !ttcore.tile<32x32, bf16>, entry_reconfiguration = 1 : i64, num_tiles = 1 : i32, page_size = 2048 : i32, storage_segments = [{nodes = {{\[\[}}1, 0{{\]\]}}}]}

#producer = #ttl.logical_kernel<kind = data_movement, identity = "producer", operation = "operation">
#consumer = #ttl.logical_kernel<kind = compute, identity = "consumer", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#r0 = #ttl.dfb_reconfiguration<0, participants[#consumer, #producer, #writer], discard_dfb_state = true>
#r1 = #ttl.dfb_reconfiguration<1, participants[#consumer, #producer, #writer], discard_dfb_state = true>

module attributes {ttl.launch_grid = [2, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @producer() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>, ttl.noc_index = 0 : i32,
    ttl.logical_kernel = #producer
  } {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %node_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %is_first = arith.cmpi eq, %node_x, %zero : index
    %is_second = arith.cmpi eq, %node_x, %one : index
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      scf.if %is_second {
        %slot1 = ttl.cb_reserve %dfb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      }
      ttl.dfb_reconfiguration #r0
      scf.if %is_first {
        %slot0 = ttl.cb_reserve %dfb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      }
      ttl.dfb_reconfiguration #r1
    }
    return
  }

  func.func @consumer() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #consumer
  } {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %node_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %is_first = arith.cmpi eq, %node_x, %zero : index
    %is_second = arith.cmpi eq, %node_x, %one : index
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      scf.if %is_second {
        %value1 = ttl.cb_wait %dfb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        ttl.opaque_call "after_pop" dfb_dependencies(
            %dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            dfb_accesses [#ttl.dfb_non_transactional_access<inspect, 0>]
            () {header = "effects.hpp"} : () -> ()
      }
      ttl.dfb_reconfiguration #r0
      scf.if %is_first {
        %value0 = ttl.cb_wait %dfb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      }
      ttl.dfb_reconfiguration #r1
    }
    return
  }

  func.func @writer() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>, ttl.noc_index = 1 : i32,
    ttl.logical_kernel = #writer
  } {
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #r0
      ttl.dfb_reconfiguration #r1
    }
    return
  }
}
