// A completed node must not suppress configuration of an unproved node.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEBUG

// CHECK: ttl.dfb_reconfiguration_plan = {{.*}}nodes = {{\[\[0, 0\], \[1, 0\]\]}}{{.*}}entry_reconfiguration = 0 : i64{{.*}}nodes = {{\[\[1, 0\]\]}}
// DEBUG: node (0,0) lifecycle_completion=complete
// DEBUG: node (1,0) lifecycle_completion=incomplete-use-order

#producer = #ttl.logical_kernel<kind = data_movement, identity = "producer", operation = "operation">
#consumer = #ttl.logical_kernel<kind = compute, identity = "consumer", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary = #ttl.dfb_reconfiguration<0, participants[#consumer, #producer, #writer]>

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
    scf.if %is_first {
      %first_slot = ttl.cb_reserve %dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    scf.if %is_second {
      %second_slot = ttl.cb_reserve %dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    ttl.dfb_reconfiguration #boundary
    return
  }

  func.func @partial_lifetime() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #consumer
  } {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %node_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %is_first = arith.cmpi eq, %node_x, %zero : index
    %is_second = arith.cmpi eq, %node_x, %one : index
    scf.if %is_first {
      %first_value = ttl.cb_wait %dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    scf.if %is_second {
      %second_value = ttl.cb_wait %dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      ttl.opaque_call "after_pop" dfb_dependencies(
          %dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          dfb_accesses [#ttl.dfb_non_transactional_access<inspect, 0>]
          () {header = "effects.hpp"} : () -> ()
    }
    ttl.dfb_reconfiguration #boundary
    return
  }

  func.func @writer() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>, ttl.noc_index = 1 : i32,
    ttl.logical_kernel = #writer
  } {
    ttl.dfb_reconfiguration #boundary
    return
  }
}
