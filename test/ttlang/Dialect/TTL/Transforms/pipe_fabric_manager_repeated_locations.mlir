// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verify ownership transfer when one callback executes multiple
// records at the same location.

// Two transfers use node (1, 0) on device 0 in each callback. Matching
// location multiplicities permit local ownership, and the repeated records
// require a runtime generation counter even without an SCF loop.
// CHECK-LABEL: module attributes
// CHECK-SAME: ttl.pipe_sync_semaphore_count = 1 : i64
// CHECK-LABEL: func.func @sender_node
// CHECK-SAME: interferingIntervals = ["external.collective"]>
// CHECK: %[[SENDER_COUNTER:.*]] = memref.alloca() : memref<1xi32>
// CHECK: %[[SENDER_INVOCATION:.*]] = memref.load %[[SENDER_COUNTER]]
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: ttkernel.routing_plane.close_connections
// CHECK: memref.store {{.*}}, %[[SENDER_COUNTER]]
// CHECK-LABEL: func.func @receiver_node
// CHECK-SAME: interferingIntervals = ["external.collective"]>
// CHECK: %[[RECEIVER_COUNTER:.*]] = memref.alloca() : memref<1xi32>
// CHECK: %[[RECEIVER_INVOCATION:.*]] = memref.load %[[RECEIVER_COUNTER]]
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: ttkernel.routing_plane.close_connections
// CHECK: memref.store {{.*}}, %[[RECEIVER_COUNTER]]
// CHECK-LABEL: func.func @external_collective
// CHECK-SAME: ttl.fabric_manager_intervals = [#ttl.fabric_manager_interval<identity = "external.collective", kind = external, claim = "collective", routeIndices = [], interferingIntervals = ["generated.0", "generated.1"]>]
// CHECK-NEXT: ttkernel.opaque_call "external_collective"
// CHECK-NEXT: return

module attributes {ttl.launch_grid = [2, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @idle_compute() attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #ttl.logical_kernel<kind = compute>} {
    return
  }
  func.func @sender_node(%arg0: tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 2], memory = interleaved>>) attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [0 : i32], ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {
    %0 = ttl.bind_cb{cb_index = 0, block_count = 1} {dfb_id = 0 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %c2_i64 = arith.constant 2 : i64
    %1 = ttl.core_x : index
    %2 = ttl.core_y : index
    %c1_i64 = arith.constant 1 : i64
    %3 = arith.index_cast %c1_i64 : i64 to index
    %4 = arith.cmpi eq, %1, %3 : index
    scf.if %4 {
      ttl.pipenet_foreach_src attributes {records = #ttl.pipenet_records<net 0 name "exchange_net" pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0, deviceTransfer = <domain = <components = <name = "device", extent = [2, 1]>>, edge = <source = <coordinates = [0, 0]>, destination = <coordinates = [1, 0]>>>>, <srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0, deviceTransfer = <domain = <components = <name = "device", extent = [2, 1]>>, edge = <source = <coordinates = [0, 0]>, destination = <coordinates = [1, 0]>>>>, <srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0, deviceTransfer = <domain = <components = <name = "device", extent = [2, 1]>>, edge = <source = <coordinates = [1, 0]>, destination = <coordinates = [0, 0]>>>>, <srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0, deviceTransfer = <domain = <components = <name = "device", extent = [2, 1]>>, edge = <source = <coordinates = [1, 0]>, destination = <coordinates = [0, 0]>>>>, <srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0, deviceTransfer = <domain = <components = <name = "device", extent = [2, 1]>>, edge = <source = <coordinates = [0, 0]>, destination = <coordinates = [1, 0]>>>>, <srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0, deviceTransfer = <domain = <components = <name = "device", extent = [2, 1]>>, edge = <source = <coordinates = [1, 0]>, destination = <coordinates = [0, 0]>>>>]>} {
      ^bb0(%arg1: !ttl.selected_pipe_src):
        %5 = ttl.cb_reserve %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %6 = ttl.attach_cb %5, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %c0 = arith.constant 0 : index
        %c0_0 = arith.constant 0 : index
        %7 = ttl.tensor_slice %arg0[%c0, %c0_0] : tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 2], memory = interleaved>> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 2], memory = interleaved>>
        %8 = ttl.copy %7, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 2], memory = interleaved>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> !ttl.transfer_handle<read>
        ttl.wait %8 : !ttl.transfer_handle<read>
        ttl.cb_push %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        %9 = ttl.cb_wait %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %10 = ttl.attach_cb %9, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %11 = ttl.copy %0, %arg1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>, !ttl.selected_pipe_src) -> !ttl.transfer_handle<write>
        ttl.wait %11 : !ttl.transfer_handle<write>
        ttl.cb_pop %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      }
    }
    return
  }
  func.func @receiver_node(%arg0: tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 2], memory = interleaved>>) attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [1 : i32], ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 1 : i32} {
    %0 = ttl.bind_cb{cb_index = 1, block_count = 1} {dfb_id = 1 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %c2_i64 = arith.constant 2 : i64
    %seed_x = arith.constant 0 : index
    %seed_y = arith.constant 0 : index
    %seed_reserved = ttl.cb_reserve %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %seed_attached = ttl.attach_cb %seed_reserved, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %seed = ttl.tensor_slice %arg0[%seed_x, %seed_y] : tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 2], memory = interleaved>> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 2], memory = interleaved>>
    %seed_copy = ttl.copy %seed, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 2], memory = interleaved>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> !ttl.transfer_handle<read>
    ttl.wait %seed_copy : !ttl.transfer_handle<read>
    ttl.cb_push %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %seed_payload = ttl.cb_wait %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %1 = ttl.core_x : index
    %2 = ttl.core_y : index
    %c1_i64 = arith.constant 1 : i64
    %3 = arith.index_cast %c1_i64 : i64 to index
    %4 = arith.cmpi eq, %1, %3 : index
    scf.if %4 {
      ttl.pipenet_foreach_dst attributes {records = #ttl.pipenet_records<net 0 name "exchange_net" pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0, deviceTransfer = <domain = <components = <name = "device", extent = [2, 1]>>, edge = <source = <coordinates = [0, 0]>, destination = <coordinates = [1, 0]>>>>, <srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0, deviceTransfer = <domain = <components = <name = "device", extent = [2, 1]>>, edge = <source = <coordinates = [0, 0]>, destination = <coordinates = [1, 0]>>>>, <srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0, deviceTransfer = <domain = <components = <name = "device", extent = [2, 1]>>, edge = <source = <coordinates = [1, 0]>, destination = <coordinates = [0, 0]>>>>, <srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0, deviceTransfer = <domain = <components = <name = "device", extent = [2, 1]>>, edge = <source = <coordinates = [1, 0]>, destination = <coordinates = [0, 0]>>>>, <srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0, deviceTransfer = <domain = <components = <name = "device", extent = [2, 1]>>, edge = <source = <coordinates = [0, 0]>, destination = <coordinates = [1, 0]>>>>, <srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0, deviceTransfer = <domain = <components = <name = "device", extent = [2, 1]>>, edge = <source = <coordinates = [1, 0]>, destination = <coordinates = [0, 0]>>>>]>} {
      ^bb0(%arg1: !ttl.selected_pipe_dst):
        %5 = ttl.cb_reserve %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %6 = ttl.attach_cb %5, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %7 = ttl.copy %arg1, %6 : (!ttl.selected_pipe_dst, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> !ttl.receive_request
        ttl.wait %7 : !ttl.receive_request
        ttl.cb_push %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        %8 = ttl.cb_wait %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %9 = ttl.attach_cb %8, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %c0 = arith.constant 0 : index
        %c0_0 = arith.constant 0 : index
        %10 = ttl.tensor_slice %arg0[%c0, %c0_0] : tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 2], memory = interleaved>> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 2], memory = interleaved>>
        %11 = ttl.copy %0, %10 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>, tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 2], memory = interleaved>>) -> !ttl.transfer_handle<write>
        ttl.wait %11 : !ttl.transfer_handle<write>
        ttl.cb_pop %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      }
    }
    return
  }
  func.func @external_collective() attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 1 : i32} {
    ttl.opaque_call "external_collective" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "collective", kind = scoped>],
        header = "fabric.hpp"} : () -> ()
    return
  }
}
