// RUN: ttlang-opt %s --ttl-verify-pipenet | FileCheck %s

// Summary: Full expansion of this 1366-trip all-to-all loop stays within the
// expansion bounds but gives each launch node more than 4096 pipe events. Its
// pipe control does not depend on the induction variable, so the schedule is
// verified in the compact form that applies when a loop exceeds the expansion
// bounds.

// CHECK-LABEL: func.func @send_dm
// CHECK-LABEL: func.func @recv_dm

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @compute() attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #ttl.logical_kernel<kind = compute>} {
    %0 = ttl.bind_cb{cb_index = 1, block_count = 1} {dfb_id = 1 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cn = arith.constant 1366 : index
    scf.for %arg0 = %c0 to %cn step %c1 {
      %2 = ttl.cb_wait %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %3 = ttl.attach_cb %2, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    return
  }
  func.func @send_dm() attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {
    %0 = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cn = arith.constant 1366 : index
    scf.for %it = %c0 to %cn step %c1 {
    ttl.pipenet_scope attributes {ttl.pipe_net_ids = [0], ttl.pipe_net_roles = [0]} {
      %1 = ttl.cb_reserve %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %2 = ttl.attach_cb %1, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.pipenet_foreach_src attributes {records = #ttl.pipenet_records<net 0 name "DEVICE_ALL_TO_ALL_NET" mappings <graph = <domain = <components = <name = "device", extent = [1, 2]>>, kind = all_to_all, componentName = "device", properties = {}>, pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>]>>} {
      ^bb0(%arg0: !ttl.selected_pipe_src):
        %3 = ttl.copy %2, %arg0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.selected_pipe_src) -> !ttl.transfer_handle<write>
        ttl.wait %3 : !ttl.transfer_handle<write>
      }
      ttl.cb_push %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    }
    return
  }
  func.func @recv_dm() attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 1 : i32} {
    %0 = ttl.bind_cb{cb_index = 1, block_count = 1} {dfb_id = 1 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cn = arith.constant 1366 : index
    scf.for %it = %c0 to %cn step %c1 {
    ttl.pipenet_foreach_dst attributes {records = #ttl.pipenet_records<net 0 name "DEVICE_ALL_TO_ALL_NET" mappings <graph = <domain = <components = <name = "device", extent = [1, 2]>>, kind = all_to_all, componentName = "device", properties = {}>, pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>]>>} {
    ^bb0(%arg0: !ttl.selected_pipe_dst):
      %1 = ttl.cb_reserve %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %2 = ttl.attach_cb %1, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %3 = ttl.copy %arg0, %2 : (!ttl.selected_pipe_dst, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> !ttl.receive_request
      ttl.wait %3 : !ttl.receive_request
      ttl.cb_push %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    }
    return
  }
}
