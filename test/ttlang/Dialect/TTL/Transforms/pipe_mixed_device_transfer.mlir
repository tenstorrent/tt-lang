// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verifies one graph PipeNet can contain same-device NoC transfers
// and distinct-device fabric transfers without assigning fabric routes to the
// same-device records.

#domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#records = #ttl.pipenet_records<net 0 name "mixed" mappings
  <graph = <domain = #domain, kind = explicit, properties = {
    edges = [
      #ttl.transfer_edge<source = <coordinates = [0]>,
                         destination = <coordinates = [0]>>,
      #ttl.transfer_edge<source = <coordinates = [1]>,
                         destination = <coordinates = [1]>>
    ]}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>]>,
  <graph = <domain = #domain, kind = all_to_all,
    componentName = "device", properties = {}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>]>>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  // CHECK-LABEL: func.func @sender
  // CHECK-SAME: ttl.fabric_routes = [
  // CHECK-SAME: local = #ttl.device_ref<coordinates = [0]>
  // CHECK-SAME: remote = #ttl.device_ref<coordinates = [1]>
  // CHECK-SAME: local = #ttl.device_ref<coordinates = [1]>
  // CHECK-SAME: remote = #ttl.device_ref<coordinates = [0]>
  // CHECK: ttkernel.noc_async_write
  // CHECK: ttkernel.routing_plane.fused_write_atomic_inc
  func.func @sender() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.pipenet_foreach_src attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %send = ttl.copy %source, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.yield
    }
    func.return
  }

  // CHECK-LABEL: func.func @receiver
  // CHECK-SAME: ttl.fabric_routes = [
  // CHECK: ttkernel.noc_semaphore_inc
  // CHECK: ttkernel.routing_plane.atomic_inc
  // CHECK: ttkernel.cb_push_back
  func.func @receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %destination = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.pipenet_foreach_dst attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %reserved = ttl.cb_reserve %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %receive = ttl.copy %pipe, %reserved
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %receive : !ttl.receive_request
      ttl.cb_push %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      ttl.yield
    }
    func.return
  }
}
