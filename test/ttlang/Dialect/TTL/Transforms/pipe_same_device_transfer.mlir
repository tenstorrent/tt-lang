// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s --implicit-check-not=ttkernel.routing_plane --implicit-check-not=ttl.fabric_routes

// Summary: Verifies that a device-qualified transfer whose endpoints are on
// the same device uses local NoC transport without allocating a fabric route.

#domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#records = #ttl.pipenet_records<net 0 name "local_device" mappings
  <graph = <domain = #domain, kind = explicit, properties = {
    edges = [#ttl.transfer_edge<source = <coordinates = [0]>,
                                destination = <coordinates = [0]>>]}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>]>>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  // CHECK-LABEL: func.func @sender
  // CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
  // CHECK: ttkernel.experimental.constant_table_lookup {{.*}}, [1, 0]
  // CHECK: scf.for
  // CHECK: ttkernel.noc_async_write
  // CHECK: ttkernel.noc_async_write_barrier
  // CHECK: ttkernel.noc_semaphore_inc
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
  // CHECK: ttkernel.experimental.constant_table_lookup {{.*}}, [1, 0]
  // CHECK: scf.for
  // CHECK: ttkernel.cb_reserve_back
  // CHECK: ttkernel.noc_semaphore_inc
  // CHECK: ttkernel.experimental.semaphore_wait_min
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
