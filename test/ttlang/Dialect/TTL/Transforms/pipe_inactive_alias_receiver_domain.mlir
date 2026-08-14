// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

// A non-pipe producer and a fabric receiver share one physical DFB identity
// but execute on distinct logical devices. The non-pipe push has exact
// execution count zero at the receiver device and cannot change its cursor.

// CHECK-LABEL: func.func @fabric_sender
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc
// CHECK-LABEL: func.func @fabric_receiver
// CHECK: ttkernel.cb_reserve_back
// CHECK: ttkernel.routing_plane.atomic_inc
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK-NEXT: ttkernel.cb_push_back

#device_domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#device_zero_to_one = #ttl.device_transfer<
    domain = #device_domain,
    edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @unrelated_device_zero_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %receiver = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %is_device_zero = ttl.is_device <coordinates = [0]> in #device_domain : i1
    scf.if %is_device_zero {
      %reserved = ttl.cb_reserve %receiver
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_push %receiver
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    func.return
  }

  func.func @fabric_sender()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #device_zero_to_one}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %is_device_zero = ttl.is_device <coordinates = [0]> in #device_domain : i1
    scf.if %is_device_zero {
      %send = ttl.copy %source, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }

  func.func @fabric_receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %receiver = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #device_zero_to_one}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %is_device_one = ttl.is_device <coordinates = [1]> in #device_domain : i1
    scf.if %is_device_one {
      %reserved = ttl.cb_reserve %receiver
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post = ttl.copy %pipe, %reserved
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      ttl.wait %post : !ttl.transfer_handle
      ttl.cb_push %receiver
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    func.return
  }
}
