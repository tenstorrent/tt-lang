// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

// Verify that PipeGraph distinguishes the same core-local DFB on different
// logical devices when assigning and releasing receiver slots.

// Device-qualified execution domains keep identical core/DFB resources on
// different devices independent. The range-predicated pop executes on both
// receiver devices and releases each device's own live receive slot.
// CHECK-LABEL: func.func @device_qualified_receiver_slots
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc
// CHECK: ttkernel.routing_plane.atomic_inc
// CHECK: ttkernel.cb_pop_front
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @device_qualified_receiver_slots()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %is_core_zero = arith.cmpi eq, %core_x, %zero : index
    %is_device_one = ttl.is_device <coordinates = [1]> in
        <components = <name = "device", extent = [3]>> : i1
    %is_device_two = ttl.is_device <coordinates = [2]> in
        <components = <name = "device", extent = [3]>> : i1
    %is_device_zero = ttl.is_device <coordinates = [0]> in
        <components = <name = "device", extent = [3]>> : i1
    %is_receiver_device = ttl.is_device_in_range
        <lo = <coordinates = [1]>, hi = <coordinates = [3]>> in
        <components = <name = "device", extent = [3]>> : i1
    %device_one_receiver = arith.andi %is_device_one, %is_core_zero : i1
    %device_two_receiver = arith.andi %is_device_two, %is_core_zero : i1

    %pipe0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        {deviceTransfer = #ttl.device_transfer<
          domain = <components = <name = "device", extent = [3]>>,
          edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %transfer0 = ttl.pipe_transfer.create %pipe0
        {expectedReceivers = 1 : i64,
         kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> -> !ttl.pipe_transfer
    scf.if %is_device_zero {
      %send = ttl.pipe_transfer.send %transfer0, %src_cb
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    scf.if %device_one_receiver {
      %recv = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer0, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }

    %pipe1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
        {deviceTransfer = #ttl.device_transfer<
          domain = <components = <name = "device", extent = [3]>>,
          edge = <source = <coordinates = [0]>, destination = <coordinates = [2]>>>}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
    %transfer1 = ttl.pipe_transfer.create %pipe1
        {expectedReceivers = 1 : i64,
         kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1> -> !ttl.pipe_transfer
    scf.if %is_device_zero {
      %send = ttl.pipe_transfer.send %transfer1, %src_cb
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    scf.if %device_two_receiver {
      %recv = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer1, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.pipe_token<net 1>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 1>
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }

    scf.if %is_receiver_device {
      %ready = ttl.cb_wait %cb
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }

    %pipe2 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 2
        {deviceTransfer = #ttl.device_transfer<
          domain = <components = <name = "device", extent = [3]>>,
          edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 2>
    %transfer2 = ttl.pipe_transfer.create %pipe2
        {expectedReceivers = 1 : i64,
         kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 2> -> !ttl.pipe_transfer
    scf.if %is_device_zero {
      %send = ttl.pipe_transfer.send %transfer2, %src_cb
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    scf.if %device_one_receiver {
      %recv = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer2, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.pipe_token<net 2>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 2>
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }

    %pipe3 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 3
        {deviceTransfer = #ttl.device_transfer<
          domain = <components = <name = "device", extent = [3]>>,
          edge = <source = <coordinates = [0]>, destination = <coordinates = [2]>>>}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 3>
    %transfer3 = ttl.pipe_transfer.create %pipe3
        {expectedReceivers = 1 : i64,
         kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 3> -> !ttl.pipe_transfer
    scf.if %is_device_zero {
      %send = ttl.pipe_transfer.send %transfer3, %src_cb
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    scf.if %device_two_receiver {
      %recv = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer3, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.pipe_token<net 3>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 3>
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }
    func.return
  }
}
