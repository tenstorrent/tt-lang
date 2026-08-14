// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

// Two conditionally executed receiver posts have no statically proven order.
// Each reserves the complete DFB, so both computed addresses are block zero.

// CHECK-LABEL: func.func @kernel
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc

#domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#transfer = #ttl.device_transfer<
    domain = #domain,
    edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @kernel(%condition_zero: i1, %condition_one: i1)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %receiver = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe_zero = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #transfer}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %pipe_one = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1 {
        deviceTransfer = #transfer}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
    scf.if %condition_zero {
      %is_source = ttl.is_device <coordinates = [0]> in #domain : i1
      scf.if %is_source {
        %send = ttl.copy %source, %pipe_zero
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
               !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
      %is_destination = ttl.is_device <coordinates = [1]> in #domain : i1
      scf.if %is_destination {
        %reserved = ttl.cb_reserve %receiver
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %post = ttl.copy %pipe_zero, %reserved
            : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.transfer_handle
        ttl.wait %post : !ttl.transfer_handle
        ttl.cb_push %receiver : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      }
    }
    scf.if %condition_one {
      %is_source = ttl.is_device <coordinates = [0]> in #domain : i1
      scf.if %is_source {
        %send = ttl.copy %source, %pipe_one
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
               !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
      %is_destination = ttl.is_device <coordinates = [1]> in #domain : i1
      scf.if %is_destination {
        %reserved = ttl.cb_reserve %receiver
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %post = ttl.copy %pipe_one, %reserved
            : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.transfer_handle
        ttl.wait %post : !ttl.transfer_handle
        ttl.cb_push %receiver : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      }
    }
    func.return
  }
}
