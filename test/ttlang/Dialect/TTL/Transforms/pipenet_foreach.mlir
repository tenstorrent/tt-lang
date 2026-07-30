// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verifies compact PipeNet foreach lowering through pipe-transfer IR.

// Small source and destination foreach operations emit direct static guards.

// CHECK-LABEL: func.func @foreach_direct
// CHECK-NOT: scf.for
// CHECK-COUNT-2: ttkernel.noc_async_write %
// CHECK-COUNT-2: ttkernel.experimental.semaphore_wait_min(
// CHECK-NOT: ttkernel.noc_inline_dw_write(
// CHECK-NOT: scf.for
// CHECK-NOT: ttl.pipenet_foreach
// CHECK-NOT: ttl.select_pipe
module attributes {ttl.launch_grid = array<i64: 2, 2>} {
  func.func @foreach_direct()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    ttl.pipenet_foreach_src attributes {
        records = #ttl.pipenet_records<net 0 name "row_net" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
          #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>
        ]>} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %xf = ttl.copy %src, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %xf : !ttl.transfer_handle<write>
      ttl.yield
    }
    ttl.pipenet_foreach_dst attributes {
        records = #ttl.pipenet_records<net 0 name "row_net" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
          #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>
        ]>} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %recv = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %xf = ttl.copy %pipe, %recv
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      ttl.wait %xf : !ttl.transfer_handle
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      ttl.yield
    }
    func.return
  }
}

// -----

// Large source and destination foreach operations retain one loop per role.

// CHECK-LABEL: func.func @foreach_compact
// CHECK: scf.for
// CHECK: ttkernel.noc_async_write %
// CHECK: scf.for
// CHECK: ttkernel.noc_inline_dw_write(
// CHECK: ttkernel.experimental.semaphore_wait_min(
// CHECK: ttkernel.noc_semaphore_set(
// CHECK-NOT: ttl.pipenet_foreach
// CHECK-NOT: ttl.select_pipe
module attributes {ttl.launch_grid = array<i64: 2, 5>} {
  func.func @foreach_compact()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    ttl.pipenet_foreach_src attributes {
        records = #ttl.pipenet_records<net 0 name "row_net" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
          #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>,
          #ttl.pipe_record<srcX = 0, srcY = 2, dstStartX = 1, dstStartY = 2, dstEndX = 1, dstEndY = 2>,
          #ttl.pipe_record<srcX = 0, srcY = 3, dstStartX = 1, dstStartY = 3, dstEndX = 1, dstEndY = 3>,
          #ttl.pipe_record<srcX = 0, srcY = 4, dstStartX = 1, dstStartY = 4, dstEndX = 1, dstEndY = 4>
        ]>} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %xf = ttl.copy %src, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %xf : !ttl.transfer_handle<write>
      ttl.yield
    }
    ttl.pipenet_foreach_dst attributes {
        records = #ttl.pipenet_records<net 0 name "row_net" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
          #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>,
          #ttl.pipe_record<srcX = 0, srcY = 2, dstStartX = 1, dstStartY = 2, dstEndX = 1, dstEndY = 2>,
          #ttl.pipe_record<srcX = 0, srcY = 3, dstStartX = 1, dstStartY = 3, dstEndX = 1, dstEndY = 3>,
          #ttl.pipe_record<srcX = 0, srcY = 4, dstStartX = 1, dstStartY = 4, dstEndX = 1, dstEndY = 4>
        ]>} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %recv = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %xf = ttl.copy %pipe, %recv
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      ttl.wait %xf : !ttl.transfer_handle
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      ttl.yield
    }
    func.return
  }
}

// -----

// Logical-device records retain compact iteration while selecting the current
// device, fabric route, synchronization resources, and completion protocol.

// CHECK-LABEL: module attributes
// CHECK-SAME: ttl.pipe_global_semaphore_count = 2 : i64
// CHECK-SAME: ttl.pipe_sync_semaphore_count = 0 : i64
// CHECK-LABEL: func.func @foreach_fabric_sender
// CHECK-SAME: ttl.fabric_routes = [
// CHECK-SAME: source_nodes = [array<i64: 0, 0>]
// CHECK: scf.for
// CHECK: ttkernel.experimental.constant_table_lookup {{.*}}, [0, 2]
// CHECK: arith.cmpi eq
// CHECK: scf.if
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc
// CHECK-LABEL: func.func @foreach_fabric_receiver
// CHECK-SAME: ttl.fabric_routes = [
// CHECK-SAME: source_nodes = [array<i64: 1, 0>, array<i64: 1, 1>]
// CHECK: scf.for
// CHECK: ttkernel.experimental.constant_table_lookup {{.*}}, [1, 3]
// CHECK: arith.cmpi eq
// CHECK: scf.if
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: ttkernel.noc_semaphore_set
// CHECK-NOT: ttl.pipenet_foreach
// CHECK-NOT: ttl.select_pipe
module attributes {ttl.launch_grid = array<i64: 2, 2>} {
  func.func @foreach_fabric_sender()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    ttl.pipenet_foreach_src attributes {
        records = #ttl.pipenet_records<net 900 name "fabric_net" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 1, isMulticast = true, pipeNetId = 100, deviceTransfer = <domain = <components = <name = "device", extent = [1, 4]>>, edge = <source = <coordinates = [0, 0]>, destination = <coordinates = [0, 1]>>>>,
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 1, isMulticast = true, pipeNetId = 101, deviceTransfer = <domain = <components = <name = "device", extent = [1, 4]>>, edge = <source = <coordinates = [0, 2]>, destination = <coordinates = [0, 3]>>>>
        ]>} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %send = ttl.copy %src, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.yield
    }
    func.return
  }

  func.func @foreach_fabric_receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    ttl.pipenet_foreach_dst attributes {
        records = #ttl.pipenet_records<net 900 name "fabric_net" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 1, isMulticast = true, pipeNetId = 100, deviceTransfer = <domain = <components = <name = "device", extent = [1, 4]>>, edge = <source = <coordinates = [0, 0]>, destination = <coordinates = [0, 1]>>>>,
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 1, isMulticast = true, pipeNetId = 101, deviceTransfer = <domain = <components = <name = "device", extent = [1, 4]>>, edge = <source = <coordinates = [0, 2]>, destination = <coordinates = [0, 3]>>>>
        ]>} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %recv = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post = ttl.copy %pipe, %recv
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      ttl.wait %post : !ttl.transfer_handle
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      ttl.yield
    }
    func.return
  }
}
