// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verifies that unicast PipeNet foreach ops lower to one loop
// around one callback body.

// CHECK-LABEL: func.func @foreach_src_send
// CHECK: scf.for
// CHECK: ttkernel.my_logical_x
// CHECK: ttkernel.noc_async_write(
// CHECK-NOT: ttkernel.noc_async_write(
// CHECK: return
// CHECK-NOT: ttl.pipenet_foreach_src
func.func @foreach_src_send() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.pipenet_foreach_src attributes {
      pipeNetId = 0 : i64,
      pipeNetName = "row_net",
      pipes = [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>
      ]} {
  ^bb0(%pipe: !ttl.selected_pipe_src):
    %xf = ttl.copy %cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.selected_pipe_src)
        -> !ttl.transfer_handle<write>
    ttl.wait %xf : !ttl.transfer_handle<write>
    ttl.yield
  }
  func.return
}

// -----

// CHECK-LABEL: func.func @foreach_dst_receive
// CHECK: memref.alloca
// CHECK: scf.for
// CHECK: ttkernel.remote_sram_write_u32
// CHECK-NOT: ttkernel.remote_sram_write_u32
// CHECK: ttkernel.experimental::semaphore_wait_min
// CHECK-NOT: ttkernel.experimental::semaphore_wait_min
// CHECK: return
// CHECK-NOT: ttl.pipenet_foreach_dst
func.func @foreach_dst_receive() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.pipenet_foreach_dst attributes {
      pipeNetId = 0 : i64,
      pipeNetName = "row_net",
      pipes = [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>
      ]} {
  ^bb0(%pipe: !ttl.selected_pipe_dst):
    %recv = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %xf = ttl.copy %pipe, %recv
        : (!ttl.selected_pipe_dst,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.transfer_handle
    ttl.wait %xf : !ttl.transfer_handle
    ttl.yield
  }
  func.return
}

// -----

// CHECK-LABEL: func.func @foreach_src_multicast_send
// CHECK: scf.for
// CHECK: ttkernel.noc_async_write_multicast_loopback_src(
// CHECK-NOT: ttkernel.noc_async_write_multicast_loopback_src(
// CHECK: ttkernel.noc_async_write_multicast(
// CHECK-NOT: ttkernel.noc_async_write_multicast(
// CHECK: ttkernel.noc_semaphore_inc_multicast(
// CHECK-NOT: ttkernel.noc_semaphore_inc_multicast(
// CHECK: ttkernel.noc_async_atomic_barrier
// CHECK: return
// CHECK-NOT: ttl.pipenet_foreach_src
func.func @foreach_src_multicast_send() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.pipenet_foreach_src attributes {
      pipeNetId = 0 : i64,
      pipeNetName = "row_net",
      pipes = [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 1, dstEndY = 0, isMulticast = true>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 2, dstEndY = 1, isMulticast = true>
      ]} {
  ^bb0(%pipe: !ttl.selected_pipe_src):
    %xf = ttl.copy %cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.selected_pipe_src)
        -> !ttl.transfer_handle<write>
    ttl.wait %xf : !ttl.transfer_handle<write>
    ttl.yield
  }
  func.return
}

// -----

// CHECK-LABEL: func.func @foreach_dst_multicast_receive
// CHECK: memref.alloca
// CHECK: scf.for
// CHECK: ttkernel.remote_sram_write_u32
// CHECK-NOT: ttkernel.remote_sram_write_u32
// CHECK: ttkernel.experimental::semaphore_wait_min
// CHECK-NOT: ttkernel.experimental::semaphore_wait_min
// CHECK: return
// CHECK-NOT: ttl.pipenet_foreach_dst
func.func @foreach_dst_multicast_receive() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.pipenet_foreach_dst attributes {
      pipeNetId = 0 : i64,
      pipeNetName = "row_net",
      pipes = [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 1, dstEndY = 0, isMulticast = true>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 2, dstEndY = 1, isMulticast = true>
      ]} {
  ^bb0(%pipe: !ttl.selected_pipe_dst):
    %recv = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %xf = ttl.copy %pipe, %recv
        : (!ttl.selected_pipe_dst,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.transfer_handle
    ttl.wait %xf : !ttl.transfer_handle
    ttl.yield
  }
  func.return
}

// -----

// CHECK-LABEL: func.func @foreach_src_nested_user_control_flow
// CHECK: scf.for
// CHECK: scf.if
// CHECK: scf.for
// CHECK: scf.if
// CHECK: ttkernel.noc_async_write(
// CHECK-NOT: ttkernel.noc_async_write(
// CHECK: return
// CHECK-NOT: ttl.pipenet_foreach_src
func.func @foreach_src_nested_user_control_flow() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %true = arith.constant true
  ttl.pipenet_foreach_src attributes {
      pipeNetId = 0 : i64,
      pipeNetName = "row_net",
      pipes = [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>
      ]} {
  ^bb0(%pipe: !ttl.selected_pipe_src):
    scf.for %iter = %c0 to %c2 step %c1 {
      scf.if %true {
        %xf = ttl.copy %cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.selected_pipe_src)
            -> !ttl.transfer_handle<write>
        ttl.wait %xf : !ttl.transfer_handle<write>
      }
    }
    ttl.yield
  }
  func.return
}

// -----

// CHECK-LABEL: func.func @foreach_dst_nested_user_control_flow
// CHECK: memref.alloca
// CHECK: scf.for
// CHECK: scf.if
// CHECK: scf.if
// CHECK: ttkernel.remote_sram_write_u32
// CHECK-NOT: ttkernel.remote_sram_write_u32
// CHECK: ttkernel.experimental::semaphore_wait_min
// CHECK-NOT: ttkernel.experimental::semaphore_wait_min
// CHECK: return
// CHECK-NOT: ttl.pipenet_foreach_dst
func.func @foreach_dst_nested_user_control_flow() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %true = arith.constant true
  ttl.pipenet_foreach_dst attributes {
      pipeNetId = 0 : i64,
      pipeNetName = "row_net",
      pipes = [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>
      ]} {
  ^bb0(%pipe: !ttl.selected_pipe_dst):
    scf.if %true {
      %recv = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %xf = ttl.copy %pipe, %recv
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      ttl.wait %xf : !ttl.transfer_handle
    }
    ttl.yield
  }
  func.return
}
