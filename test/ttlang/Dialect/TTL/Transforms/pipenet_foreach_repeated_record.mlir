// RUN: ttlang-opt %s -ttl-verify-pipenet-guards
// RUN: ttlang-opt %s -ttl-to-ttkernel-pipeline

// A selected callback can contain repeated records with the same endpoints.
// Each receive wait must complete the post from its own record execution before
// the next execution reuses the posted address.

module attributes {ttl.launch_grid = array<i64: 3, 1>} {
func.func @kernel()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %receive_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      {dfb_id = 1 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  ttl.pipenet_foreach_dst attributes {
      records = #ttl.pipenet_records<net 0 name "repeated" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 2, dstStartY = 0, dstEndX = 2, dstEndY = 0>,
        #ttl.pipe_record<srcX = 1, srcY = 0, dstStartX = 2, dstStartY = 0, dstEndX = 2, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 2, dstStartY = 0, dstEndX = 2, dstEndY = 0>,
        #ttl.pipe_record<srcX = 1, srcY = 0, dstStartX = 2, dstStartY = 0, dstEndX = 2, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 2, dstStartY = 0, dstEndX = 2, dstEndY = 0>,
        #ttl.pipe_record<srcX = 1, srcY = 0, dstStartX = 2, dstStartY = 0, dstEndX = 2, dstEndY = 0>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_dst):
    %reserved = ttl.cb_reserve %receive_dfb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %receive_buffer = ttl.attach_cb %reserved, %receive_dfb
        : (tensor<1x1x!ttcore.tile<32x32, f32>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %receive = ttl.copy %pipe, %receive_buffer
        : (!ttl.selected_pipe_dst,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %receive : !ttl.transfer_handle
    ttl.cb_push %receive_dfb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    %received = ttl.cb_wait %receive_dfb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_pop %receive_dfb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.yield
  }
  ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 0 name "repeated" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 2, dstStartY = 0, dstEndX = 2, dstEndY = 0>,
        #ttl.pipe_record<srcX = 1, srcY = 0, dstStartX = 2, dstStartY = 0, dstEndX = 2, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 2, dstStartY = 0, dstEndX = 2, dstEndY = 0>,
        #ttl.pipe_record<srcX = 1, srcY = 0, dstStartX = 2, dstStartY = 0, dstEndX = 2, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 2, dstStartY = 0, dstEndX = 2, dstEndY = 0>,
        #ttl.pipe_record<srcX = 1, srcY = 0, dstStartX = 2, dstStartY = 0, dstEndX = 2, dstEndY = 0>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_src):
    %reserved = ttl.cb_reserve %send_dfb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_push %send_dfb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    %send_buffer = ttl.cb_wait %send_dfb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %attached = ttl.attach_cb %send_buffer, %send_dfb
        : (tensor<1x1x!ttcore.tile<32x32, f32>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %send = ttl.copy %send_dfb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.selected_pipe_src)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    ttl.cb_pop %send_dfb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.yield
  }
  func.return
}
}
