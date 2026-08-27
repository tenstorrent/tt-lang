// Verifies exact conversion rejects combined PipeNet, reset, and DFB overflow.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=false l1-budget-override=8255})'

// expected-error @below {{combined DFB and runtime resources require 8256 L1 bytes but the budget is 8255 (DFB=8192, scratch=64, global semaphores=0, reconfiguration state=0)}}
module attributes {
  ttl.launch_grid = array<i64: 2, 2>,
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @pipe_receive()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.pipenet_foreach_dst attributes {
        records = #ttl.pipenet_records<net 0 name "reset_pipe" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>
        ]>} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %destination = ttl.cb_reserve %dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %transfer = ttl.copy %pipe, %destination
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      ttl.wait %transfer : !ttl.transfer_handle
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      ttl.yield
    }
    return
  }

  func.func @pipe_and_reset()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.pipenet_foreach_src attributes {
        records = #ttl.pipenet_records<net 0 name "reset_pipe" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>
        ]>} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %transfer = ttl.copy %dfb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %transfer : !ttl.transfer_handle<write>
      ttl.yield
    }
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
    return
  }
}
