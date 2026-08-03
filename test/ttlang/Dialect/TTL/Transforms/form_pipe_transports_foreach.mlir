// RUN: ttlang-opt %s --ttl-form-pipe-transports | FileCheck %s

// Summary: Verifies that transport grouping preserves PipeNet foreach
// callbacks for record-selection lowering.

// CHECK-LABEL: func.func @preserve_foreach
// CHECK: ttl.pipenet_foreach_src
// CHECK: ttl.copy
// CHECK: ttl.wait
// CHECK-NEXT: }
func.func @preserve_foreach()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 0 name "row_net" pipes [
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
  func.return
}
