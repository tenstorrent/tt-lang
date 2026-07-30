// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// Summary: Negative tests for PipeNet foreach and selected-pipe IR.

// Point-to-point records must have exactly one receiver.
func.func @point_to_point_record_receiver_count() {
  ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 0 pipes [
        // expected-error @below {{point-to-point pipe record must have exactly one receiver}}
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 2, dstEndY = 0>
        // expected-error @below {{failed to parse TTL_PipeNetRecordsAttr parameter 'pipes'}}
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_src):
    ttl.yield
  }
  func.return
}

// -----

// Foreach records must have a uniform pipe kind.
func.func @foreach_mixed_record_kind() {
  // expected-error @+1 {{'ttl.pipenet_foreach_src' op all pipe records must be either point-to-point or collective}}
  ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 0 pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1, isCollective = true>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_src):
    ttl.yield
  }
  func.return
}

// -----

// Selected transfer creation requires a direct select_pipe definition.
func.func @selected_transfer_requires_direct_def(%pipe: !ttl.selected_pipe_src) {
  // expected-error @+1 {{'ttl.pipe_transfer.create' op selected pipe operand must be a direct result of ttl.select_pipe_src or ttl.select_pipe_dst}}
  %transfer = ttl.pipe_transfer.create %pipe {kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.selected_pipe_src -> !ttl.pipe_transfer
  func.return
}

// -----

// Selected transfer kind must match the selected records kind.
func.func @selected_transfer_kind_mismatch() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %false = arith.constant false
  %src = ttl.select_pipe_src net 0 record(%c0) src (%c0, %c0) dst (%c1, %c0) to (%c1, %c0)
      num_dests (%c1) src_in_dst (%false)
      devices (%c0, %c0) collective (true)
      {records = #ttl.pipenet_records<net 0 pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0, isCollective = true>
      ]>} : !ttl.selected_pipe_src
  // expected-error @+1 {{'ttl.pipe_transfer.create' op selected pipe transfer kind must match the records kind}}
  %transfer = ttl.pipe_transfer.create %src {kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.selected_pipe_src -> !ttl.pipe_transfer
  func.return
}
