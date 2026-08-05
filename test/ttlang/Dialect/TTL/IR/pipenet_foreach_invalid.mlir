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

// PipeNet record tables must not be empty.
func.func @empty_record_table() {
  ttl.pipenet_foreach_src attributes {
      // expected-error @+2 {{expected '<'}}
      // expected-error @below {{failed to parse TTL_PipeNetRecordsAttr parameter 'pipes'}}
      records = #ttl.pipenet_records<net 0 pipes []>} {
  ^bb0(%pipe: !ttl.selected_pipe_src):
    ttl.yield
  }
  func.return
}

// -----

// Foreach records must have a uniform pipe kind.
func.func @foreach_mixed_record_kind() {
  ttl.pipenet_foreach_src attributes {
      // expected-error @below {{all pipe records must be either point-to-point or collective}}
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
  // expected-error @below {{'ttl.pipe_transfer.create' op selected pipe operand must be a direct result of ttl.select_pipe_src or ttl.select_pipe_dst}}
  %transfer = ttl.pipe_transfer.create %pipe {kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.selected_pipe_src -> !ttl.pipe_transfer
  func.return
}

// -----

// Selected-pipe copy operands require a select operation or foreach block
// argument that supplies their record table.
func.func @selected_copy_requires_record_definition(
    %pipe: !ttl.selected_pipe_dst) {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve = ttl.cb_reserve %cb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.copy' op selected pipe operand must be defined by ttl.select_pipe_src, ttl.select_pipe_dst, ttl.pipenet_foreach_src, or ttl.pipenet_foreach_dst}}
  %copy = ttl.copy %pipe, %reserve
      : (!ttl.selected_pipe_dst,
         tensor<1x1x!ttcore.tile<32x32, bf16>>)
      -> !ttl.transfer_handle
  func.return
}

// -----

// Selected transfer kind must match the selected records kind.
func.func @selected_transfer_kind_mismatch() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %false = arith.constant false
  %src = ttl.select_pipe_src record(%c0) src (%c0, %c0) dst (%c1, %c0) to (%c1, %c0)
      num_dests (%c1) src_in_dst (%false)
      {records = #ttl.pipenet_records<net 0 pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0, isCollective = true>
      ]>} : !ttl.selected_pipe_src
  // expected-error @below {{'ttl.pipe_transfer.create' op selected pipe transfer kind must match the records kind}}
  %transfer = ttl.pipe_transfer.create %src {kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.selected_pipe_src -> !ttl.pipe_transfer
  func.return
}
