// RUN: ttlang-opt %s --split-input-file -verify-diagnostics

// Test: cannot copy directly between two pipes.
func.func @pipe_to_pipe_copy() {
  %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p2 = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>
  // expected-error @+1 {{'ttl.copy' op cannot copy directly between pipes}}
  %xf = ttl.copy %p1, %p2 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>) -> !ttl.transfer_handle
  ttl.wait %xf : !ttl.transfer_handle
  func.return
}

// -----

// Test: pipe receive without a reserved destination DFB slot.
func.func @pipe_receive_without_reserve(%t: tensor<32x32xf32>) {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  // expected-error @+1 {{'ttl.copy' op pipe receive requires a cb_reserve destination}}
  %xf = ttl.copy %p, %t : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<32x32xf32>) -> !ttl.transfer_handle
  ttl.wait %xf : !ttl.transfer_handle
  func.return
}

// -----

// Test: an internal pipe transfer must deliver at least one original DFB block.
func.func @pipe_transfer_block_span_positive() {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  // expected-error @+1 {{'ttl.pipe_transfer.create' op attribute 'block_span' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive}}
  %transfer = ttl.pipe_transfer.create %p {block_span = 0 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  func.return
}

// -----

// Test: an internal pipe transfer destination group depth must be positive.
func.func @pipe_transfer_destination_group_depth_positive() {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  // expected-error @+1 {{'ttl.pipe_transfer.create' op attribute 'destination_group_depth' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive}}
  %transfer = ttl.pipe_transfer.create %p {destination_group_depth = 0 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  func.return
}

// -----

// A selected record identifies one scalar transfer at runtime, so grouping
// must remain on statically identified pipe transfers.
func.func @selected_pipe_transfer_block_span() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %false = arith.constant false
  %selected = ttl.select_pipe_src record(%c0) src (%c0, %c0) dst (%c1, %c0) to (%c1, %c0)
      num_dests (%c1) src_in_dst (%false)
      {records = #ttl.pipenet_records<net 0 pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>
      ]>} : !ttl.selected_pipe_src
  // expected-error @below {{'ttl.pipe_transfer.create' op selected pipe transfer block_span must be 1}}
  %transfer = ttl.pipe_transfer.create %selected {block_span = 2 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.selected_pipe_src -> !ttl.pipe_transfer
  func.return
}

// -----

// Selected record-table lowering allocates synchronization and address state
// per scalar record rather than per grouped destination slot.
func.func @selected_pipe_transfer_destination_group_depth() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %false = arith.constant false
  %selected = ttl.select_pipe_dst record(%c0) src (%c0, %c0) dst (%c1, %c0) to (%c1, %c0)
      num_dests (%c1) src_in_dst (%false)
      {records = #ttl.pipenet_records<net 0 pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>
      ]>} : !ttl.selected_pipe_dst
  // expected-error @below {{'ttl.pipe_transfer.create' op selected pipe transfer destination_group_depth must be 1}}
  %transfer = ttl.pipe_transfer.create %selected {destination_group_depth = 2 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.selected_pipe_dst -> !ttl.pipe_transfer
  func.return
}

// -----
// Test: point-to-point pipe transfer cannot target multiple receivers.
func.func @pipe_transfer_point_to_point_multi_receiver() {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
  // expected-error @+1 {{'ttl.pipe_transfer.create' op point_to_point transfer requires one receiver}}
  %transfer = ttl.pipe_transfer.create %p {kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> -> !ttl.pipe_transfer
  func.return
}

// -----

// Test: pipe transfer send result is a write handle.
func.func @pipe_transfer_send_requires_write_handle() {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %p {kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  // expected-error @+1 {{'ttl.pipe_transfer.send' op requires a write transfer handle result}}
  %xf = ttl.pipe_transfer.send %transfer, %cb
      : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      -> !ttl.transfer_handle<read>
  func.return
}

// -----

// Test: negative source coordinates.
// expected-error @+1 {{'ttl.create_pipe' op source coordinates must be non-negative}}
%p = ttl.create_pipe src(-1, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(-1, 0) dst(1, 0) to(1, 0) net 0>

// -----

// Test: negative destination coordinates.
// expected-error @+1 {{'ttl.create_pipe' op destination coordinates must be non-negative}}
%p = ttl.create_pipe src(0, 0) dst(-1, 0) to(-1, 0) net 0 : !ttl.pipe<src(0, 0) dst(-1, 0) to(-1, 0) net 0>

// -----

// Test: attributes must match result pipe type.
// expected-error @+1 {{'ttl.create_pipe' op attributes must match result pipe type}}
%p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>

// -----

// Test: dst start must not exceed dst end on x.
// expected-error @+1 {{'ttl.create_pipe' op destination start must not exceed destination end on any axis}}
%p = ttl.create_pipe src(0, 0) dst(3, 0) to(0, 0) net 0 : !ttl.pipe<src(0, 0) dst(3, 0) to(0, 0) net 0>

// -----

// Test: dst start must not exceed dst end on y.
// expected-error @+1 {{'ttl.create_pipe' op destination start must not exceed destination end on any axis}}
%p = ttl.create_pipe src(0, 0) dst(0, 5) to(0, 2) net 0 : !ttl.pipe<src(0, 0) dst(0, 5) to(0, 2) net 0>

// -----

// Test: explicit point-to-point metadata cannot contradict a multi-receiver pipe.
// expected-error @+1 {{'ttl.create_pipe' op isCollective=false is invalid for a multi-receiver pipe}}
%p = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0 {isCollective = false} : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
