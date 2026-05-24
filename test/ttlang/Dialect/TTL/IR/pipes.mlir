// RUN: ttlang-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @create_pipe_point_to_point
// CHECK: ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : <src(0, 0) dst(1, 0) to(1, 0) net 0>
func.func @create_pipe_point_to_point() {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  func.return
}

// -----

// CHECK-LABEL: func.func @create_pipe_collective
// CHECK: ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : <src(0, 0) dst(1, 0) to(1, 3) net 0>
func.func @create_pipe_collective() {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  func.return
}

// -----

// CHECK-LABEL: func.func @create_pipe_single_receiver_collective
// CHECK: ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 {isCollective = true} : <src(0, 0) dst(1, 0) to(1, 0) net 0>
func.func @create_pipe_single_receiver_collective() {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 {isCollective = true} : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  func.return
}

// -----

// CHECK-LABEL: func.func @create_pipe_2d_grid
// CHECK: ttl.create_pipe src(0, 0) dst(0, 1) to(2, 3) net 0 : <src(0, 0) dst(0, 1) to(2, 3) net 0>
func.func @create_pipe_2d_grid() {
  %p = ttl.create_pipe src(0, 0) dst(0, 1) to(2, 3) net 0 : !ttl.pipe<src(0, 0) dst(0, 1) to(2, 3) net 0>
  func.return
}

// -----

// CHECK-LABEL: func.func @if_src_basic
// CHECK: %[[P:.*]] = ttl.create_pipe
// CHECK: ttl.if_src %[[P]] : <src(0, 0) dst(1, 0) to(1, 0) net 0> {
// CHECK: }
func.func @if_src_basic() {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  ttl.if_src %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
  }
  func.return
}

// -----

// CHECK-LABEL: func.func @if_dst_basic
// CHECK: %[[P:.*]] = ttl.create_pipe
// CHECK: ttl.if_dst %[[P]] : <src(0, 0) dst(1, 0) to(1, 3) net 0> {
// CHECK: }
func.func @if_dst_basic() {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0> {
  }
  func.return
}

// -----

// CHECK-LABEL: func.func @if_src_if_dst_combo
// CHECK: %[[P:.*]] = ttl.create_pipe
// CHECK: ttl.if_src %[[P]]
// CHECK: ttl.if_dst %[[P]]
func.func @if_src_if_dst_combo() {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  ttl.if_src %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0> {
    // Source-side operations would go here
  }
  ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0> {
    // Destination-side operations would go here
  }
  func.return
}

// -----

// CHECK-LABEL: func.func @copy_cb_to_pipe
// CHECK: %[[CB:.*]] = ttl.bind_cb
// CHECK: %[[P:.*]] = ttl.create_pipe
// CHECK: %[[XF:.*]] = ttl.copy %[[CB]], %[[P]]
// CHECK: ttl.wait %[[XF]]
func.func @copy_cb_to_pipe() {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], f32, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %xf = ttl.copy %cb, %p : (!ttl.cb<[1, 1], f32, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}

// -----

// CHECK-LABEL: func.func @copy_pipe_to_cb
// CHECK: %[[CB:.*]] = ttl.bind_cb
// CHECK: %[[P:.*]] = ttl.create_pipe
// CHECK: %[[RECV:.*]] = ttl.cb_reserve %[[CB]]
// CHECK: %[[XF:.*]] = ttl.copy %[[P]], %[[RECV]]
// CHECK: ttl.wait %[[XF]]
// CHECK: ttl.cb_push %[[CB]]
func.func @copy_pipe_to_cb() {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], f32, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %recv = ttl.cb_reserve %cb : <[1, 1], f32, 2> -> tensor<1x1xf32>
  %xf = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1xf32>) -> !ttl.transfer_handle
  ttl.wait %xf : !ttl.transfer_handle
  ttl.cb_push %cb : <[1, 1], f32, 2>
  func.return
}

// -----

// CHECK-LABEL: func.func @pipe_transfer_ir
// CHECK: %[[P:.*]] = ttl.create_pipe
// CHECK: %[[TRANSFER:.*]] = ttl.pipe_transfer.create %[[P]]
// CHECK-SAME: expectedReceivers = 1 : i64
// CHECK-SAME: kind = #ttl.pipe_transfer_kind<point_to_point>
// CHECK: %[[TOKEN:.*]] = ttl.pipe_transfer.post %[[TRANSFER]]
// CHECK: %[[XF:.*]] = ttl.pipe_transfer.send %[[TRANSFER]]
// CHECK: ttl.wait %[[XF]]
// CHECK: ttl.pipe_transfer.wait %[[TOKEN]]
func.func @pipe_transfer_ir() {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], f32, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %p {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  %recv = ttl.cb_reserve %cb : <[1, 1], f32, 2> -> tensor<1x1xf32>
  %token = ttl.pipe_transfer.post %transfer, %recv
      : (!ttl.pipe_transfer, tensor<1x1xf32>) -> !ttl.pipe_token<net 0>
  %xf = ttl.pipe_transfer.send %transfer, %cb
      : (!ttl.pipe_transfer, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
  func.return
}

// -----

// Test: pipe transfer wait accepts a token carried through a loop iter_arg.
// CHECK-LABEL: func.func @pipe_transfer_loop_carried_token
// CHECK: %[[TOKEN_INIT:.*]] = ttl.pipe_transfer.post
// CHECK: %[[TOKEN:.*]] = scf.for
// CHECK: ttl.pipe_transfer.wait %[[TOKEN]]
func.func @pipe_transfer_loop_carried_token() {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], f32, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %p {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  %recv = ttl.cb_reserve %cb : <[1, 1], f32, 2> -> tensor<1x1xf32>
  %token_init = ttl.pipe_transfer.post %transfer, %recv
      : (!ttl.pipe_transfer, tensor<1x1xf32>) -> !ttl.pipe_token<net 0>
  %token = scf.for %iter = %zero to %one step %one
      iter_args(%token_arg = %token_init) -> (!ttl.pipe_token<net 0>) {
    scf.yield %token_arg : !ttl.pipe_token<net 0>
  }
  ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
  func.return
}

// -----

// Test: pipe transfer create prints both transfer kind enum values symbolically.
// CHECK-LABEL: func.func @pipe_transfer_kind_printing
// CHECK: %[[PTP_PIPE:.*]] = ttl.create_pipe
// CHECK: ttl.pipe_transfer.create %[[PTP_PIPE]]
// CHECK-SAME: expectedReceivers = 1 : i64
// CHECK-SAME: kind = #ttl.pipe_transfer_kind<point_to_point>
// CHECK: %[[COLLECTIVE_PIPE:.*]] = ttl.create_pipe
// CHECK: ttl.pipe_transfer.create %[[COLLECTIVE_PIPE]]
// CHECK-SAME: expectedReceivers = 2 : i64
// CHECK-SAME: kind = #ttl.pipe_transfer_kind<collective>
func.func @pipe_transfer_kind_printing() {
  %ptp_pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %ptp_transfer = ttl.pipe_transfer.create %ptp_pipe {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer

  %collective_pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
  %collective_transfer = ttl.pipe_transfer.create %collective_pipe {expectedReceivers = 2 : i64, kind = #ttl.pipe_transfer_kind<collective>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> -> !ttl.pipe_transfer

  func.return
}
