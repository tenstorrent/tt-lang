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
