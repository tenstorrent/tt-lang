// RUN: ttlang-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @create_pipe_unicast
// CHECK: ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) : <src(0, 0) dst(1, 0) to(1, 0)>
func.func @create_pipe_unicast() {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0)>
  func.return
}

// -----

// CHECK-LABEL: func.func @create_pipe_multicast
// CHECK: ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) : <src(0, 0) dst(1, 0) to(1, 3)>
func.func @create_pipe_multicast() {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3)>
  func.return
}

// -----

// CHECK-LABEL: func.func @create_pipe_2d_grid
// CHECK: ttl.create_pipe src(0, 0) dst(0, 1) to(2, 3) : <src(0, 0) dst(0, 1) to(2, 3)>
func.func @create_pipe_2d_grid() {
  %p = ttl.create_pipe src(0, 0) dst(0, 1) to(2, 3) : !ttl.pipe<src(0, 0) dst(0, 1) to(2, 3)>
  func.return
}

// -----

// CHECK-LABEL: func.func @if_src_basic
// CHECK: %[[P:.*]] = ttl.create_pipe
// CHECK: ttl.if_src %[[P]] : <src(0, 0) dst(1, 0) to(1, 0)> {
// CHECK: }
func.func @if_src_basic() {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0)>
  ttl.if_src %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0)> {
  }
  func.return
}

// -----

// CHECK-LABEL: func.func @if_dst_basic
// CHECK: %[[P:.*]] = ttl.create_pipe
// CHECK: ttl.if_dst %[[P]] : <src(0, 0) dst(1, 0) to(1, 3)> {
// CHECK: }
func.func @if_dst_basic() {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3)>
  ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3)> {
  }
  func.return
}

// -----

// CHECK-LABEL: func.func @if_src_if_dst_combo
// CHECK: %[[P:.*]] = ttl.create_pipe
// CHECK: ttl.if_src %[[P]]
// CHECK: ttl.if_dst %[[P]]
func.func @if_src_if_dst_combo() {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3)>
  ttl.if_src %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3)> {
    // Source-side operations would go here
  }
  ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3)> {
    // Destination-side operations would go here
  }
  func.return
}

// -----

// CHECK-LABEL: func.func @copy_cb_to_pipe
// CHECK: %[[CB:.*]] = ttl.bind_cb
// CHECK: %[[P:.*]] = ttl.create_pipe
// CHECK: ttl.copy %[[CB]], %[[P]]
// CHECK: ttl.wait
func.func @copy_cb_to_pipe() {
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0)>
  %xf = ttl.copy %cb, %p : (!ttl.cb<[1, 1], f32, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0)>) -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}

// -----

// CHECK-LABEL: func.func @copy_pipe_to_cb
// CHECK: %[[CB:.*]] = ttl.bind_cb
// CHECK: %[[P:.*]] = ttl.create_pipe
// CHECK: ttl.copy %[[P]], %[[CB]]
// CHECK: ttl.wait
func.func @copy_pipe_to_cb() {
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0)>
  %xf = ttl.copy %p, %cb : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0)>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
  ttl.wait %xf : !ttl.transfer_handle<read>
  func.return
}
