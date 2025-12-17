// RUN: ttlang-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @bind_cb
// CHECK: %[[CB:.*]] = ttl.bind_cb() {buffer_factor = 2 : i64, element_type = f32, shape = [1, 1]} : <[1, 1], f32, 2>
func.func @bind_cb() {
  %cb = ttl.bind_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2}
        : !ttl.cb<[1, 1], f32, 2>
  func.return
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout_interleaved = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>,
                      memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: func.func @copy_read_wait
// CHECK-SAME: (%[[T:.*]]: tensor<32x32xf32, #ttnn_layout>)
// CHECK: %[[CB:.*]] = ttl.bind_cb() {buffer_factor = 2 : i64, element_type = f32, shape = [1, 1]} : <[1, 1], f32, 2>
// CHECK: %[[XF:.*]] = ttl.copy %[[T]], %[[CB]] : (tensor<32x32xf32, #ttnn_layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
// CHECK: ttl.wait %[[XF]] : !ttl.transfer_handle<read>
func.func @copy_read_wait(%t: tensor<32x32xf32, #layout_interleaved>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2}
        : !ttl.cb<[1, 1], f32, 2>
  %xf = ttl.copy %t, %cb : (tensor<32x32xf32, #layout_interleaved>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
  ttl.wait %xf : !ttl.transfer_handle<read>
  func.return
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout_interleaved = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>,
                      memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: func.func @copy_write_wait
// CHECK-SAME: (%[[T:.*]]: tensor<32x32xf32, #ttnn_layout>)
// CHECK: %[[CB:.*]] = ttl.bind_cb() {buffer_factor = 2 : i64, element_type = f32, shape = [1, 1]} : <[1, 1], f32, 2>
// CHECK: %[[XF:.*]] = ttl.copy %[[CB]], %[[T]] : (!ttl.cb<[1, 1], f32, 2>, tensor<32x32xf32, #ttnn_layout>) -> !ttl.transfer_handle<write>
// CHECK: ttl.wait %[[XF]] : !ttl.transfer_handle<write>
func.func @copy_write_wait(%t: tensor<32x32xf32, #layout_interleaved>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2}
        : !ttl.cb<[1, 1], f32, 2>
  %xf = ttl.copy %cb, %t : (!ttl.cb<[1, 1], f32, 2>, tensor<32x32xf32, #layout_interleaved>) -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}

// -----

#l1 = #ttnn.buffer_type<l1>
#layout_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>,
               memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>

// CHECK-LABEL: func.func @copy_read_wait_tile_layout
// CHECK-SAME: (%[[T:.*]]: tensor<32x32xf32, #ttnn_layout>)
// CHECK: %[[CB:.*]] = ttl.bind_cb() {buffer_factor = 2 : i64, element_type = f32, shape = [1, 1]} : <[1, 1], f32, 2>
// CHECK: %[[XF:.*]] = ttl.copy %[[T]], %[[CB]] : (tensor<32x32xf32, #ttnn_layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
// CHECK: ttl.wait %[[XF]] : !ttl.transfer_handle<read>
func.func @copy_read_wait_tile_layout(%t: tensor<32x32xf32, #layout_tile>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2}
        : !ttl.cb<[1, 1], f32, 2>
  %xf = ttl.copy %t, %cb : (tensor<32x32xf32, #layout_tile>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
  ttl.wait %xf : !ttl.transfer_handle<read>
  func.return
}
