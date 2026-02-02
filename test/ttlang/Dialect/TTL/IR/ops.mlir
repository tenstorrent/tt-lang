// RUN: ttlang-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @bind_cb
// CHECK: %[[CB:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[1, 1], f32, 2>
func.func @bind_cb() {
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
  func.return
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout_interleaved = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>,
                      memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: func.func @copy_read_wait
// CHECK-SAME: (%[[T:.*]]: tensor<1x1x!ttcore.tile<32x32, f32>, #ttnn_layout>)
// CHECK: %[[CB:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[1, 1], f32, 2>
// CHECK: %[[XF:.*]] = ttl.copy %[[T]], %[[CB]] : (tensor<1x1x!ttcore.tile<32x32, f32>, #ttnn_layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
// CHECK: ttl.wait %[[XF]] : !ttl.transfer_handle<read>
func.func @copy_read_wait(%t: tensor<1x1x!ttcore.tile<32x32, f32>, #layout_interleaved>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
  %xf = ttl.copy %t, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout_interleaved>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
  ttl.wait %xf : !ttl.transfer_handle<read>
  func.return
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout_interleaved = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>,
                      memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: func.func @copy_write_wait
// CHECK-SAME: (%[[T:.*]]: tensor<1x1x!ttcore.tile<32x32, f32>, #ttnn_layout>)
// CHECK: %[[CB:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[1, 1], f32, 2>
// CHECK: %[[XF:.*]] = ttl.copy %[[CB]], %[[T]] : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #ttnn_layout>) -> !ttl.transfer_handle<write>
// CHECK: ttl.wait %[[XF]] : !ttl.transfer_handle<write>
func.func @copy_write_wait(%t: tensor<1x1x!ttcore.tile<32x32, f32>, #layout_interleaved>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
  %xf = ttl.copy %cb, %t : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout_interleaved>) -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}

// -----

#l1 = #ttnn.buffer_type<l1>
#layout_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>,
               memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>

// CHECK-LABEL: func.func @copy_read_wait_tile_layout
// CHECK-SAME: (%[[T:.*]]: tensor<1x1x!ttcore.tile<32x32, f32>, #ttnn_layout>)
// CHECK: %[[CB:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[1, 1], f32, 2>
// CHECK: %[[XF:.*]] = ttl.copy %[[T]], %[[CB]] : (tensor<1x1x!ttcore.tile<32x32, f32>, #ttnn_layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
// CHECK: ttl.wait %[[XF]] : !ttl.transfer_handle<read>
func.func @copy_read_wait_tile_layout(%t: tensor<1x1x!ttcore.tile<32x32, f32>, #layout_tile>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
  %xf = ttl.copy %t, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout_tile>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
  ttl.wait %xf : !ttl.transfer_handle<read>
  func.return
}

// -----

// CHECK-LABEL: func.func @copy_tile_basic
// CHECK: %[[CB:.*]] = ttl.bind_cb
// CHECK: %[[T_CB:.*]] = ttl.attach_cb %[[TENS:.*]], %[[CB]]
// CHECK: %[[RES:.*]] = ttl.compute
// CHECK: ^bb0(%[[T:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// CHECK:   %[[DTOK:.*]], %[[DTILE:.*]] = ttl.copy_tile %[[T]], %[[SRC_IDX:.*]], %[[DST_IDX:.*]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK:   ttl.yield %[[DTILE]] : !ttcore.tile<32x32, f32>
// CHECK: }
func.func @copy_tile_basic(%t_tensor: tensor<1x1x!ttcore.tile<32x32, f32>>, %src_idx: index, %dst_idx: index) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %t_attached = ttl.attach_cb %t_tensor, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%t_attached : tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%t_attached : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%t: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
    %dst, %dst_tile = ttl.copy_tile %t, %src_idx, %dst_idx : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
    ttl.yield %dst_tile : !ttcore.tile<32x32, f32>
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<1x1x!ttcore.tile<32x32, f32>>
}
