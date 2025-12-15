// RUN: ttlang-opt --convert-ttl-to-ttkernel --split-input-file %s | FileCheck %s
// Summary: Test CopyLowering patterns with runtime args.

// -----

// CopyLowering: Tensor to CB (read) generates NOC async read.
// Tensor args stay in function signature, CB args are converted.
// CHECK-LABEL: func.func @copy_tensor_to_cb(
// CHECK: ttkernel.get_compile_time_arg_val
// CHECK: %[[RT_IDX:.*]] = arith.constant 0 : index
// CHECK: ttkernel.get_common_arg_val(%[[RT_IDX]])
// CHECK: ttkernel.get_write_ptr
// CHECK: ttkernel.get_noc_addr
// CHECK: ttkernel.noc_async_read
// CHECK: ttkernel.noc_async_read_barrier
#l1_1 = #ttnn.buffer_type<l1>
#ttnn_layout_1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1_1>, <height_sharded>>
module {
  func.func @copy_tensor_to_cb(
      %tensor: tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_1>,
      %cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
      attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %xf = ttl.copy %tensor, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_1>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    return
  }
}

// -----

// CopyLowering: CB to tensor (write) generates NOC async write.
// CHECK-LABEL: func.func @copy_cb_to_tensor(
// CHECK: ttkernel.get_compile_time_arg_val
// CHECK: ttkernel.get_common_arg_val
// CHECK: ttkernel.get_read_ptr
// CHECK: ttkernel.get_noc_addr
// CHECK: ttkernel.noc_async_write
// CHECK: ttkernel.noc_async_write_barrier
#l1_2 = #ttnn.buffer_type<l1>
#ttnn_layout_2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1_2>, <height_sharded>>
module {
  func.func @copy_cb_to_tensor(
      %cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %tensor: tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_2>)
      attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %xf = ttl.copy %cb, %tensor : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_2>) -> !ttl.transfer_handle<write>
    ttl.wait %xf : !ttl.transfer_handle<write>
    return
  }
}

// -----

// CopyLowering: Multiple copies increment runtime arg index.
// CHECK-LABEL: func.func @multiple_copies(
// CHECK: %[[IDX0:.*]] = arith.constant 0 : index
// CHECK: ttkernel.get_common_arg_val(%[[IDX0]])
// CHECK: ttkernel.noc_async_read
// CHECK: %[[IDX1:.*]] = arith.constant 1 : index
// CHECK: ttkernel.get_common_arg_val(%[[IDX1]])
// CHECK: ttkernel.noc_async_read
#l1_3 = #ttnn.buffer_type<l1>
#ttnn_layout_3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1_3>, <height_sharded>>
module {
  func.func @multiple_copies(
      %t1: tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_3>,
      %t2: tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_3>,
      %cb1: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %cb2: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
      attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %xf1 = ttl.copy %t1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_3>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf1 : !ttl.transfer_handle<read>
    %xf2 = ttl.copy %t2, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_3>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf2 : !ttl.transfer_handle<read>
    return
  }
}

// -----

// CopyLowering: Read then write with correct barriers.
// CHECK-LABEL: func.func @read_then_write(
// CHECK: ttkernel.noc_async_read
// CHECK: ttkernel.noc_async_read_barrier
// CHECK: ttkernel.noc_async_write
// CHECK: ttkernel.noc_async_write_barrier
#l1_4 = #ttnn.buffer_type<l1>
#ttnn_layout_4 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1_4>, <height_sharded>>
module {
  func.func @read_then_write(
      %src: tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_4>,
      %dst: tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_4>,
      %cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
      attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %read = ttl.copy %src, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_4>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %read : !ttl.transfer_handle<read>
    %write = ttl.copy %cb, %dst : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_4>) -> !ttl.transfer_handle<write>
    ttl.wait %write : !ttl.transfer_handle<write>
    return
  }
}
