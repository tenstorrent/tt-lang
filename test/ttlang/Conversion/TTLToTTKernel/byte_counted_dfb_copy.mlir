// RUN: ttlang-opt %s --convert-ttl-to-ttkernel --canonicalize -cse | FileCheck %s

// CHECK-LABEL: func.func @copy_valid_prefix_between_dfb_blocks
// CHECK-DAG: %[[NOC:.*]] = arith.constant 0 : i8
// CHECK-DAG: %[[SIZE:.*]] = arith.constant 896 : i32
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[DST_DFB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_read_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_ADDR:.*]] = ttkernel.get_write_ptr(%[[DST_DFB]])
// CHECK: %[[SRC_X:.*]] = ttkernel.my_x(%[[NOC]])
// CHECK: %[[SRC_Y:.*]] = ttkernel.my_y(%[[NOC]])
// CHECK: ttkernel.noc_async_read core[%[[SRC_X]], %[[SRC_Y]]], %[[SRC_ADDR]], %[[DST_ADDR]], %[[SIZE]], noc %[[NOC]]
// CHECK: ttkernel.noc_async_read_barrier(%[[NOC]])
func.func @copy_valid_prefix_between_dfb_blocks()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %compact_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[14, 1], !ttcore.tile<1x32, bf16>, 1>
  %full_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %compact_wait = ttl.cb_wait %compact_dfb
      : <[14, 1], !ttcore.tile<1x32, bf16>, 1>
      -> tensor<14x1x!ttcore.tile<1x32, bf16>>
  %compact_block = ttl.attach_cb %compact_wait, %compact_dfb
      : (tensor<14x1x!ttcore.tile<1x32, bf16>>,
         !ttl.cb<[14, 1], !ttcore.tile<1x32, bf16>, 1>)
      -> tensor<14x1x!ttcore.tile<1x32, bf16>>
  %full_reserve = ttl.cb_reserve %full_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %full_block = ttl.attach_cb %full_reserve, %full_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %transfer = ttl.copy %compact_block, %full_block {byte_count = 896 : i64}
      : (tensor<14x1x!ttcore.tile<1x32, bf16>>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>)
      -> !ttl.transfer_handle<read>
  ttl.wait %transfer : !ttl.transfer_handle<read>
  func.return
}

// CHECK-LABEL: func.func @copy_valid_prefix_between_writer_dfb_blocks
// CHECK-DAG: %[[WRITER_NOC:.*]] = arith.constant 1 : i8
// CHECK-DAG: %[[WRITER_SIZE:.*]] = arith.constant 896 : i32
// CHECK: %[[WRITER_SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[WRITER_DST_DFB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: %[[WRITER_SRC_ADDR:.*]] = ttkernel.get_read_ptr(%[[WRITER_SRC_DFB]])
// CHECK: %[[WRITER_DST_ADDR:.*]] = ttkernel.get_write_ptr(%[[WRITER_DST_DFB]])
// CHECK: %[[DST_X:.*]] = ttkernel.my_x(%[[WRITER_NOC]])
// CHECK: %[[DST_Y:.*]] = ttkernel.my_y(%[[WRITER_NOC]])
// CHECK: ttkernel.noc_async_write %[[WRITER_SRC_ADDR]], core[%[[DST_X]], %[[DST_Y]]], %[[WRITER_DST_ADDR]], %[[WRITER_SIZE]], noc %[[WRITER_NOC]]
// CHECK: ttkernel.noc_async_write_barrier(%[[WRITER_NOC]])
func.func @copy_valid_prefix_between_writer_dfb_blocks()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                ttl.noc_index = 1 : i64} {
  %compact_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[14, 1], !ttcore.tile<1x32, bf16>, 1>
  %full_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %compact_wait = ttl.cb_wait %compact_dfb
      : <[14, 1], !ttcore.tile<1x32, bf16>, 1>
      -> tensor<14x1x!ttcore.tile<1x32, bf16>>
  %compact_block = ttl.attach_cb %compact_wait, %compact_dfb
      : (tensor<14x1x!ttcore.tile<1x32, bf16>>,
         !ttl.cb<[14, 1], !ttcore.tile<1x32, bf16>, 1>)
      -> tensor<14x1x!ttcore.tile<1x32, bf16>>
  %full_reserve = ttl.cb_reserve %full_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %full_block = ttl.attach_cb %full_reserve, %full_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %transfer = ttl.copy %compact_block, %full_block {byte_count = 896 : i64}
      : (tensor<14x1x!ttcore.tile<1x32, bf16>>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>)
      -> !ttl.transfer_handle<read>
  ttl.wait %transfer : !ttl.transfer_handle<read>
  func.return
}
