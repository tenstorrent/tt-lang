// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// Verify the structural, type, and capacity contracts for byte-counted copies.

// A local DFB-to-DFB copy requires an explicit transfer size.
func.func @missing_byte_count()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[14, 1], !ttcore.tile<1x32, bf16>, 1>
  %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %src_wait = ttl.cb_wait %src_dfb
      : <[14, 1], !ttcore.tile<1x32, bf16>, 1>
      -> tensor<14x1x!ttcore.tile<1x32, bf16>>
  %src = ttl.attach_cb %src_wait, %src_dfb
      : (tensor<14x1x!ttcore.tile<1x32, bf16>>,
         !ttl.cb<[14, 1], !ttcore.tile<1x32, bf16>, 1>)
      -> tensor<14x1x!ttcore.tile<1x32, bf16>>
  %dst_reserve = ttl.cb_reserve %dst_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %dst = ttl.attach_cb %dst_reserve, %dst_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{dataflow-buffer block copies require an explicit byte_count}}
  %copy = ttl.copy %src, %dst
      : (tensor<14x1x!ttcore.tile<1x32, bf16>>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>)
      -> !ttl.transfer_handle<read>
  func.return
}

// -----

// A local DFB-to-DFB copy returns a read handle for NoC completion.
func.func @block_copy_requires_read_handle()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %src_wait = ttl.cb_wait %src_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %src = ttl.attach_cb %src_wait, %src_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %dst_reserve = ttl.cb_reserve %dst_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %dst = ttl.attach_cb %dst_reserve, %dst_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{dataflow-buffer block copy requires !ttl.transfer_handle<read> result}}
  %copy = ttl.copy %src, %dst {byte_count = 896 : i64}
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>)
      -> !ttl.receive_request
  func.return
}

// -----

// The source must be the current readable block from the source DFB.
func.func @source_must_be_wait_view()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %src_reserve = ttl.cb_reserve %src_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %src = ttl.attach_cb %src_reserve, %src_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %dst_reserve = ttl.cb_reserve %dst_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %dst = ttl.attach_cb %dst_reserve, %dst_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{dataflow-buffer copy source must be the exact view returned by ttl.cb_wait}}
  %copy = ttl.copy %src, %dst {byte_count = 896 : i64}
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>)
      -> !ttl.transfer_handle<read>
  func.return
}

// -----

// The destination must be the current writable block from the destination DFB.
func.func @destination_must_be_reserve_view()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %src_wait = ttl.cb_wait %src_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %src = ttl.attach_cb %src_wait, %src_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %dst_wait = ttl.cb_wait %dst_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %dst = ttl.attach_cb %dst_wait, %dst_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{dataflow-buffer copy destination must be the exact view returned by ttl.cb_reserve}}
  %copy = ttl.copy %src, %dst {byte_count = 896 : i64}
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>)
      -> !ttl.transfer_handle<read>
  func.return
}

// -----

// A raw local copy cannot read and write the same DFB interface.
func.func @distinct_dataflow_buffers_required()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %src_wait = ttl.cb_wait %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %src = ttl.attach_cb %src_wait, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %dst_reserve = ttl.cb_reserve %dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %dst = ttl.attach_cb %dst_reserve, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{byte-counted copy requires distinct source and destination dataflow buffers}}
  %copy = ttl.copy %src, %dst {byte_count = 896 : i64}
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>)
      -> !ttl.transfer_handle<read>
  func.return
}

// -----

// Raw bytes cannot be reinterpreted as a different data format.
func.func @matching_data_formats_required()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[14, 1], !ttcore.tile<1x32, bf16>, 1>
  %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %src_wait = ttl.cb_wait %src_dfb
      : <[14, 1], !ttcore.tile<1x32, bf16>, 1>
      -> tensor<14x1x!ttcore.tile<1x32, bf16>>
  %src = ttl.attach_cb %src_wait, %src_dfb
      : (tensor<14x1x!ttcore.tile<1x32, bf16>>,
         !ttl.cb<[14, 1], !ttcore.tile<1x32, bf16>, 1>)
      -> tensor<14x1x!ttcore.tile<1x32, bf16>>
  %dst_reserve = ttl.cb_reserve %dst_dfb
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %dst = ttl.attach_cb %dst_reserve, %dst_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{byte-counted copy data formats must match}}
  %copy = ttl.copy %src, %dst {byte_count = 896 : i64}
      : (tensor<14x1x!ttcore.tile<1x32, bf16>>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle<read>
  func.return
}

// -----

// The requested bytes must fit the acquired source view.
func.func @source_view_capacity_required()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[14, 1], !ttcore.tile<1x32, bf16>, 1>
  %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %src_wait = ttl.cb_wait %src_dfb
      : <[14, 1], !ttcore.tile<1x32, bf16>, 1>
      -> tensor<14x1x!ttcore.tile<1x32, bf16>>
  %src = ttl.attach_cb %src_wait, %src_dfb
      : (tensor<14x1x!ttcore.tile<1x32, bf16>>,
         !ttl.cb<[14, 1], !ttcore.tile<1x32, bf16>, 1>)
      -> tensor<14x1x!ttcore.tile<1x32, bf16>>
  %dst_reserve = ttl.cb_reserve %dst_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %dst = ttl.attach_cb %dst_reserve, %dst_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{byte_count 897 exceeds source acquired dataflow-buffer view capacity 896}}
  %copy = ttl.copy %src, %dst {byte_count = 897 : i64}
      : (tensor<14x1x!ttcore.tile<1x32, bf16>>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>)
      -> !ttl.transfer_handle<read>
  func.return
}

// -----

// The requested bytes must fit the acquired destination view.
func.func @destination_view_capacity_required()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[14, 1], !ttcore.tile<1x32, bf16>, 1>
  %src_wait = ttl.cb_wait %src_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %src = ttl.attach_cb %src_wait, %src_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %dst_reserve = ttl.cb_reserve %dst_dfb
      : <[14, 1], !ttcore.tile<1x32, bf16>, 1>
      -> tensor<14x1x!ttcore.tile<1x32, bf16>>
  %dst = ttl.attach_cb %dst_reserve, %dst_dfb
      : (tensor<14x1x!ttcore.tile<1x32, bf16>>,
         !ttl.cb<[14, 1], !ttcore.tile<1x32, bf16>, 1>)
      -> tensor<14x1x!ttcore.tile<1x32, bf16>>
  // expected-error @below {{byte_count 897 exceeds destination acquired dataflow-buffer view capacity 896}}
  %copy = ttl.copy %src, %dst {byte_count = 897 : i64}
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         tensor<14x1x!ttcore.tile<1x32, bf16>>)
      -> !ttl.transfer_handle<read>
  func.return
}

// -----

// A zero-byte transfer has no supported asynchronous-copy meaning.
func.func @byte_count_must_be_positive() {
  %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  // expected-error @below {{'ttl.copy' op attribute 'byte_count' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive}}
  %send = ttl.copy %src_dfb, %pipe {byte_count = 0 : i64}
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
         !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
      -> !ttl.transfer_handle<write>
  func.return
}

// -----

// The transfer size must fit the unsigned 32-bit TTKernel NoC operand.
func.func @byte_count_must_fit_noc_size() {
  %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[2097153, 1], !ttcore.tile<32x32, bf16>, 1>
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  // expected-error @below {{'ttl.copy' op attribute 'byte_count' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive whose maximum value is 4294967295}}
  %send = ttl.copy %src_dfb, %pipe {byte_count = 4294967296 : i64}
      : (!ttl.cb<[2097153, 1], !ttcore.tile<32x32, bf16>, 1>,
         !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
      -> !ttl.transfer_handle<write>
  func.return
}

// -----

// A PipeNet send cannot read beyond one source DFB block.
func.func @pipe_source_capacity_required() {
  %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  // expected-error @below {{byte_count 2049 exceeds source dataflow-buffer block capacity 2048}}
  %send = ttl.copy %src_dfb, %pipe {byte_count = 2049 : i64}
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
         !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
      -> !ttl.transfer_handle<write>
  func.return
}

// -----

// A PipeNet receive cannot write beyond its acquired destination view.
func.func @pipe_receiver_capacity_required() {
  %dst_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[14, 1], !ttcore.tile<1x32, bf16>, 1>
  %dst_reserve = ttl.cb_reserve %dst_dfb
      : <[14, 1], !ttcore.tile<1x32, bf16>, 1>
      -> tensor<14x1x!ttcore.tile<1x32, bf16>>
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  // expected-error @below {{byte_count 897 exceeds destination acquired dataflow-buffer view capacity 896}}
  %receive = ttl.copy %pipe, %dst_reserve {byte_count = 897 : i64}
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
         tensor<14x1x!ttcore.tile<1x32, bf16>>)
      -> !ttl.receive_request
  func.return
}

// -----

// Byte-counted DFB copies require a tiled DFB data format.
func.func @block_copy_requires_tiled_elements()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], f32, 1>
  %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], f32, 1>
  %src_wait = ttl.cb_wait %src_dfb
      : <[1, 1], f32, 1> -> tensor<1x1xf32>
  %src = ttl.attach_cb %src_wait, %src_dfb
      : (tensor<1x1xf32>, !ttl.cb<[1, 1], f32, 1>) -> tensor<1x1xf32>
  %dst_reserve = ttl.cb_reserve %dst_dfb
      : <[1, 1], f32, 1> -> tensor<1x1xf32>
  %dst = ttl.attach_cb %dst_reserve, %dst_dfb
      : (tensor<1x1xf32>, !ttl.cb<[1, 1], f32, 1>) -> tensor<1x1xf32>
  // expected-error @below {{byte-counted dataflow-buffer copies require tiled element types}}
  %copy = ttl.copy %src, %dst {byte_count = 4 : i64}
      : (tensor<1x1xf32>, tensor<1x1xf32>)
      -> !ttl.transfer_handle<read>
  func.return
}

// -----

// Ordinary tensor copies continue to use tile counts rather than byte counts.
func.func @byte_count_rejects_ordinary_tensor_copy() {
  %src = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %dst = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{byte_count is supported only for dataflow-buffer block copies and pipe copies}}
  %copy = ttl.copy %src, %dst {byte_count = 896 : i64}
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>)
      -> !ttl.transfer_handle<read>
  func.return
}
