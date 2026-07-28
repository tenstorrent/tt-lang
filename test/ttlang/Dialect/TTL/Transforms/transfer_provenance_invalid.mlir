// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -ttl-verify-transfer-provenance

// This file tests non-local transfer provenance diagnostics.

// A receive post requires a transfer created by ttl.pipe_transfer.create.
func.func @post_requires_created_transfer() {
  %transfer = builtin.unrealized_conversion_cast to !ttl.pipe_transfer
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst = ttl.cb_reserve %cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{'ttl.pipe_transfer.post' op requires every possible transfer value to derive from the same ttl.pipe_transfer.create}}
  %token = ttl.pipe_transfer.post %transfer, %dst
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.pipe_token<net 0>
  func.return
}

// -----

// A receive post requires storage reserved from a destination dataflow buffer.
func.func @post_requires_reserved_destination() {
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %pipe {
      expectedReceivers = 1 : i64,
      kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
      -> !ttl.pipe_transfer
  %dst = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{'ttl.pipe_transfer.post' op requires a cb_reserve destination}}
  %token = ttl.pipe_transfer.post %transfer, %dst
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.pipe_token<net 0>
  func.return
}

// -----

// A post token and its transfer must reference the same PipeNet.
func.func @post_token_net_mismatch() {
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %pipe {
      expectedReceivers = 1 : i64,
      kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
      -> !ttl.pipe_transfer
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst = ttl.cb_reserve %cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{'ttl.pipe_transfer.post' op token pipeNetId must match transfer pipeNetId}}
  %token = ttl.pipe_transfer.post %transfer, %dst
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.pipe_token<net 1>
  func.return
}

// -----

// A send requires a transfer created by ttl.pipe_transfer.create.
func.func @send_requires_created_transfer() {
  %transfer = builtin.unrealized_conversion_cast to !ttl.pipe_transfer
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  // expected-error @below {{'ttl.pipe_transfer.send' op requires every possible transfer value to derive from the same ttl.pipe_transfer.create}}
  %handle = ttl.pipe_transfer.send %transfer, %cb
      : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      -> !ttl.transfer_handle<write>
  func.return
}

// -----

// A pipe wait requires a token produced by a receive post in the same PipeNet.
func.func @pipe_wait_requires_post_token() {
  %token = builtin.unrealized_conversion_cast to !ttl.pipe_token<net 0>
  // expected-error @below {{'ttl.pipe_transfer.wait' op requires every possible token value to derive from a ttl.pipe_transfer.post in the same PipeNet}}
  ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
  func.return
}

// -----

// A transfer wait requires a recognized asynchronous transfer producer.
func.func @wait_requires_transfer_source() {
  %handle = builtin.unrealized_conversion_cast to !ttl.transfer_handle<write>
  // expected-error @below {{'ttl.wait' op expects operand to be derived from ttl.copy or ttl.pipe_transfer.send}}
  ttl.wait %handle : !ttl.transfer_handle<write>
  func.return
}

// -----

// A handle routed through a tensor container still requires a recognized
// asynchronous transfer producer.
func.func @wait_container_requires_transfer_source(
    %handle: !ttl.transfer_handle<read>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %handles = tensor.empty(%one) : tensor<?x!ttl.transfer_handle<read>>
  %inserted = tensor.insert %handle into %handles[%zero]
      : tensor<?x!ttl.transfer_handle<read>>
  %extracted = tensor.extract %inserted[%zero]
      : tensor<?x!ttl.transfer_handle<read>>
  // expected-error @below {{'ttl.wait' op expects operand to be derived from ttl.copy or ttl.pipe_transfer.send}}
  ttl.wait %extracted : !ttl.transfer_handle<read>
  func.return
}

// -----

// A valid copy at another tensor index does not validate the selected handle.
#indexed_layout = #ttl.layout<shape = [1, 1],
    element_type = !ttcore.tile<32x32, f32>, buffer = dram,
    grid = [1, 1], memory = interleaved>
func.func @wait_selected_container_element_requires_transfer_source(
    %tensor: tensor<1x1x!ttcore.tile<32x32, f32>, #indexed_layout>,
    %invalid: !ttl.transfer_handle<read>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %copy = ttl.copy %tensor, %cb
      : (tensor<1x1x!ttcore.tile<32x32, f32>, #indexed_layout>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      -> !ttl.transfer_handle<read>
  %handles = tensor.empty(%two) : tensor<?x!ttl.transfer_handle<read>>
  %with_invalid = tensor.insert %invalid into %handles[%zero]
      : tensor<?x!ttl.transfer_handle<read>>
  %with_copy = tensor.insert %copy into %with_invalid[%one]
      : tensor<?x!ttl.transfer_handle<read>>
  %selected = tensor.extract %with_copy[%zero]
      : tensor<?x!ttl.transfer_handle<read>>
  // expected-error @below {{'ttl.wait' op expects operand to be derived from ttl.copy or ttl.pipe_transfer.send}}
  ttl.wait %selected : !ttl.transfer_handle<read>
  func.return
}

// -----

// Every control-flow predecessor must provide a recognized transfer handle.
#if_layout = #ttl.layout<shape = [1, 1],
    element_type = !ttcore.tile<32x32, f32>, buffer = dram,
    grid = [1, 1], memory = interleaved>
func.func @wait_control_flow_requires_transfer_source(
    %condition: i1,
    %tensor: tensor<1x1x!ttcore.tile<32x32, f32>, #if_layout>,
    %invalid: !ttl.transfer_handle<read>) {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %copy = ttl.copy %tensor, %cb
      : (tensor<1x1x!ttcore.tile<32x32, f32>, #if_layout>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      -> !ttl.transfer_handle<read>
  %handle = scf.if %condition -> (!ttl.transfer_handle<read>) {
    scf.yield %copy : !ttl.transfer_handle<read>
  } else {
    scf.yield %invalid : !ttl.transfer_handle<read>
  }
  // expected-error @below {{'ttl.wait' op expects operand to be derived from ttl.copy or ttl.pipe_transfer.send}}
  ttl.wait %handle : !ttl.transfer_handle<read>
  func.return
}

// -----

// A merged wait cannot combine a pipe send with a transfer that requires a
// NOC write barrier.
#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                     buffer = dram, grid = [1, 1], memory = interleaved>
func.func @wait_requires_uniform_semantics(
    %condition: i1,
    %tensor: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) {
  %zero = arith.constant 0 : index
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %slice = ttl.tensor_slice %tensor[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
      -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
  %copy = ttl.copy %cb, %slice
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
         tensor<1x1x!ttcore.tile<32x32, f32>, #layout>)
      -> !ttl.transfer_handle<write>
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %pipe {
      expectedReceivers = 1 : i64,
      kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
      -> !ttl.pipe_transfer
  %send = ttl.pipe_transfer.send %transfer, %cb
      : (!ttl.pipe_transfer,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      -> !ttl.transfer_handle<write>
  %handle = scf.if %condition -> (!ttl.transfer_handle<write>) {
    scf.yield %copy : !ttl.transfer_handle<write>
  } else {
    scf.yield %send : !ttl.transfer_handle<write>
  }
  // expected-error @below {{'ttl.wait' op requires all possible sources to have the same wait semantics}}
  ttl.wait %handle : !ttl.transfer_handle<write>
  func.return
}
