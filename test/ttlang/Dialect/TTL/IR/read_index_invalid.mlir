// Verifier rejection tests for ttl.read_index thread, ownership, coordinate,
// and element-type requirements.
// RUN: ttlang-opt --verify-diagnostics --split-input-file %s

// ttl.read_index requires an enclosing kernel thread.
func.func @read_index_without_kernel_thread(
    %block: tensor<1x1x!ttcore.tile<32x32, f32>>) {
  %zero = arith.constant 0 : index
  // expected-error @below {{must be inside a function with 'ttl.kernel_thread' attribute}}
  %index = ttl.read_index %block[%zero, %zero] : tensor<1x1x!ttcore.tile<32x32, f32>> -> index
  func.return
}

// -----

// ttl.read_index is unavailable in compute threads.
func.func @read_index_in_compute_thread(
    %block: tensor<1x1x!ttcore.tile<32x32, f32>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %zero = arith.constant 0 : index
  // expected-error @below {{is only allowed in data movement (noc) threads}}
  %index = ttl.read_index %block[%zero, %zero] : tensor<1x1x!ttcore.tile<32x32, f32>> -> index
  func.return
}

// -----

// ttl.read_index requires a block acquired from a dataflow buffer.
func.func @read_index_from_unacquired_block(
    %block: tensor<1x1x!ttcore.tile<32x32, f32>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %zero = arith.constant 0 : index
  // expected-error @below {{block must be a tensor view acquired from ttl.cb_wait}}
  %index = ttl.read_index %block[%zero, %zero] : tensor<1x1x!ttcore.tile<32x32, f32>> -> index
  func.return
}

// -----

// ttl.read_index requires a consumer block acquired by ttl.cb_wait.
func.func @read_index_from_reserved_block()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %block = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %zero = arith.constant 0 : index
  // expected-error @below {{block must be acquired from ttl.cb_wait, but traces to ttl.cb_reserve}}
  %index = ttl.read_index %block[%zero, %zero] : tensor<1x1x!ttcore.tile<32x32, f32>> -> index
  func.return
}

// -----

// ttl.read_index requires one coordinate per block dimension.
func.func @read_index_coordinate_count_mismatch()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %zero = arith.constant 0 : index
  // expected-error @below {{coordinate count (1) must match block tensor rank (2)}}
  %index = ttl.read_index %block[%zero] : tensor<1x1x!ttcore.tile<32x32, f32>> -> index
  func.return
}

// -----

// ttl.read_index accepts only f32 and bf16 block elements.
func.func @read_index_unsupported_element_type()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[16], i32, 2>
  %block = ttl.cb_wait %cb : <[16], i32, 2> -> tensor<16xi32>
  %zero = arith.constant 0 : index
  // expected-error @below {{requires an f32 or bf16 block element type, got 'i32'}}
  %index = ttl.read_index %block[%zero] : tensor<16xi32> -> index
  func.return
}
