// Verifier rejection tests for ttl.raw_element_read and ttl.raw_element_write.
// RUN: ttlang-opt --verify-diagnostics --split-input-file %s

// -----

// raw_element_read outside a kernel thread function.
func.func @read_no_kernel_thread(
    %block: tensor<1x1x!ttcore.tile<32x32, f32>>) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{must be inside a function with 'ttl.kernel_thread' attribute}}
  %val = ttl.raw_element_read %block[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
  func.return
}

// -----

// raw_element_read in a compute thread (only noc allowed).
func.func @read_compute_thread(
    %block: tensor<1x1x!ttcore.tile<32x32, f32>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  // expected-error @below {{is only allowed in data movement (noc) threads}}
  %val = ttl.raw_element_read %block[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
  func.return
}

// -----

// raw_element_read on a tensor not backed by a circular buffer.
func.func @read_no_cb(
    %block: tensor<1x1x!ttcore.tile<32x32, f32>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  // expected-error @below {{block must be a tensor view acquired from ttl.cb_wait or ttl.cb_reserve}}
  %val = ttl.raw_element_read %block[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
  func.return
}

// -----

// raw_element_read with wrong number of coordinates (1 coord for rank-2 block).
func.func @read_coord_count_mismatch()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  // expected-error @below {{coordinate count (1) must match block tensor rank (2)}}
  %val = ttl.raw_element_read %block[%c0] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
  func.return
}

// -----

// raw_element_read with scalar type mismatch (bf16 result from f32 block).
func.func @read_dtype_mismatch()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  // expected-error @below {{scalar type ('bf16') must match block element dtype ('f32')}}
  %val = ttl.raw_element_read %block[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>> -> bf16
  func.return
}

// -----

// raw_element_write outside a kernel thread function.
func.func @write_no_kernel_thread(
    %block: tensor<1x1x!ttcore.tile<32x32, f32>>, %val: f32) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{must be inside a function with 'ttl.kernel_thread' attribute}}
  ttl.raw_element_write %block[%c0, %c0], %val : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
  func.return
}

// -----

// raw_element_write in a compute thread (only noc allowed).
func.func @write_compute_thread(
    %block: tensor<1x1x!ttcore.tile<32x32, f32>>, %val: f32)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  // expected-error @below {{is only allowed in data movement (noc) threads}}
  ttl.raw_element_write %block[%c0, %c0], %val : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
  func.return
}

// -----

// raw_element_write on a tensor not backed by a circular buffer.
func.func @write_no_cb(
    %block: tensor<1x1x!ttcore.tile<32x32, f32>>, %val: f32)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  // expected-error @below {{block must be a tensor view acquired from ttl.cb_wait or ttl.cb_reserve}}
  ttl.raw_element_write %block[%c0, %c0], %val : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
  func.return
}

// -----

// raw_element_write with wrong number of coordinates.
func.func @write_coord_count_mismatch(%val: f32)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %block = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  // expected-error @below {{coordinate count (3) must match block tensor rank (2)}}
  ttl.raw_element_write %block[%c0, %c0, %c0], %val : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
  func.return
}

// -----

// raw_element_write with scalar type mismatch (f32 value into bf16 block).
func.func @write_dtype_mismatch(%val: f32)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %block = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  // expected-error @below {{scalar type ('f32') must match block element dtype ('bf16')}}
  ttl.raw_element_write %block[%c0, %c0], %val : tensor<1x1x!ttcore.tile<32x32, bf16>>, f32
  func.return
}

// -----

// raw_element_read on a tensor from ttl.attach_cb (not a direct CB acquire).
func.func @read_attach_cb_not_allowed(
    %t: tensor<1x1x!ttcore.tile<32x32, f32>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %block = ttl.attach_cb %t, %cb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  // expected-error @below {{block must be a tensor view acquired from ttl.cb_wait or ttl.cb_reserve}}
  %val = ttl.raw_element_read %block[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
  func.return
}

// -----

// raw_element_write on a tensor from ttl.attach_cb (not a direct CB acquire).
func.func @write_attach_cb_not_allowed(
    %t: tensor<1x1x!ttcore.tile<32x32, f32>>, %val: f32)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %block = ttl.attach_cb %t, %cb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  // expected-error @below {{block must be a tensor view acquired from ttl.cb_wait or ttl.cb_reserve}}
  ttl.raw_element_write %block[%c0, %c0], %val : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
  func.return
}
