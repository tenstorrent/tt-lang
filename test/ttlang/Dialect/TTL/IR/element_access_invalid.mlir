// Summary: verify element_read/element_write verifiers reject invalid inputs.
// Tests W2 (static bounds check) and W3 (single-tile block requirement).
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// -----
// W2: element_read with row index out of bounds (row=32 on a 32x32 tile).

func.func @read_row_out_of_bounds()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  // expected-error @below {{'ttl.element_read' op row index 32 is out of range [0, 31] for tile of size 32x32}}
  %val = ttl.element_read %wait[%c32, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> i32
  func.return
}

// -----
// W2: element_read with col index out of bounds (col=32 on a 32x32 tile).

func.func @read_col_out_of_bounds()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  // expected-error @below {{'ttl.element_read' op col index 32 is out of range [0, 31] for tile of size 32x32}}
  %val = ttl.element_read %wait[%c0, %c32] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> i32
  func.return
}

// -----
// W2: element_write with row index out of bounds (row=33).

func.func @write_row_out_of_bounds()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %reserve = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c33 = arith.constant 33 : index
  %c0 = arith.constant 0 : index
  %val = arith.constant 42 : i32
  // expected-error @below {{'ttl.element_write' op row index 33 is out of range [0, 31] for tile of size 32x32}}
  ttl.element_write %reserve[%c33, %c0], %val : tensor<1x1x!ttcore.tile<32x32, bf16>>, i32
  func.return
}

// -----
// W2: element_write with negative row index (negative constants are invalid).

func.func @write_negative_row(%row: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %reserve = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %cn1 = arith.constant -1 : index
  %c0 = arith.constant 0 : index
  %val = arith.constant 42 : i32
  // expected-error @below {{'ttl.element_write' op row index -1 is out of range [0, 31] for tile of size 32x32}}
  ttl.element_write %reserve[%cn1, %c0], %val : tensor<1x1x!ttcore.tile<32x32, bf16>>, i32
  func.return
}

// -----
// W3: element_read on a multi-tile block (tensor<2x1x...>).

func.func @read_multi_tile_block()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %cb0 : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  // expected-error @below {{'ttl.element_read' op element access requires a single-tile block (all tensor dimensions must be 1), but dimension 0 is 2}}
  %val = ttl.element_read %wait[%c0, %c0] : tensor<2x1x!ttcore.tile<32x32, bf16>> -> i32
  func.return
}

// -----
// W3: element_write on a multi-tile block (tensor<1x2x...>).

func.func @write_multi_tile_block()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
  %reserve = ttl.cb_reserve %cb0 : <[1, 2], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %val = arith.constant 42 : i32
  // expected-error @below {{'ttl.element_write' op element access requires a single-tile block (all tensor dimensions must be 1), but dimension 1 is 2}}
  ttl.element_write %reserve[%c0, %c0], %val : tensor<1x2x!ttcore.tile<32x32, bf16>>, i32
  func.return
}

// -----
// W2: element_read with f32 tile and col=32 (still 32x32 tile bounds).

func.func @read_f32_col_out_of_bounds()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %wait = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  // expected-error @below {{'ttl.element_read' op col index 32 is out of range [0, 31] for tile of size 32x32}}
  %val = ttl.element_read %wait[%c0, %c32] : tensor<1x1x!ttcore.tile<32x32, f32>> -> i32
  func.return
}
