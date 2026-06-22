// Negative tests for raw_element_write lowering: materializeIntBits fails
// when the value source is not from raw_element_read, a float constant,
// or arith.truncf.
// RUN: ttlang-opt --convert-ttl-to-ttkernel --verify-diagnostics --split-input-file %s

// -----

// raw_element_write with a block argument (unsupported value source).
// materializeIntBits cannot extract integer bits from a bare function argument.
// expected-error @below {{failed to legalize operation 'ttl.raw_element_write' that was explicitly marked illegal}}
module {
  func.func @write_unsupported_source(%val: f32)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %block = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %c0 = arith.constant 0 : index
    ttl.raw_element_write %block[%c0, %c0], %val : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
    func.return
  }
}
