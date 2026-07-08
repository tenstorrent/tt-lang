// Negative tests for raw element operations in convert-ttl-to-ttkernel.
// RUN: ttlang-opt --convert-ttl-to-ttkernel --verify-diagnostics --split-input-file %s

// -----

// raw_element_write with a value that does not trace to a CB (missing bind_cb).
module {
  func.func @write_no_cb(%block: tensor<1x1x!ttcore.tile<32x32, f32>>, %val_int: i32)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %val = builtin.unrealized_conversion_cast %val_int : i32 to f32
    // expected-error @below {{block must be a tensor view acquired from ttl.cb_reserve}}
    ttl.raw_element_write %block[%c0, %c0], %val : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
    func.return
  }
}
