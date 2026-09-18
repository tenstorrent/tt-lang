// RUN: ttlang-opt --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute))' --verify-diagnostics --split-input-file %s

// Verify that compute creation requires explicit physical tile types instead
// of assigning default tile dimensions to scalar-element tensors.

// A tensor operation cannot produce a scalar-element tensor because its
// physical tile dimensions are unavailable.
module {
  func.func @scalar_result() {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], bf16, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], bf16, 2>
    %wait = ttl.cb_wait %input_dfb
        : <[1, 1], bf16, 2> -> tensor<1x1xbf16>
    %input = ttl.attach_cb %wait, %input_dfb
        : (tensor<1x1xbf16>, !ttl.cb<[1, 1], bf16, 2>)
          -> tensor<1x1xbf16>
    %output = ttl.cb_reserve %output_dfb
        : <[1, 1], bf16, 2> -> tensor<1x1xbf16>
    // expected-error @below {{'ttl.exp' op compute result must be a tensor of ttcore.tile elements, got tensor<1x1xbf16>}}
    %result = ttl.exp %input : tensor<1x1xbf16> -> tensor<1x1xbf16>
    ttl.store %result, %output : tensor<1x1xbf16>, tensor<1x1xbf16>
    func.return
  }
}

// -----

// A tensor operation cannot consume a scalar-element tensor when its result
// has an explicit physical tile type.
module {
  func.func @scalar_input() {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], bf16, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %wait = ttl.cb_wait %input_dfb
        : <[1, 1], bf16, 2> -> tensor<1x1xbf16>
    %input = ttl.attach_cb %wait, %input_dfb
        : (tensor<1x1xbf16>, !ttl.cb<[1, 1], bf16, 2>)
          -> tensor<1x1xbf16>
    %output = ttl.cb_reserve %output_dfb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    // expected-error @below {{'ttl.typecast' op compute input 0 must be a tensor of ttcore.tile elements, got tensor<1x1xbf16>}}
    %result = ttl.typecast %input
        : (tensor<1x1xbf16>)
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.store %result, %output
        : tensor<1x1x!ttcore.tile<32x32, f32>>,
          tensor<1x1x!ttcore.tile<32x32, f32>>
    func.return
  }
}

// -----

// A passthrough store cannot create a compute operation without an explicit
// physical tile type.
module {
  func.func @scalar_passthrough() {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], bf16, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], bf16, 2>
    %wait = ttl.cb_wait %input_dfb
        : <[1, 1], bf16, 2> -> tensor<1x1xbf16>
    %input = ttl.attach_cb %wait, %input_dfb
        : (tensor<1x1xbf16>, !ttl.cb<[1, 1], bf16, 2>)
          -> tensor<1x1xbf16>
    %output = ttl.cb_reserve %output_dfb
        : <[1, 1], bf16, 2> -> tensor<1x1xbf16>
    // expected-error @below {{'ttl.store' op cannot lower tensor store to ttl.compute: passthrough store input must be a tensor of ttcore.tile elements, got tensor<1x1xbf16>}}
    ttl.store %input, %output : tensor<1x1xbf16>, tensor<1x1xbf16>
    func.return
  }
}
