// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// Verify that tensor copies use DFB subviews from the correct acquisition.

#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                      buffer = dram, grid = [1, 1], memory = interleaved>

module {
  func.func @read_requires_reserved_subview(
      %input: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 1>
    %waited = ttl.cb_wait %dfb
        : <[1, 2], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x2x!ttcore.tile<32x32, f32>>
    %block = ttl.attach_cb %waited, %dfb
        : (tensor<1x2x!ttcore.tile<32x32, f32>>,
           !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, f32>>
    %view = tensor.extract_slice %block[0, 1] [1, 1] [1, 1]
        : tensor<1x2x!ttcore.tile<32x32, f32>>
        to tensor<1x1x!ttcore.tile<32x32, f32>>
    // expected-error @below {{tensor copy destination DFB view must come from ttl.cb_reserve}}
    %transfer = ttl.copy %input, %view
        : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle<read>
    ttl.wait %transfer : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                      buffer = dram, grid = [1, 1], memory = interleaved>

module {
  func.func @write_requires_waited_subview(
      %output: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 1>
    %reserved = ttl.cb_reserve %dfb
        : <[1, 2], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x2x!ttcore.tile<32x32, f32>>
    %block = ttl.attach_cb %reserved, %dfb
        : (tensor<1x2x!ttcore.tile<32x32, f32>>,
           !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, f32>>
    %view = tensor.extract_slice %block[0, 1] [1, 1] [1, 1]
        : tensor<1x2x!ttcore.tile<32x32, f32>>
        to tensor<1x1x!ttcore.tile<32x32, f32>>
    // expected-error @below {{tensor copy source DFB view must come from ttl.cb_wait}}
    %transfer = ttl.copy %view, %output
        : (tensor<1x1x!ttcore.tile<32x32, f32>>,
           tensor<1x1x!ttcore.tile<32x32, f32>, #layout>)
        -> !ttl.transfer_handle<write>
    ttl.wait %transfer : !ttl.transfer_handle<write>
    func.return
  }
}

// -----

#layout = #ttl.layout<shape = [1, 2], element_type = !ttcore.tile<32x32, f32>,
                      buffer = dram, grid = [1, 1], memory = interleaved>

module {
  func.func @tensor_shape_must_match_subview(
      %output: tensor<1x2x!ttcore.tile<32x32, f32>, #layout>)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 1>
    %waited = ttl.cb_wait %dfb
        : <[1, 2], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x2x!ttcore.tile<32x32, f32>>
    %block = ttl.attach_cb %waited, %dfb
        : (tensor<1x2x!ttcore.tile<32x32, f32>>,
           !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, f32>>
    %view = tensor.extract_slice %block[0, 1] [1, 1] [1, 1]
        : tensor<1x2x!ttcore.tile<32x32, f32>>
        to tensor<1x1x!ttcore.tile<32x32, f32>>
    // expected-error @below {{tensor shape 1, 2 must match DFB view shape 1, 1}}
    %transfer = ttl.copy %view, %output
        : (tensor<1x1x!ttcore.tile<32x32, f32>>,
           tensor<1x2x!ttcore.tile<32x32, f32>, #layout>)
        -> !ttl.transfer_handle<write>
    ttl.wait %transfer : !ttl.transfer_handle<write>
    func.return
  }
}
