// Unsupported block shape views fail conversion with a precise diagnostic
// instead of being erased as DFB aliases.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --convert-ttl-to-ttkernel

// Equal tile count does not permit splitting a non-singleton dimension.
func.func @general_expand() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>
  %input = ttl.cb_wait %dfb : <[4], !ttcore.tile<32x32, bf16>, 2>
      -> tensor<4x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{block shape views require static shapes, identical element types and encodings, and singleton-dimension insertion or removal}}
  %view = tensor.expand_shape %input [[0, 1]] output_shape [2, 2]
      : tensor<4x!ttcore.tile<32x32, bf16>>
        into tensor<2x2x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// Equal tile count does not permit merging two non-singleton dimensions.
func.func @general_collapse() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %input = ttl.cb_wait %dfb : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
      -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{block shape views require static shapes, identical element types and encodings, and singleton-dimension insertion or removal}}
  %view = tensor.collapse_shape %input [[0, 1]]
      : tensor<2x2x!ttcore.tile<32x32, bf16>>
        into tensor<4x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// Dynamic singleton expansion lacks the required static-shape proof.
func.func @dynamic_expand(%input: tensor<?x!ttcore.tile<32x32, bf16>>, %size: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  // expected-error @below {{block shape views require static shapes, identical element types and encodings, and singleton-dimension insertion or removal}}
  %view = tensor.expand_shape %input [[0, 1]] output_shape [1, %size]
      : tensor<?x!ttcore.tile<32x32, bf16>>
        into tensor<1x?x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// Encoding changes cannot be lowered as aliases of the same DFB.
func.func @encoding_change(%input: tensor<4x!ttcore.tile<32x32, bf16>, "source">)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  // expected-error @below {{block shape views require static shapes, identical element types and encodings, and singleton-dimension insertion or removal}}
  %view = tensor.expand_shape %input [[0, 1]] output_shape [1, 4]
      : tensor<4x!ttcore.tile<32x32, bf16>, "source">
        into tensor<1x4x!ttcore.tile<32x32, bf16>, "result">
  return
}
