// Tensor reinterpretation casts must not act as unchecked DFB shape views.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --convert-ttl-to-ttkernel

// Equal tile count does not make an arbitrary tensor cast a DFB shape view.
func.func @shape_reinterpretation(%tensor: tensor<4x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>
  %attached = ttl.attach_cb %tensor, %cb
      : (tensor<4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<4x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{DFB views cannot use tensor reinterpretation casts; use checked singleton-dimension expand_shape or collapse_shape operations}}
  %view = builtin.unrealized_conversion_cast %attached
      : tensor<4x!ttcore.tile<32x32, bf16>> to tensor<2x2x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// A tensor cast cannot retain DFB identity while changing its encoding.
func.func @encoding_reinterpretation(%tensor: tensor<4x!ttcore.tile<32x32, bf16>, "source">)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>
  %attached = ttl.attach_cb %tensor, %cb
      : (tensor<4x!ttcore.tile<32x32, bf16>, "source">, !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<4x!ttcore.tile<32x32, bf16>, "source">
  // expected-error @below {{DFB views cannot use tensor reinterpretation casts; use checked singleton-dimension expand_shape or collapse_shape operations}}
  %view = builtin.unrealized_conversion_cast %attached
      : tensor<4x!ttcore.tile<32x32, bf16>, "source"> to tensor<4x!ttcore.tile<32x32, bf16>, "result">
  return
}

// -----

// A tensor cast cannot retain DFB identity while changing its tile data type.
func.func @dtype_reinterpretation(%tensor: tensor<4x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>
  %attached = ttl.attach_cb %tensor, %cb
      : (tensor<4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<4x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{DFB views cannot use tensor reinterpretation casts; use checked singleton-dimension expand_shape or collapse_shape operations}}
  %view = builtin.unrealized_conversion_cast %attached
      : tensor<4x!ttcore.tile<32x32, bf16>> to tensor<4x!ttcore.tile<32x32, f32>>
  return
}

// -----

// A one-input, two-result cast is not a one-to-one DFB conversion bridge.
func.func @multi_result_reinterpretation(%tensor: tensor<4x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>
  %attached = ttl.attach_cb %tensor, %cb
      : (tensor<4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<4x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{DFB views cannot use tensor reinterpretation casts; use checked singleton-dimension expand_shape or collapse_shape operations}}
  %view, %other = builtin.unrealized_conversion_cast %attached
      : tensor<4x!ttcore.tile<32x32, bf16>> to tensor<4x!ttcore.tile<32x32, bf16>>, tensor<4x!ttcore.tile<32x32, bf16>>
  return
}
