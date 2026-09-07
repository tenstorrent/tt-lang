// Tensor reinterpretation casts must not act as unchecked DFB shape views.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(func.func(ttl-annotate-cb-associations))'

// Equal tile count does not make an arbitrary tensor cast a DFB shape view.
func.func @shape_reinterpretation(%tensor: tensor<4x!ttcore.tile<32x32, bf16>>, %input: !ttcore.tile<32x32, bf16>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>
  %attached = ttl.attach_cb %tensor, %cb
      : (tensor<4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<4x!ttcore.tile<32x32, bf16>>
  %view = builtin.unrealized_conversion_cast %attached
      : tensor<4x!ttcore.tile<32x32, bf16>> to tensor<2x2x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %tile = tensor.extract %view[%c0, %c0] : tensor<2x2x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{output does not have an attached circular buffer}}
  %result = ttl.tile_bcast %input, %tile 3 : i32 into dst[%c0]
      : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
  return
}

// -----

// A tensor cast cannot retain DFB identity while changing its encoding.
func.func @encoding_reinterpretation(%tensor: tensor<4x!ttcore.tile<32x32, bf16>, "source">, %input: !ttcore.tile<32x32, bf16>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>
  %attached = ttl.attach_cb %tensor, %cb
      : (tensor<4x!ttcore.tile<32x32, bf16>, "source">, !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<4x!ttcore.tile<32x32, bf16>, "source">
  %view = builtin.unrealized_conversion_cast %attached
      : tensor<4x!ttcore.tile<32x32, bf16>, "source"> to tensor<4x!ttcore.tile<32x32, bf16>, "result">
  %c0 = arith.constant 0 : index
  %tile = tensor.extract %view[%c0] : tensor<4x!ttcore.tile<32x32, bf16>, "result">
  // expected-error @below {{output does not have an attached circular buffer}}
  %result = ttl.tile_bcast %input, %tile 3 : i32 into dst[%c0]
      : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
  return
}

// -----

// A tensor cast cannot retain DFB identity while changing its tile data type.
func.func @dtype_reinterpretation(%tensor: tensor<4x!ttcore.tile<32x32, bf16>>, %input: !ttcore.tile<32x32, f32>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>
  %attached = ttl.attach_cb %tensor, %cb
      : (tensor<4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<4x!ttcore.tile<32x32, bf16>>
  %view = builtin.unrealized_conversion_cast %attached
      : tensor<4x!ttcore.tile<32x32, bf16>> to tensor<4x!ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %tile = tensor.extract %view[%c0] : tensor<4x!ttcore.tile<32x32, f32>>
  // expected-error @below {{output does not have an attached circular buffer}}
  %result = ttl.tile_bcast %input, %tile 3 : i32 into dst[%c0]
      : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
  return
}

// -----

// A one-input, two-result cast is not a one-to-one DFB conversion bridge.
func.func @multi_result_reinterpretation(%tensor: tensor<4x!ttcore.tile<32x32, bf16>>, %input: !ttcore.tile<32x32, bf16>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>
  %attached = ttl.attach_cb %tensor, %cb
      : (tensor<4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<4x!ttcore.tile<32x32, bf16>>
  %view, %other = builtin.unrealized_conversion_cast %attached
      : tensor<4x!ttcore.tile<32x32, bf16>> to tensor<4x!ttcore.tile<32x32, bf16>>, tensor<4x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %tile = tensor.extract %view[%c0] : tensor<4x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{output does not have an attached circular buffer}}
  %result = ttl.tile_bcast %input, %tile 3 : i32 into dst[%c0]
      : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
  return
}
