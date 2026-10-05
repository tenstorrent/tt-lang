// General reshapes, dynamic shapes, and encoding changes do not preserve CB
// association, even when their source tensor is backed by a DFB.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(func.func(ttl-annotate-cb-associations))'

// Splitting a non-singleton dimension preserves tile count, not block identity.
func.func @general_expand(%input: !ttcore.tile<32x32, bf16>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>
  %waited = ttl.cb_wait %cb : <[4], !ttcore.tile<32x32, bf16>, 2>
      -> tensor<4x!ttcore.tile<32x32, bf16>>
  %view = tensor.expand_shape %waited [[0, 1]] output_shape [2, 2]
      : tensor<4x!ttcore.tile<32x32, bf16>>
        into tensor<2x2x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %tile = tensor.extract %view[%c0, %c0] : tensor<2x2x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{output does not have an attached circular buffer}}
  %result = ttl.tile_bcast %input, %tile 3 : i32 into dst[%c0]
      : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
        -> !ttcore.tile<32x32, bf16>
  return
}

// -----

// Merging non-singleton dimensions is not a squeeze.
func.func @general_collapse(%input: !ttcore.tile<32x32, bf16>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %waited = ttl.cb_wait %cb : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
      -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %view = tensor.collapse_shape %waited [[0, 1]]
      : tensor<2x2x!ttcore.tile<32x32, bf16>>
        into tensor<4x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %tile = tensor.extract %view[%c0] : tensor<4x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{output does not have an attached circular buffer}}
  %result = ttl.tile_bcast %input, %tile 3 : i32 into dst[%c0]
      : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
        -> !ttcore.tile<32x32, bf16>
  return
}

// -----

// A dynamic extent remains unsupported even when the inserted extent is one.
func.func @dynamic_expand(%input: !ttcore.tile<32x32, bf16>,
    %tensor: tensor<?x!ttcore.tile<32x32, bf16>>, %size: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>
  %attached = ttl.attach_cb %tensor, %cb
      : (tensor<?x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<?x!ttcore.tile<32x32, bf16>>
  %view = tensor.expand_shape %attached [[0, 1]] output_shape [1, %size]
      : tensor<?x!ttcore.tile<32x32, bf16>>
        into tensor<1x?x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %tile = tensor.extract %view[%c0, %c0] : tensor<1x?x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{output does not have an attached circular buffer}}
  %result = ttl.tile_bcast %input, %tile 3 : i32 into dst[%c0]
      : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
        -> !ttcore.tile<32x32, bf16>
  return
}

// -----

// A dynamic collapse does not establish the required static-view proof.
func.func @dynamic_collapse(%input: !ttcore.tile<32x32, bf16>,
    %tensor: tensor<1x?x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
  %attached = ttl.attach_cb %tensor, %cb
      : (tensor<1x?x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x?x!ttcore.tile<32x32, bf16>>
  %view = tensor.collapse_shape %attached [[0, 1]]
      : tensor<1x?x!ttcore.tile<32x32, bf16>>
        into tensor<?x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %tile = tensor.extract %view[%c0] : tensor<?x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{output does not have an attached circular buffer}}
  %result = ttl.tile_bcast %input, %tile 3 : i32 into dst[%c0]
      : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
        -> !ttcore.tile<32x32, bf16>
  return
}

// -----

// An encoding change must not inherit the source DFB's association.
func.func @encoding_change(%input: !ttcore.tile<32x32, bf16>,
    %tensor: tensor<4x!ttcore.tile<32x32, bf16>, "source-encoding">)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>
  %attached = ttl.attach_cb %tensor, %cb
      : (tensor<4x!ttcore.tile<32x32, bf16>, "source-encoding">,
         !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<4x!ttcore.tile<32x32, bf16>, "source-encoding">
  %view = tensor.expand_shape %attached [[0, 1]] output_shape [1, 4]
      : tensor<4x!ttcore.tile<32x32, bf16>, "source-encoding">
        into tensor<1x4x!ttcore.tile<32x32, bf16>, "result-encoding">
  %c0 = arith.constant 0 : index
  %tile = tensor.extract %view[%c0, %c0]
      : tensor<1x4x!ttcore.tile<32x32, bf16>, "result-encoding">
  // expected-error @below {{output does not have an attached circular buffer}}
  %result = ttl.tile_bcast %input, %tile 3 : i32 into dst[%c0]
      : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
        -> !ttcore.tile<32x32, bf16>
  return
}
