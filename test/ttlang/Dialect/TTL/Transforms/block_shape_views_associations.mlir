// Singleton views preserve CB association through extracts, including encoded
// tensors. Equal encodings do not change the underlying DFB association.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-annotate-cb-associations))' | FileCheck %s

// A view chain through both reshape operations retains the output CB index.
// CHECK-LABEL: func.func @encoded_singleton_views
// CHECK: %[[CB:.*]] = ttl.bind_cb{cb_index = 7
// CHECK: %[[ATTACHED:.*]] = ttl.attach_cb %{{.*}}, %[[CB]]
// CHECK-NEXT: %[[EXPANDED:.*]] = tensor.expand_shape %[[ATTACHED]]
// CHECK-NEXT: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[EXPANDED]]
// CHECK: %[[TILE:.*]] = tensor.extract %[[COLLAPSED]]
// CHECK: ttl.tile_bcast %{{.*}}, %[[TILE]]
// CHECK-SAME: ttl.bcast_output_cb_index = 7 : index
func.func @encoded_singleton_views(
    %input: !ttcore.tile<32x32, bf16>,
    %tensor: tensor<2x3x!ttcore.tile<32x32, bf16>, "same-encoding">)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 7, block_count = 2}
      : !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 2>
  %attached = ttl.attach_cb %tensor, %cb
      : (tensor<2x3x!ttcore.tile<32x32, bf16>, "same-encoding">,
         !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<2x3x!ttcore.tile<32x32, bf16>, "same-encoding">
  %expanded = tensor.expand_shape %attached [[0, 1], [2, 3]] output_shape [1, 2, 3, 1]
      : tensor<2x3x!ttcore.tile<32x32, bf16>, "same-encoding">
        into tensor<1x2x3x1x!ttcore.tile<32x32, bf16>, "same-encoding">
  %collapsed = tensor.collapse_shape %expanded [[0, 1], [2, 3]]
      : tensor<1x2x3x1x!ttcore.tile<32x32, bf16>, "same-encoding">
        into tensor<2x3x!ttcore.tile<32x32, bf16>, "same-encoding">
  %c0 = arith.constant 0 : index
  %tile = tensor.extract %collapsed[%c0, %c0]
      : tensor<2x3x!ttcore.tile<32x32, bf16>, "same-encoding">
  %result = ttl.tile_bcast %input, %tile 3 : i32 into dst[%c0]
      : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
        -> !ttcore.tile<32x32, bf16>
  return
}

// -----

// Removing every singleton dimension and restoring them preserves the DFB.
// CHECK-LABEL: func.func @rank_zero_roundtrip
// CHECK: %[[CB:.*]] = ttl.bind_cb{cb_index = 3
// CHECK: %[[WAIT:.*]] = ttl.cb_wait %[[CB]]
// CHECK-NEXT: %[[SCALAR:.*]] = tensor.collapse_shape %[[WAIT]] []
// CHECK-NEXT: %[[EXPANDED:.*]] = tensor.expand_shape %[[SCALAR]] []
// CHECK: %[[TILE:.*]] = tensor.extract %[[EXPANDED]]
// CHECK: ttl.tile_bcast %{{.*}}, %[[TILE]]
// CHECK-SAME: ttl.bcast_output_cb_index = 3 : index
func.func @rank_zero_roundtrip(%input: !ttcore.tile<32x32, bf16>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 3, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %waited = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scalar = tensor.collapse_shape %waited []
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        into tensor<!ttcore.tile<32x32, bf16>>
  %expanded = tensor.expand_shape %scalar [] output_shape [1, 1]
      : tensor<!ttcore.tile<32x32, bf16>>
        into tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %tile = tensor.extract %expanded[%c0, %c0]
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.tile_bcast %input, %tile 3 : i32 into dst[%c0]
      : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
        -> !ttcore.tile<32x32, bf16>
  return
}
