// Shape views above nonzero subblock slices retain the original DFB offsets
// when lowering both tile loads and tile stores.
// RUN: ttlang-opt %s --split-input-file --convert-ttl-to-ttkernel --canonicalize --cse | FileCheck %s --implicit-check-not=tensor.expand_shape --implicit-check-not=tensor.collapse_shape

// Local element (1,1) of a [2,2] slice at (0,2) in [2,4] is tile 7;
// the same element in a [2,2] slice at (0,3) in [2,5] is tile 9.
// CHECK-LABEL: func.func @nonzero_slice_offsets_through_views
// CHECK-DAG: %[[SEVEN:.*]] = arith.constant 7 : index
// CHECK-DAG: %[[NINE:.*]] = arith.constant 9 : index
// CHECK: %[[INPUT:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[OUTPUT:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.copy_tile(%[[INPUT]], %[[SEVEN]],
// CHECK-NEXT: ttkernel.pack_tile(%{{.*}}, %[[OUTPUT]], %[[NINE]], true)
func.func @nonzero_slice_offsets_through_views()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 2, 4], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[2, 5], !ttcore.tile<32x32, bf16>, 2>
  %input = ttl.cb_wait %input_dfb
      : <[1, 2, 4], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x2x4x!ttcore.tile<32x32, bf16>>
  %slice = tensor.extract_slice %input [0, 0, 2] [1, 2, 2] [1, 1, 1]
      : tensor<1x2x4x!ttcore.tile<32x32, bf16>>
        to tensor<1x2x2x!ttcore.tile<32x32, bf16>>
  %squeezed = tensor.collapse_shape %slice [[0, 1], [2]]
      : tensor<1x2x2x!ttcore.tile<32x32, bf16>>
        into tensor<2x2x!ttcore.tile<32x32, bf16>>
  %expanded = tensor.expand_shape %squeezed [[0], [1, 2]] output_shape [2, 2, 1]
      : tensor<2x2x!ttcore.tile<32x32, bf16>>
        into tensor<2x2x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[2, 5], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<2x5x!ttcore.tile<32x32, bf16>>
  %output_slice = tensor.extract_slice %output [0, 3] [2, 2] [1, 1]
      : tensor<2x5x!ttcore.tile<32x32, bf16>>
        to tensor<2x2x!ttcore.tile<32x32, bf16>>
  %output_view = tensor.expand_shape %output_slice [[0, 1], [2]] output_shape [1, 2, 2]
      : tensor<2x2x!ttcore.tile<32x32, bf16>>
        into tensor<1x2x2x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tile = tensor.extract %expanded[%c1, %c1, %c0]
      : tensor<2x2x1x!ttcore.tile<32x32, bf16>>
  %dst, %copied = ttl.copy_tile %tile[%c1, %c1, %c0] into dst[%c0]
      : !ttcore.tile<32x32, bf16> -> !ttl.dst, !ttcore.tile<32x32, bf16>
  ttl.tile_store %copied, %output_view[%c0, %c1, %c1] from dst[%c0]
      : !ttcore.tile<32x32, bf16>, tensor<1x2x2x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// Two nonzero slices separated by a checked shape view compose: source (1,1)
// becomes full-source (0,3,3), tile 21; destination (1,1) becomes (2,4), tile 18.
// CHECK-LABEL: func.func @nested_nonzero_slice_offsets_through_views
// CHECK-DAG: %[[TWENTY_ONE:.*]] = arith.constant 21 : index
// CHECK-DAG: %[[EIGHTEEN:.*]] = arith.constant 18 : index
// CHECK: %[[INPUT:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[OUTPUT:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.copy_tile(%[[INPUT]], %[[TWENTY_ONE]],
// CHECK-NEXT: ttkernel.pack_tile(%{{.*}}, %[[OUTPUT]], %[[EIGHTEEN]], true)
func.func @nested_nonzero_slice_offsets_through_views()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 4, 6], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[3, 7], !ttcore.tile<32x32, bf16>, 2>
  %input = ttl.cb_wait %input_dfb
      : <[1, 4, 6], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x4x6x!ttcore.tile<32x32, bf16>>
  %slice = tensor.extract_slice %input [0, 1, 1] [1, 3, 4] [1, 1, 1]
      : tensor<1x4x6x!ttcore.tile<32x32, bf16>>
        to tensor<1x3x4x!ttcore.tile<32x32, bf16>>
  %squeezed = tensor.collapse_shape %slice [[0, 1], [2]]
      : tensor<1x3x4x!ttcore.tile<32x32, bf16>>
        into tensor<3x4x!ttcore.tile<32x32, bf16>>
  %inner_slice = tensor.extract_slice %squeezed [1, 1] [2, 2] [1, 1]
      : tensor<3x4x!ttcore.tile<32x32, bf16>>
        to tensor<2x2x!ttcore.tile<32x32, bf16>>
  %expanded = tensor.expand_shape %inner_slice [[0], [1, 2]] output_shape [2, 1, 2]
      : tensor<2x2x!ttcore.tile<32x32, bf16>>
        into tensor<2x1x2x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[3, 7], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<3x7x!ttcore.tile<32x32, bf16>>
  %output_slice = tensor.extract_slice %output [1, 1] [2, 5] [1, 1]
      : tensor<3x7x!ttcore.tile<32x32, bf16>>
        to tensor<2x5x!ttcore.tile<32x32, bf16>>
  %output_expanded = tensor.expand_shape %output_slice [[0], [1, 2]] output_shape [2, 1, 5]
      : tensor<2x5x!ttcore.tile<32x32, bf16>>
        into tensor<2x1x5x!ttcore.tile<32x32, bf16>>
  %output_inner_slice = tensor.extract_slice %output_expanded [0, 0, 2] [2, 1, 2] [1, 1, 1]
      : tensor<2x1x5x!ttcore.tile<32x32, bf16>>
        to tensor<2x1x2x!ttcore.tile<32x32, bf16>>
  %output_view = tensor.collapse_shape %output_inner_slice [[0], [1, 2]]
      : tensor<2x1x2x!ttcore.tile<32x32, bf16>>
        into tensor<2x2x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tile = tensor.extract %expanded[%c1, %c0, %c1]
      : tensor<2x1x2x!ttcore.tile<32x32, bf16>>
  %dst, %copied = ttl.copy_tile %tile[%c1, %c0, %c1] into dst[%c0]
      : !ttcore.tile<32x32, bf16> -> !ttl.dst, !ttcore.tile<32x32, bf16>
  ttl.tile_store %copied, %output_view[%c1, %c1] from dst[%c0]
      : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
  return
}
