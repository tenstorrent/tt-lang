// Static singleton-dimension views preserve DFB inputs and releases through
// compute creation and the complete TTKernel pipeline.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-insert-cb-sync,convert-ttl-to-compute,ttl-annotate-cb-associations))' | FileCheck %s --check-prefix=COMPUTE
// RUN: ttlang-opt %s --split-input-file --ttl-to-ttkernel-pipeline | FileCheck %s --check-prefix=LOWER --implicit-check-not=tensor.expand_shape --implicit-check-not=tensor.collapse_shape --implicit-check-not=builtin.unrealized_conversion_cast

// Removing several singleton dimensions and reinserting one keeps the input
// attached to the original BF16 DFB until its compute use completes.
// COMPUTE-LABEL: func.func @bf16_singleton_chain
// COMPUTE: %[[INPUT_DFB:.*]] = ttl.bind_cb{cb_index = 0
// COMPUTE: %[[OUTPUT_DFB:.*]] = ttl.bind_cb{cb_index = 1
// COMPUTE: %[[WAIT:.*]] = ttl.cb_wait %[[INPUT_DFB]]
// COMPUTE: %[[SQUEEZED:.*]] = tensor.collapse_shape %[[WAIT]]
// COMPUTE-NEXT: %[[EXPANDED:.*]] = tensor.expand_shape %[[SQUEEZED]]
// COMPUTE: ttl.compute ins(%[[EXPANDED]]
// COMPUTE-SAME: ttl.cb_index.0 = 0 : i64
// COMPUTE: ttl.tile_exp
// COMPUTE: ttl.tile_store
// COMPUTE: ttl.yield
// COMPUTE: ttl.cb_pop %[[INPUT_DFB]]
// COMPUTE: ttl.cb_push %[[OUTPUT_DFB]]
// COMPUTE-NEXT: return
// LOWER-LABEL: func.func @bf16_singleton_chain
// LOWER: %[[INPUT:.*]] = ttkernel.get_compile_time_arg_val(0)
// LOWER: %[[OUTPUT:.*]] = ttkernel.get_compile_time_arg_val(1)
// LOWER: ttkernel.cb_wait_front(%[[INPUT]],
// LOWER: ttkernel.copy_tile(%[[INPUT]],
// LOWER: ttkernel.exp_tile
// LOWER: ttkernel.pack_tile_block({{.*}}%[[OUTPUT]]
// LOWER: ttkernel.cb_pop_front(%[[INPUT]],
// LOWER: ttkernel.cb_push_back(%[[OUTPUT]],
// COMPUTE-LABEL: func.func @bf16_io
// LOWER-LABEL: func.func @bf16_io
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
func.func @bf16_singleton_chain()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 2, 1, 3], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
      : !ttl.cb<[2, 1, 3], !ttcore.tile<32x32, bf16>, 2>
  %input = ttl.cb_wait %input_dfb
      : <[1, 2, 1, 3], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x2x1x3x!ttcore.tile<32x32, bf16>>
  %squeezed = tensor.collapse_shape %input [[0, 1], [2, 3]]
      : tensor<1x2x1x3x!ttcore.tile<32x32, bf16>>
        into tensor<2x3x!ttcore.tile<32x32, bf16>>
  %expanded = tensor.expand_shape %squeezed [[0], [1, 2]] output_shape [2, 1, 3]
      : tensor<2x3x!ttcore.tile<32x32, bf16>>
        into tensor<2x1x3x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[2, 1, 3], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<2x1x3x!ttcore.tile<32x32, bf16>>
  %result = ttl.exp %expanded
      : tensor<2x1x3x!ttcore.tile<32x32, bf16>>
        -> tensor<2x1x3x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %output
      : tensor<2x1x3x!ttcore.tile<32x32, bf16>>,
        tensor<2x1x3x!ttcore.tile<32x32, bf16>>
  return
}
func.func @bf16_io() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 2, 1, 3], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
      : !ttl.cb<[2, 1, 3], !ttcore.tile<32x32, bf16>, 2>
  %input = ttl.cb_reserve %input_dfb
      : <[1, 2, 1, 3], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x2x1x3x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %input_dfb : <[1, 2, 1, 3], !ttcore.tile<32x32, bf16>, 2>
  %output = ttl.cb_wait %output_dfb
      : <[2, 1, 3], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<2x1x3x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %output_dfb : <[2, 1, 3], !ttcore.tile<32x32, bf16>, 2>
  return
}
}

// -----

// A trailing singleton dimension preserves FP32 DFB identity and indexing.
// COMPUTE-LABEL: func.func @fp32_trailing_singleton
// COMPUTE: %[[INPUT_DFB:.*]] = ttl.bind_cb{cb_index = 0
// COMPUTE: %[[OUTPUT_DFB:.*]] = ttl.bind_cb{cb_index = 1
// COMPUTE: %[[WAIT:.*]] = ttl.cb_wait %[[INPUT_DFB]]
// COMPUTE-NEXT: %[[EXPANDED:.*]] = tensor.expand_shape %[[WAIT]]
// COMPUTE: ttl.compute ins(%[[EXPANDED]]
// COMPUTE-SAME: ttl.cb_index.0 = 0 : i64
// COMPUTE: ttl.tile_exp
// COMPUTE: ttl.tile_store
// COMPUTE: ttl.yield
// COMPUTE: ttl.cb_pop %[[INPUT_DFB]]
// COMPUTE: ttl.cb_push %[[OUTPUT_DFB]]
// COMPUTE-NEXT: return
// LOWER-LABEL: func.func @fp32_trailing_singleton
// LOWER: %[[INPUT:.*]] = ttkernel.get_compile_time_arg_val(0)
// LOWER: %[[OUTPUT:.*]] = ttkernel.get_compile_time_arg_val(1)
// LOWER: ttkernel.cb_wait_front(%[[INPUT]],
// LOWER: ttkernel.copy_tile(%[[INPUT]],
// LOWER: ttkernel.exp_tile
// LOWER: ttkernel.pack_tile({{.*}}%[[OUTPUT]]
// LOWER: ttkernel.cb_pop_front(%[[INPUT]],
// LOWER: ttkernel.cb_push_back(%[[OUTPUT]],
// COMPUTE-LABEL: func.func @fp32_io
// LOWER-LABEL: func.func @fp32_io
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
func.func @fp32_trailing_singleton()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
      : !ttl.cb<[2, 3, 1], !ttcore.tile<32x32, f32>, 2>
  %input = ttl.cb_wait %input_dfb
      : <[2, 3], !ttcore.tile<32x32, f32>, 2>
        -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %expanded = tensor.expand_shape %input [[0], [1, 2]] output_shape [2, 3, 1]
      : tensor<2x3x!ttcore.tile<32x32, f32>>
        into tensor<2x3x1x!ttcore.tile<32x32, f32>>
  %output = ttl.cb_reserve %output_dfb
      : <[2, 3, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<2x3x1x!ttcore.tile<32x32, f32>>
  %result = ttl.exp %expanded
      : tensor<2x3x1x!ttcore.tile<32x32, f32>>
        -> tensor<2x3x1x!ttcore.tile<32x32, f32>>
  ttl.store %result, %output
      : tensor<2x3x1x!ttcore.tile<32x32, f32>>,
        tensor<2x3x1x!ttcore.tile<32x32, f32>>
  return
}
func.func @fp32_io() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
      : !ttl.cb<[2, 3, 1], !ttcore.tile<32x32, f32>, 2>
  %input = ttl.cb_reserve %input_dfb
      : <[2, 3], !ttcore.tile<32x32, f32>, 2>
        -> tensor<2x3x!ttcore.tile<32x32, f32>>
  ttl.cb_push %input_dfb : <[2, 3], !ttcore.tile<32x32, f32>, 2>
  %output = ttl.cb_wait %output_dfb
      : <[2, 3, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<2x3x1x!ttcore.tile<32x32, f32>>
  ttl.cb_pop %output_dfb : <[2, 3, 1], !ttcore.tile<32x32, f32>, 2>
  return
}
}
