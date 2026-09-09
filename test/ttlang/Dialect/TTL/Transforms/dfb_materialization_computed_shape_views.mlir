// Shape views of computed block expressions preserve values through the full
// pipeline, not only when applied directly to acquired DFB tensors.
// RUN: ttlang-opt %s --split-input-file --ttl-to-ttkernel-pipeline | FileCheck %s --implicit-check-not=tensor.expand_shape --implicit-check-not=tensor.collapse_shape --implicit-check-not=builtin.unrealized_conversion_cast

// A computed BF16 expression can lose a leading singleton before its store.
// The compiler-created CB must receive the computed tiles before publication,
// then be waited on, copied to the output, and popped after that copy.
// CHECK-LABEL: func.func @squeezed_computed_expression
// CHECK: %[[SIX_BF16:.*]] = arith.constant 6 : i32
// CHECK: %[[INPUT_BF16:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<12, !ttcore.tile<32x32, bf16>>
// CHECK-NEXT: %[[OUTPUT_BF16:.*]] = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<12, !ttcore.tile<32x32, bf16>>
// CHECK-NEXT: %[[TEMP_BF16:.*]] = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<6, !ttcore.tile<32x32, bf16>>
// CHECK-NEXT: ttkernel.cb_wait_front(%[[INPUT_BF16]], %[[SIX_BF16]])
// CHECK-NEXT: ttkernel.cb_reserve_back(%[[TEMP_BF16]], %[[SIX_BF16]])
// CHECK: ttkernel.copy_tile(%[[INPUT_BF16]],
// CHECK: ttkernel.exp_tile
// CHECK: ttkernel.pack_tile_block({{[^,]+}}, %[[TEMP_BF16]],
// CHECK-NEXT: ttkernel.tile_regs_release()
// CHECK-NEXT: ttkernel.cb_push_back(%[[TEMP_BF16]], %[[SIX_BF16]])
// CHECK-NEXT: ttkernel.cb_wait_front(%[[TEMP_BF16]], %[[SIX_BF16]])
// CHECK-NEXT: ttkernel.cb_reserve_back(%[[OUTPUT_BF16]], %[[SIX_BF16]])
// CHECK: ttkernel.copy_tile(%[[TEMP_BF16]],
// CHECK: ttkernel.pack_tile_block({{[^,]+}}, %[[OUTPUT_BF16]],
// CHECK-NEXT: ttkernel.tile_regs_release()
// CHECK-NEXT: ttkernel.cb_pop_front(%[[TEMP_BF16]], %[[SIX_BF16]])
// CHECK-NEXT: ttkernel.cb_pop_front(%[[INPUT_BF16]], %[[SIX_BF16]])
// CHECK-NEXT: ttkernel.cb_push_back(%[[OUTPUT_BF16]], %[[SIX_BF16]])
// CHECK-NEXT: return
// CHECK-LABEL: func.func @squeezed_expression_io
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
func.func @squeezed_computed_expression()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 2, 3], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
      : !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 2>
  %input = ttl.cb_wait %input_dfb
      : <[1, 2, 3], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x2x3x!ttcore.tile<32x32, bf16>>
  %result = ttl.exp %input : tensor<1x2x3x!ttcore.tile<32x32, bf16>>
      -> tensor<1x2x3x!ttcore.tile<32x32, bf16>>
  %squeezed = tensor.collapse_shape %result [[0, 1], [2]]
      : tensor<1x2x3x!ttcore.tile<32x32, bf16>>
        into tensor<2x3x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[2, 3], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  ttl.store %squeezed, %output
      : tensor<2x3x!ttcore.tile<32x32, bf16>>,
        tensor<2x3x!ttcore.tile<32x32, bf16>>
  return
}
func.func @squeezed_expression_io()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 2, 3], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
      : !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 2>
  %input = ttl.cb_reserve %input_dfb
      : <[1, 2, 3], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x2x3x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %input_dfb : <[1, 2, 3], !ttcore.tile<32x32, bf16>, 2>
  %output = ttl.cb_wait %output_dfb
      : <[2, 3], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %output_dfb : <[2, 3], !ttcore.tile<32x32, bf16>, 2>
  return
}
}

// -----

// A computed FP32 expression can acquire a trailing singleton before its store.
// FP32 subblocking must preserve the same intermediate publication and lifetime.
// CHECK-LABEL: func.func @unsqueezed_computed_expression
// CHECK: %[[SIX_FP32:.*]] = arith.constant 6 : i32
// CHECK: %[[INPUT_FP32:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<12, !ttcore.tile<32x32, f32>>
// CHECK-NEXT: %[[OUTPUT_FP32:.*]] = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<12, !ttcore.tile<32x32, f32>>
// CHECK-NEXT: %[[TEMP_FP32:.*]] = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<6, !ttcore.tile<32x32, f32>>
// CHECK-NEXT: ttkernel.cb_wait_front(%[[INPUT_FP32]], %[[SIX_FP32]])
// CHECK-NEXT: ttkernel.cb_reserve_back(%[[TEMP_FP32]], %[[SIX_FP32]])
// CHECK: ttkernel.copy_tile(%[[INPUT_FP32]],
// CHECK: ttkernel.exp_tile
// CHECK: ttkernel.pack_tile({{[^,]+}}, %[[TEMP_FP32]],
// CHECK: ttkernel.tile_regs_release()
// CHECK-NEXT: } {ttl.subblock_dim
// CHECK-NEXT: ttkernel.cb_push_back(%[[TEMP_FP32]], %[[SIX_FP32]])
// CHECK-NEXT: ttkernel.cb_wait_front(%[[TEMP_FP32]], %[[SIX_FP32]])
// CHECK-NEXT: ttkernel.cb_reserve_back(%[[OUTPUT_FP32]], %[[SIX_FP32]])
// CHECK: ttkernel.copy_tile(%[[TEMP_FP32]],
// CHECK: ttkernel.pack_tile({{[^,]+}}, %[[OUTPUT_FP32]],
// CHECK: ttkernel.tile_regs_release()
// CHECK-NEXT: } {ttl.subblock_dim
// CHECK-NEXT: ttkernel.cb_pop_front(%[[TEMP_FP32]], %[[SIX_FP32]])
// CHECK-NEXT: ttkernel.cb_pop_front(%[[INPUT_FP32]], %[[SIX_FP32]])
// CHECK-NEXT: ttkernel.cb_push_back(%[[OUTPUT_FP32]], %[[SIX_FP32]])
// CHECK-NEXT: return
// CHECK-LABEL: func.func @unsqueezed_expression_io
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
func.func @unsqueezed_computed_expression()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
      : !ttl.cb<[2, 3, 1], !ttcore.tile<32x32, f32>, 2>
  %input = ttl.cb_wait %input_dfb
      : <[2, 3], !ttcore.tile<32x32, f32>, 2>
        -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %result = ttl.exp %input : tensor<2x3x!ttcore.tile<32x32, f32>>
      -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %expanded = tensor.expand_shape %result [[0], [1, 2]] output_shape [2, 3, 1]
      : tensor<2x3x!ttcore.tile<32x32, f32>>
        into tensor<2x3x1x!ttcore.tile<32x32, f32>>
  %output = ttl.cb_reserve %output_dfb
      : <[2, 3, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<2x3x1x!ttcore.tile<32x32, f32>>
  ttl.store %expanded, %output
      : tensor<2x3x1x!ttcore.tile<32x32, f32>>,
        tensor<2x3x1x!ttcore.tile<32x32, f32>>
  return
}
func.func @unsqueezed_expression_io()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
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
