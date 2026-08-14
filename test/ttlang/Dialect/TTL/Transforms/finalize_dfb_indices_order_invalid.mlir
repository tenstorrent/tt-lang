// Verifies that DFB finalization rejects attributes containing copied
// provisional indices from passes that must run after finalization.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-set-compute-kernel-config,ttl-finalize-dfb-indices)'

// Compute configuration copies an f32 input DFB index to the kernel.
// expected-error @below {{'func.func' op contains derived DFB-index attribute 'ttl.unpack_to_dest_fp32' before DFB index finalization}}
func.func @unpack_configuration_before_finalization()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.unpack_to_dest_fp32 = array<i32: 1>} {
  %compiler_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      {ttl.compiler_allocated}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  return
}

// -----

#identity = affine_map<(d0, d1) -> (d0, d1)>

// DFB association annotation copies each compute input index.
func.func @compute_association_before_finalization()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %zero = arith.constant 0 : index
  %empty = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %compiler_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      {ttl.compiler_allocated}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %input = ttl.attach_cb %empty, %compiler_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.attach_cb %empty, %output_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output_view = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.compute' op contains derived DFB-index attribute 'ttl.cb_index.0' before DFB index finalization}}
  %result = ttl.compute
      ins(%input : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%output : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#identity, #identity],
       iterator_types = ["parallel", "parallel"],
       ttl.cb_index.0 = 1 : index} {
    ^bb0(%input_tile: !ttcore.tile<32x32, bf16>,
         %output_tile: !ttcore.tile<32x32, bf16>):
      %row = ttl.iter_index 0 : index
      %column = ttl.iter_index 1 : index
      %value = ttl.tile_exp %input_tile into dst[%zero]
          : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
      ttl.tile_store %value, %output_view[%row, %column] from dst[%zero]
          : !ttcore.tile<32x32, bf16>,
            tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// Broadcast annotation copies its output DFB index to the tile operation.
func.func @broadcast_association_before_finalization()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %zero = arith.constant 0 : index
  %empty = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %compiler_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      {ttl.compiler_allocated}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %attached = ttl.attach_cb %empty, %compiler_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %tile = tensor.extract %attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.tile_bcast' op contains derived DFB-index attribute 'ttl.bcast_output_cb_index' before DFB index finalization}}
  %result = ttl.tile_bcast %tile, %tile 1 : i32 into dst[%zero]
      {ttl.bcast_output_cb_index = 1 : index}
      : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
        -> !ttcore.tile<32x32, bf16>
  return
}

// -----

// Reduce annotation copies its output DFB index to the tile operation.
func.func @reduce_association_before_finalization()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %zero = arith.constant 0 : index
  %empty = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %compiler_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      {ttl.compiler_allocated}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %attached = ttl.attach_cb %empty, %compiler_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %tile = tensor.extract %attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.tile_reduce' op contains derived DFB-index attribute 'ttl.reduce_output_cb_index' before DFB index finalization}}
  %result = ttl.tile_reduce %tile, %tile, %tile 0 : i32 <reduce_dim_col>
      into dst[%zero] {ttl.reduce_output_cb_index = 1 : index}
      : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>,
         !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
  return
}

// -----

// Transpose annotation copies its output DFB index to the tile operation.
func.func @transpose_association_before_finalization()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %zero = arith.constant 0 : index
  %empty = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %compiler_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      {ttl.compiler_allocated}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %attached = ttl.attach_cb %empty, %compiler_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %tile = tensor.extract %attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.tile_transpose' op contains derived DFB-index attribute 'ttl.transpose_output_cb_index' before DFB index finalization}}
  %result = ttl.tile_transpose %tile, %tile into dst[%zero]
      {ttl.transpose_output_cb_index = 1 : index}
      : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
        -> !ttcore.tile<32x32, bf16>
  return
}
