// Verify explicit unpack-to-DST-f32 entries for non-f32 dataflow buffers do not
// require 32-bit destination elements.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config{matmul-full-fp32=0 reduce-full-fp32=0}))' | FileCheck %s --check-prefix=AUTO
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config{fp32-dest-acc-en=disabled matmul-full-fp32=0 reduce-full-fp32=0}))' | FileCheck %s --check-prefix=DISABLED

// A fixed SFPU consumer ignores the f32 route entry for bf16.
// AUTO-LABEL: func.func @bf16_sfpu_unary
// AUTO-SAME: fp32_dest_acc_en = false
// AUTO-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0>
// DISABLED-LABEL: func.func @bf16_sfpu_unary
// DISABLED-SAME: fp32_dest_acc_en = false
// DISABLED-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0>
func.func @bf16_sfpu_unary(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.unpack_to_dest_fp32 = array<i32: 0>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %attached = ttl.attach_cb %input, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %zero = arith.constant 0 : index
  %tile = tensor.extract %attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %absolute = ttl.tile_abs %tile into dst[%zero]
      : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
  return
}

// -----

// A fixed source-register consumer ignores the f32 route entry for
// bf16.
// AUTO-LABEL: func.func @bf16_broadcast
// AUTO-SAME: fp32_dest_acc_en = false
// AUTO-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0>
// DISABLED-LABEL: func.func @bf16_broadcast
// DISABLED-SAME: fp32_dest_acc_en = false
// DISABLED-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0>
func.func @bf16_broadcast(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.unpack_to_dest_fp32 = array<i32: 0>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %attached = ttl.attach_cb %input, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %zero = arith.constant 0 : index
  %tile = tensor.extract %attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %broadcast = ttl.tile_bcast %tile, %tile 2 : i32 into dst[%zero]
      : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
        -> !ttcore.tile<32x32, bf16>
  return
}

// -----

// A strategy-dependent binary consumer ignores the f32 route entry for
// bf16.
// AUTO-LABEL: func.func @bf16_strategy_binary
// AUTO-SAME: fp32_dest_acc_en = false
// AUTO-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0>
// DISABLED-LABEL: func.func @bf16_strategy_binary
// DISABLED-SAME: fp32_dest_acc_en = false
// DISABLED-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0>
func.func @bf16_strategy_binary(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.unpack_to_dest_fp32 = array<i32: 0>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %attached = ttl.attach_cb %input, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %zero = arith.constant 0 : index
  %tile = tensor.extract %attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sum = ttl.tile_add %tile, %tile into dst[%zero]
      : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>
        -> !ttcore.tile<32x32, bf16>
  return
}

// -----

// An explicit DFB-to-destination copy ignores the f32 route entry for
// bf16.
// AUTO-LABEL: func.func @bf16_copy
// AUTO-SAME: fp32_dest_acc_en = false
// AUTO-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0>
// DISABLED-LABEL: func.func @bf16_copy
// DISABLED-SAME: fp32_dest_acc_en = false
// DISABLED-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0>
func.func @bf16_copy(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.unpack_to_dest_fp32 = array<i32: 0>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %attached = ttl.attach_cb %input, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %zero = arith.constant 0 : index
  %tile = tensor.extract %attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %token, %copy = ttl.copy_tile %tile[%zero, %zero] into dst[%zero]
      : !ttcore.tile<32x32, bf16>
        -> !ttl.dst, !ttcore.tile<32x32, bf16>
  return
}

// -----

// A fixed SFPU consumer ignores the f32 route entry for f16.
// AUTO-LABEL: func.func @f16_sfpu_unary
// AUTO-SAME: fp32_dest_acc_en = false
// AUTO-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0>
// DISABLED-LABEL: func.func @f16_sfpu_unary
// DISABLED-SAME: fp32_dest_acc_en = false
// DISABLED-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0>
func.func @f16_sfpu_unary(
    %input: tensor<1x1x!ttcore.tile<32x32, f16>>)
    attributes {ttl.unpack_to_dest_fp32 = array<i32: 0>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f16>, 2>
  %attached = ttl.attach_cb %input, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, f16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f16>>
  %zero = arith.constant 0 : index
  %tile = tensor.extract %attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f16>>
  %absolute = ttl.tile_abs %tile into dst[%zero]
      : !ttcore.tile<32x32, f16> -> !ttcore.tile<32x32, f16>
  return
}

// -----

// A fixed source-register consumer ignores the f32 route entry for
// f16.
// AUTO-LABEL: func.func @f16_broadcast
// AUTO-SAME: fp32_dest_acc_en = false
// AUTO-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0>
// DISABLED-LABEL: func.func @f16_broadcast
// DISABLED-SAME: fp32_dest_acc_en = false
// DISABLED-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0>
func.func @f16_broadcast(
    %input: tensor<1x1x!ttcore.tile<32x32, f16>>)
    attributes {ttl.unpack_to_dest_fp32 = array<i32: 0>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f16>, 2>
  %attached = ttl.attach_cb %input, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, f16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f16>>
  %zero = arith.constant 0 : index
  %tile = tensor.extract %attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f16>>
  %broadcast = ttl.tile_bcast %tile, %tile 2 : i32 into dst[%zero]
      : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>)
        -> !ttcore.tile<32x32, f16>
  return
}

// -----

// A strategy-dependent binary consumer ignores the f32 route entry for
// f16.
// AUTO-LABEL: func.func @f16_strategy_binary
// AUTO-SAME: fp32_dest_acc_en = false
// AUTO-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0>
// DISABLED-LABEL: func.func @f16_strategy_binary
// DISABLED-SAME: fp32_dest_acc_en = false
// DISABLED-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0>
func.func @f16_strategy_binary(
    %input: tensor<1x1x!ttcore.tile<32x32, f16>>)
    attributes {ttl.unpack_to_dest_fp32 = array<i32: 0>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f16>, 2>
  %attached = ttl.attach_cb %input, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, f16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f16>>
  %zero = arith.constant 0 : index
  %tile = tensor.extract %attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f16>>
  %sum = ttl.tile_add %tile, %tile into dst[%zero]
      : !ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>
        -> !ttcore.tile<32x32, f16>
  return
}

// -----

// An explicit DFB-to-destination copy ignores the f32 route entry for
// f16.
// AUTO-LABEL: func.func @f16_copy
// AUTO-SAME: fp32_dest_acc_en = false
// AUTO-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0>
// DISABLED-LABEL: func.func @f16_copy
// DISABLED-SAME: fp32_dest_acc_en = false
// DISABLED-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0>
func.func @f16_copy(
    %input: tensor<1x1x!ttcore.tile<32x32, f16>>)
    attributes {ttl.unpack_to_dest_fp32 = array<i32: 0>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f16>, 2>
  %attached = ttl.attach_cb %input, %dfb
      : (tensor<1x1x!ttcore.tile<32x32, f16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f16>>
  %zero = arith.constant 0 : index
  %tile = tensor.extract %attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f16>>
  %token, %copy = ttl.copy_tile %tile[%zero, %zero] into dst[%zero]
      : !ttcore.tile<32x32, f16>
        -> !ttl.dst, !ttcore.tile<32x32, f16>
  return
}
