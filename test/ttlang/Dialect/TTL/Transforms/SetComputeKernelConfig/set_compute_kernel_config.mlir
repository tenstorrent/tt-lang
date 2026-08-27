// Verify kernel configuration resolution from operation requirements, target
// capabilities, and pass policy.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttl-set-compute-kernel-config)' --split-input-file | FileCheck %s --check-prefix=DEFAULT
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttl-set-compute-kernel-config{matmul-full-fp32=0})' --split-input-file | FileCheck %s --check-prefix=NO-MATMUL-FP32
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttl-set-compute-kernel-config{reduce-full-fp32=0})' --split-input-file | FileCheck %s --check-prefix=NO-REDUCE-FP32
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttl-set-compute-kernel-config{enable-fpu-binary-ops=0})' --split-input-file | FileCheck %s --check-prefix=FPUOFF
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttl-set-compute-kernel-config)' --split-input-file | FileCheck %s --check-prefix=BLACKHOLE
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttl-set-compute-kernel-config)' --split-input-file | FileCheck %s --check-prefix=WORMHOLE

#map = affine_map<(d0, d1) -> (d0, d1)>

// f32 DST values require f32 destination accumulation. The resolver selects an
// FPU binary strategy compatible with the complete kernel configuration.
// DEFAULT-LABEL: func.func @f32_auto_enable
// DEFAULT-SAME: dst_full_sync_en = false
// DEFAULT-SAME: fp32_dest_acc_en = true
// DEFAULT-SAME: ttl.unpack_to_dest_fp32 = array<i32>
// DEFAULT-NOT: ttl.enable_fpu_binary_ops
// DEFAULT: ttl.tile_add {{.*}}ttl.tile_execution_strategy = #ttl.tile_execution_strategy<fpu>
// Disabling a matmul preference does not alter the f32 semantic requirement.
// NO-MATMUL-FP32-LABEL: func.func @f32_auto_enable
// NO-MATMUL-FP32-SAME: fp32_dest_acc_en = true
// FPUOFF-LABEL: func.func @f32_auto_enable
// FPUOFF: ttl.tile_add {{.*}}ttl.tile_execution_strategy = #ttl.tile_execution_strategy<sfpu>
func.func @f32_auto_enable(%a: tensor<1x1x!ttcore.tile<32x32, f32>>,
                           %b: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %out_view = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %res = ttl.compute
      ins(%a_cb, %b_cb : tensor<1x1x!ttcore.tile<32x32, f32>>,
                         tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_arg: !ttcore.tile<32x32, f32>, %b_arg: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      %c0 = arith.constant 0 : index
      %sum = ttl.tile_add %a_arg, %b_arg into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
      ttl.tile_store %sum, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  return %res : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

// Blackhole accepts index 32 in both explicit unpack policy and attached DFB
// validation.
// DEFAULT-LABEL: func.func @blackhole_accepts_dfb_index_32
// DEFAULT-SAME: ttl.unpack_to_dest_fp32 = array<i32: 32>
// BLACKHOLE-LABEL: func.func @blackhole_accepts_dfb_index_32
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @blackhole_accepts_dfb_index_32(
      %input: tensor<1x1x!ttcore.tile<32x32, f32>>)
      attributes {ttl.unpack_to_dest_fp32 = array<i32: 32>} {
    %input_dfb = ttl.bind_cb {cb_index = 32, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %input_attached = ttl.attach_cb %input, %input_dfb
        : (tensor<1x1x!ttcore.tile<32x32, f32>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %zero = arith.constant 0 : index
    %input_tile = tensor.extract %input_attached[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, f32>>
    %result = ttl.tile_exp %input_tile into dst[%zero]
        : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    return
  }
}

// -----

// Direct f32 tile operations in ttl.dst_section constrain kernel config even
// when they are not nested in ttl.compute.
// DEFAULT-LABEL: func.func @direct_f32_dst_section_auto_enable
// DEFAULT-SAME: dst_full_sync_en = false
// DEFAULT-SAME: fp32_dest_acc_en = true
// DEFAULT-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0>
// NO-MATMUL-FP32-LABEL: func.func @direct_f32_dst_section_auto_enable
// NO-MATMUL-FP32-SAME: fp32_dest_acc_en = true
// NO-REDUCE-FP32-LABEL: func.func @direct_f32_dst_section_auto_enable
// NO-REDUCE-FP32-SAME: fp32_dest_acc_en = true
// FPUOFF-LABEL: func.func @direct_f32_dst_section_auto_enable
// FPUOFF-SAME: fp32_dest_acc_en = true
func.func @direct_f32_dst_section_auto_enable(
    %input: tensor<1x1x!ttcore.tile<32x32, f32>>) {
  %zero = arith.constant 0 : index
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 16, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %input_attached = ttl.attach_cb %input, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %tile = tensor.extract %input_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.dst_section {
    %dst_token, %copied = ttl.copy_tile %tile[%zero, %zero] into dst[%zero]
        : !ttcore.tile<32x32, f32> -> !ttl.dst, !ttcore.tile<32x32, f32>
    ttl.tile_store %copied, %output[%zero, %zero] from dst[%zero]
        : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
  }
  return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// A bf16 kernel without an accumulation preference selects default DST mode.
// DEFAULT-LABEL: func.func @bf16_no_special_ops
// DEFAULT-SAME: dst_full_sync_en = false
// DEFAULT-SAME: fp32_dest_acc_en = false
// NO-MATMUL-FP32-LABEL: func.func @bf16_no_special_ops
// NO-MATMUL-FP32-SAME: dst_full_sync_en = false
// NO-MATMUL-FP32-SAME: fp32_dest_acc_en = false
// NO-REDUCE-FP32-LABEL: func.func @bf16_no_special_ops
// NO-REDUCE-FP32-SAME: fp32_dest_acc_en = false
func.func @bf16_no_special_ops(%a: tensor<1x1x!ttcore.tile<32x32, bf16>>,
                               %b: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %a_cb = ttl.attach_cb %a, %cb0
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b_cb = ttl.attach_cb %b, %cb1
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init_cb = ttl.attach_cb %init, %cb2
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %out_view_0 = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %res = ttl.compute
      ins(%a_cb, %b_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                         tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_arg: !ttcore.tile<32x32, bf16>, %b_arg: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      ttl.tile_store %out, %out_view_0[%i, %j] from dst[%c0] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  return %res : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

#map_reduce_col = affine_map<(d0, d1) -> (d0, d1)>

// A supported reduce preference selects f32 DST mode.
// DEFAULT-LABEL: func.func @bf16_reduce_col_auto_fp32
// DEFAULT-SAME: fp32_dest_acc_en = true
// NO-MATMUL-FP32-LABEL: func.func @bf16_reduce_col_auto_fp32
// NO-MATMUL-FP32-SAME: fp32_dest_acc_en = true
// NO-REDUCE-FP32-LABEL: func.func @bf16_reduce_col_auto_fp32
// NO-REDUCE-FP32-SAME: fp32_dest_acc_en = false
func.func @bf16_reduce_col_auto_fp32(
    %a: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %scaler: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %a_cb = ttl.attach_cb %a, %cb0
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler_cb = ttl.attach_cb %scaler, %cb1
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init_cb = ttl.attach_cb %init, %cb2
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %out_view = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %res = ttl.compute
      ins(%a_cb, %scaler_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                              tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map_reduce_col, #map_reduce_col, #map_reduce_col],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_tile: !ttcore.tile<32x32, bf16>, %scaler_tile: !ttcore.tile<32x32, bf16>, %out_tile: !ttcore.tile<32x32, bf16>):
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      %red = ttl.tile_reduce %a_tile, %scaler_tile, %out_tile 0 : i32 <reduce_dim_col> into dst[%c0] : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      ttl.tile_store %red, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  return %res : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

#map_reduce_row = affine_map<(d0, d1) -> (d0, d1)>

// A row reduce on an unspecified target retains the supported f32 preference.
// DEFAULT-LABEL: func.func @bf16_reduce_row_auto_fp32
// DEFAULT-SAME: fp32_dest_acc_en = true
// NO-MATMUL-FP32-LABEL: func.func @bf16_reduce_row_auto_fp32
// NO-MATMUL-FP32-SAME: fp32_dest_acc_en = true
// NO-REDUCE-FP32-LABEL: func.func @bf16_reduce_row_auto_fp32
// NO-REDUCE-FP32-SAME: fp32_dest_acc_en = false
func.func @bf16_reduce_row_auto_fp32(
    %a: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %scaler: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %a_cb = ttl.attach_cb %a, %cb0
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler_cb = ttl.attach_cb %scaler, %cb1
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init_cb = ttl.attach_cb %init, %cb2
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %out_view = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %res = ttl.compute
      ins(%a_cb, %scaler_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                              tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map_reduce_row, #map_reduce_row, #map_reduce_row],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_tile: !ttcore.tile<32x32, bf16>, %scaler_tile: !ttcore.tile<32x32, bf16>, %out_tile: !ttcore.tile<32x32, bf16>):
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      %red = ttl.tile_reduce %a_tile, %scaler_tile, %out_tile 0 : i32 <reduce_dim_row> into dst[%c0] : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      ttl.tile_store %red, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  return %res : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

#map_reduce_row = affine_map<(d0, d1) -> (d0, d1)>

// Blackhole row reduce does not support full-fp32 accumulation.
// BLACKHOLE-LABEL: func.func @blackhole_bf16_reduce_row_no_auto_fp32
// BLACKHOLE-SAME: fp32_dest_acc_en = false
// BLACKHOLE-SAME: ttl.kernel_thread = #ttkernel.thread<compute>
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @blackhole_bf16_reduce_row_no_auto_fp32(
      %a: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %scaler: tensor<1x1x!ttcore.tile<32x32, bf16>>)
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %c0 = arith.constant 0 : index
    %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>

    %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

    %a_cb = ttl.attach_cb %a, %cb0
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %scaler_cb = ttl.attach_cb %scaler, %cb1
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %init_cb = ttl.attach_cb %init, %cb2
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>

    %out_view = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %res = ttl.compute
        ins(%a_cb, %scaler_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                tensor<1x1x!ttcore.tile<32x32, bf16>>)
        outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>)
        {indexing_maps = [#map_reduce_row, #map_reduce_row, #map_reduce_row],
         iterator_types = ["parallel", "parallel"]} {
      ^bb0(%a_tile: !ttcore.tile<32x32, bf16>, %scaler_tile: !ttcore.tile<32x32, bf16>, %out_tile: !ttcore.tile<32x32, bf16>):
        %i = ttl.iter_index 0 : index
        %j = ttl.iter_index 1 : index
        %red = ttl.tile_reduce %a_tile, %scaler_tile, %out_tile 0 : i32 <reduce_dim_row> into dst[%c0] : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
        ttl.tile_store %red, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.yield
    } -> tensor<1x1x!ttcore.tile<32x32, bf16>>

    return %res : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
}

// -----

#map_mixed_reduce = affine_map<(d0, d1) -> (d0, d1)>

// A supported column reduce selects f32 mode even when another reduce cannot
// use full-fp32 accumulation.
// BLACKHOLE-LABEL: func.func @blackhole_bf16_reduce_row_and_col_auto_fp32
// BLACKHOLE-SAME: fp32_dest_acc_en = true
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @blackhole_bf16_reduce_row_and_col_auto_fp32(
      %a: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %scaler: tensor<1x1x!ttcore.tile<32x32, bf16>>)
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>

    %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

    %a_cb = ttl.attach_cb %a, %cb0
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %scaler_cb = ttl.attach_cb %scaler, %cb1
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %init_cb = ttl.attach_cb %init, %cb2
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>

    %out_view = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %res = ttl.compute
        ins(%a_cb, %scaler_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                tensor<1x1x!ttcore.tile<32x32, bf16>>)
        outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>)
        {indexing_maps = [#map_mixed_reduce, #map_mixed_reduce, #map_mixed_reduce],
         iterator_types = ["parallel", "parallel"]} {
      ^bb0(%a_tile: !ttcore.tile<32x32, bf16>, %scaler_tile: !ttcore.tile<32x32, bf16>, %out_tile: !ttcore.tile<32x32, bf16>):
        %i = ttl.iter_index 0 : index
        %j = ttl.iter_index 1 : index
        %row = ttl.tile_reduce %a_tile, %scaler_tile, %out_tile 0 : i32 <reduce_dim_row> into dst[%c0] : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
        %col = ttl.tile_reduce %a_tile, %scaler_tile, %out_tile 0 : i32 <reduce_dim_col> into dst[%c1] : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
        ttl.tile_store %col, %out_view[%i, %j] from dst[%c1] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.yield
    } -> tensor<1x1x!ttcore.tile<32x32, bf16>>

    return %res : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
}

// -----

#map_reduce_row = affine_map<(d0, d1) -> (d0, d1)>

// Wormhole does not support full-fp32 reduction.
// WORMHOLE-LABEL: func.func @wormhole_bf16_reduce_row_no_auto_fp32
// WORMHOLE-SAME: fp32_dest_acc_en = false
// WORMHOLE-SAME: ttl.kernel_thread = #ttkernel.thread<compute>
module attributes {ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  func.func @wormhole_bf16_reduce_row_no_auto_fp32(
      %a: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %scaler: tensor<1x1x!ttcore.tile<32x32, bf16>>)
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %c0 = arith.constant 0 : index
    %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>

    %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

    %a_cb = ttl.attach_cb %a, %cb0
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %scaler_cb = ttl.attach_cb %scaler, %cb1
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %init_cb = ttl.attach_cb %init, %cb2
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>

    %out_view = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %res = ttl.compute
        ins(%a_cb, %scaler_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                tensor<1x1x!ttcore.tile<32x32, bf16>>)
        outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>)
        {indexing_maps = [#map_reduce_row, #map_reduce_row, #map_reduce_row],
         iterator_types = ["parallel", "parallel"]} {
      ^bb0(%a_tile: !ttcore.tile<32x32, bf16>, %scaler_tile: !ttcore.tile<32x32, bf16>, %out_tile: !ttcore.tile<32x32, bf16>):
        %i = ttl.iter_index 0 : index
        %j = ttl.iter_index 1 : index
        %red = ttl.tile_reduce %a_tile, %scaler_tile, %out_tile 0 : i32 <reduce_dim_row> into dst[%c0] : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
        ttl.tile_store %red, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.yield
    } -> tensor<1x1x!ttcore.tile<32x32, bf16>>

    return %res : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Function attributes are hard constraints and override pass preferences.
// DEFAULT-LABEL: func.func @preserve_existing
// DEFAULT-SAME: dst_full_sync_en = false
// DEFAULT-SAME: fp32_dest_acc_en = true
// NO-MATMUL-FP32-LABEL: func.func @preserve_existing
// NO-MATMUL-FP32-SAME: dst_full_sync_en = false
// NO-MATMUL-FP32-SAME: fp32_dest_acc_en = true
// NO-REDUCE-FP32-LABEL: func.func @preserve_existing
// NO-REDUCE-FP32-SAME: fp32_dest_acc_en = true
func.func @preserve_existing(%a: tensor<1x1x!ttcore.tile<32x32, f32>>,
                             %b: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>>
    attributes {dst_full_sync_en = false, fp32_dest_acc_en = true} {
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %out_view_1 = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %res = ttl.compute
      ins(%a_cb, %b_cb : tensor<1x1x!ttcore.tile<32x32, f32>>,
                         tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_arg: !ttcore.tile<32x32, f32>, %b_arg: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      ttl.tile_store %out, %out_view_1[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  return %res : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

#map3 = affine_map<(d0, d1) -> (d0, d1)>

// Matmul policy prefers f32 DST mode when supported.
// DEFAULT-LABEL: func.func @bf16_matmul_auto_fp32
// DEFAULT-SAME: fp32_dest_acc_en = true
// NO-MATMUL-FP32-LABEL: func.func @bf16_matmul_auto_fp32
// NO-MATMUL-FP32-SAME: fp32_dest_acc_en = false
// NO-REDUCE-FP32-LABEL: func.func @bf16_matmul_auto_fp32
// NO-REDUCE-FP32-SAME: fp32_dest_acc_en = true
func.func @bf16_matmul_auto_fp32(
    %a: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %b: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %a_cb = ttl.attach_cb %a, %cb0
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b_cb = ttl.attach_cb %b, %cb1
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init_cb = ttl.attach_cb %init, %cb2
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %out_view_2 = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %res = ttl.compute
      ins(%a_cb, %b_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                         tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map3, #map3, #map3],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_tile: !ttcore.tile<32x32, bf16>, %b_tile: !ttcore.tile<32x32, bf16>, %out_tile: !ttcore.tile<32x32, bf16>):
      %mm = ttl.tile_matmul_block %a_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      ttl.tile_store %mm, %out_view_2[%i, %j] from dst[%c0] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  return %res : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

#map4 = affine_map<(d0, d1) -> (d0, d1)>

// A column broadcast supports f32 DST mode, so the matmul preference remains
// effective. Disabling that preference selects default DST mode.
// DEFAULT-LABEL: func.func @bf16_matmul_column_bcast
// DEFAULT-SAME: fp32_dest_acc_en = true
// NO-MATMUL-FP32-LABEL: func.func @bf16_matmul_column_bcast
// NO-MATMUL-FP32-SAME: fp32_dest_acc_en = false
// NO-REDUCE-FP32-LABEL: func.func @bf16_matmul_column_bcast
// NO-REDUCE-FP32-SAME: fp32_dest_acc_en = true
func.func @bf16_matmul_column_bcast(
    %a: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %b: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %bias: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %a_cb = ttl.attach_cb %a, %cb0
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b_cb = ttl.attach_cb %b, %cb1
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %bias_cb = ttl.attach_cb %bias, %cb2
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init_cb = ttl.attach_cb %init, %cb3
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  // Compute with matmul and bcast in the same body.
  %out_view_3 = ttl.cb_reserve %cb3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %res2 = ttl.compute
      ins(%a_cb, %b_cb, %bias_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                    tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                    tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map4, #map4, #map4, #map4],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_tile: !ttcore.tile<32x32, bf16>, %b_tile: !ttcore.tile<32x32, bf16>,
         %bias_tile: !ttcore.tile<32x32, bf16>, %out_tile: !ttcore.tile<32x32, bf16>):
      %mm = ttl.tile_matmul_block %a_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
      %bc = ttl.tile_bcast %bias_tile, %out_tile 1 : i32 into dst[%c0] : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      %sum = ttl.tile_add %mm, %bc into dst[%c0] : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
      %i0 = ttl.iter_index 0 : index
      %j0 = ttl.iter_index 1 : index
      ttl.tile_store %sum, %out_view_3[%i0, %j0] from dst[%c0] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  return %res2 : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

// An unspecified target applies no architecture-specific broadcast
// restriction while still resolving target-independent f32 requirements.
// DEFAULT-LABEL: func.func @unspecified_target_row_broadcast
// DEFAULT-SAME: fp32_dest_acc_en = true
// DEFAULT-SAME: ttl.unpack_to_dest_fp32 = array<i32: 0>
func.func @unspecified_target_row_broadcast(
    %f32_input: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %bf16_input: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %output: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %f32_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %bf16_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %f32_attached = ttl.attach_cb %f32_input, %f32_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %bf16_attached = ttl.attach_cb %bf16_input, %bf16_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %zero = arith.constant 0 : index
  %f32_tile = tensor.extract %f32_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  %bf16_tile = tensor.extract %bf16_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output_tile = tensor.extract %output[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %exponential = ttl.tile_exp %f32_tile into dst[%zero]
      : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  %broadcast = ttl.tile_bcast %bf16_tile, %output_tile 2 : i32
      into dst[%zero]
      : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
        -> !ttcore.tile<32x32, bf16>
  return
}
