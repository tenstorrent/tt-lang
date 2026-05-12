// Summary: Verify ttl-set-compute-kernel-config sets kernel config on func.func.
// Attributes are per-kernel (set on the function, not individual compute ops).
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config))' --split-input-file | FileCheck %s --check-prefix=DEFAULT
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config{fp32-dest-acc-en=1 dst-full-sync-en=1}))' --split-input-file | FileCheck %s --check-prefix=OVERRIDE
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config{matmul-full-fp32=0}))' --split-input-file | FileCheck %s --check-prefix=NO-MATMUL-FP32
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config{reduce-full-fp32=0}))' --split-input-file | FileCheck %s --check-prefix=NO-REDUCE-FP32
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config))' --split-input-file | FileCheck %s --check-prefix=BLACKHOLE
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config))' --split-input-file | FileCheck %s --check-prefix=WORMHOLE
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-mark-fpu-binaries, ttl-set-compute-kernel-config))' --split-input-file | FileCheck %s --check-prefix=UNPACK
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-mark-fpu-binaries{enable-fpu-binary-ops=0}, ttl-set-compute-kernel-config))' --split-input-file | FileCheck %s --check-prefix=UNPACK-NOFPU

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: f32 tile args enable ttl.fp32_dest_acc_en on the function.
// DEFAULT-LABEL: func.func @f32_auto_enable
// DEFAULT-SAME: ttl.fp32_dest_acc_en = true
// DEFAULT-NOT: ttl.dst_full_sync_en
// OVERRIDE-LABEL: func.func @f32_auto_enable
// OVERRIDE-SAME: ttl.dst_full_sync_en = true
// OVERRIDE-SAME: ttl.fp32_dest_acc_en = true
// f32 tile args still trigger fp32 even with matmul-full-fp32=0.
// NO-MATMUL-FP32-LABEL: func.func @f32_auto_enable
// NO-MATMUL-FP32-SAME: ttl.fp32_dest_acc_en = true
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

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: bf16 with no special ops -- no ttl.fp32_dest_acc_en by default,
// but override enables both.
// DEFAULT-LABEL: func.func @bf16_no_special_ops
// DEFAULT-NOT: ttl.fp32_dest_acc_en
// DEFAULT-NOT: ttl.dst_full_sync_en
// OVERRIDE-LABEL: func.func @bf16_no_special_ops
// OVERRIDE-SAME: ttl.dst_full_sync_en = true
// OVERRIDE-SAME: ttl.fp32_dest_acc_en = true
// NO-MATMUL-FP32-LABEL: func.func @bf16_no_special_ops
// NO-MATMUL-FP32-NOT: ttl.fp32_dest_acc_en
// NO-MATMUL-FP32-NOT: ttl.dst_full_sync_en
// NO-REDUCE-FP32-LABEL: func.func @bf16_no_special_ops
// NO-REDUCE-FP32-NOT: ttl.fp32_dest_acc_en
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

// Purpose: bf16 non-ROW reduce triggers ttl.fp32_dest_acc_en through reduce-full-fp32.
// DEFAULT-LABEL: func.func @bf16_reduce_col_auto_fp32
// DEFAULT-SAME: ttl.fp32_dest_acc_en = true
// OVERRIDE-LABEL: func.func @bf16_reduce_col_auto_fp32
// OVERRIDE-SAME: ttl.fp32_dest_acc_en = true
// NO-MATMUL-FP32-LABEL: func.func @bf16_reduce_col_auto_fp32
// NO-MATMUL-FP32-SAME: ttl.fp32_dest_acc_en = true
// NO-REDUCE-FP32-LABEL: func.func @bf16_reduce_col_auto_fp32
// NO-REDUCE-FP32-NOT: ttl.fp32_dest_acc_en
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

// Purpose: bf16 ROW reduce triggers ttl.fp32_dest_acc_en when no target_arch is
// set (the issue #533 workaround applies only on Blackhole).
// DEFAULT-LABEL: func.func @bf16_reduce_row_auto_fp32
// DEFAULT-SAME: ttl.fp32_dest_acc_en = true
// OVERRIDE-LABEL: func.func @bf16_reduce_row_auto_fp32
// OVERRIDE-SAME: ttl.fp32_dest_acc_en = true
// NO-MATMUL-FP32-LABEL: func.func @bf16_reduce_row_auto_fp32
// NO-MATMUL-FP32-SAME: ttl.fp32_dest_acc_en = true
// NO-REDUCE-FP32-LABEL: func.func @bf16_reduce_row_auto_fp32
// NO-REDUCE-FP32-NOT: ttl.fp32_dest_acc_en
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

// Purpose: Blackhole ROW reduce does not trigger ttl.fp32_dest_acc_en while issue
// #533 remains open.
// BLACKHOLE-LABEL: func.func @blackhole_bf16_reduce_row_no_auto_fp32
// BLACKHOLE-SAME: attributes {ttl.kernel_thread = #ttkernel.thread<compute>}
module attributes {ttl.target_arch = "blackhole"} {
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

// Purpose: A Blackhole compute op containing both a ROW and a COL reduce
// must still enable ttl.fp32_dest_acc_en — the workaround only suppresses the
// auto-enable when the *only* fp32-justifying reduces are ROW reduces.
// BLACKHOLE-LABEL: func.func @blackhole_bf16_reduce_row_and_col_auto_fp32
// BLACKHOLE-SAME: ttl.fp32_dest_acc_en = true
module attributes {ttl.target_arch = "blackhole"} {
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

// Purpose: Wormhole reduce does not trigger ttl.fp32_dest_acc_en from
// reduce-full-fp32.
// WORMHOLE-LABEL: func.func @wormhole_bf16_reduce_row_no_auto_fp32
// WORMHOLE-NOT: ttl.fp32_dest_acc_en
// WORMHOLE-SAME: attributes {ttl.kernel_thread = #ttkernel.thread<compute>}
module attributes {ttl.target_arch = "wormhole_b0"} {
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

// Purpose: Existing func-level attributes are preserved (not overwritten).
// DEFAULT-LABEL: func.func @preserve_existing
// DEFAULT-SAME: ttl.dst_full_sync_en = false
// DEFAULT-SAME: ttl.fp32_dest_acc_en = false
// OVERRIDE-LABEL: func.func @preserve_existing
// OVERRIDE-SAME: ttl.dst_full_sync_en = false
// OVERRIDE-SAME: ttl.fp32_dest_acc_en = false
// NO-MATMUL-FP32-LABEL: func.func @preserve_existing
// NO-MATMUL-FP32-SAME: ttl.dst_full_sync_en = false
// NO-MATMUL-FP32-SAME: ttl.fp32_dest_acc_en = false
// NO-REDUCE-FP32-LABEL: func.func @preserve_existing
// NO-REDUCE-FP32-SAME: ttl.fp32_dest_acc_en = false
func.func @preserve_existing(%a: tensor<1x1x!ttcore.tile<32x32, f32>>,
                             %b: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>>
    attributes {ttl.dst_full_sync_en = false, ttl.fp32_dest_acc_en = false} {
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

// Purpose: bf16 matmul triggers ttl.fp32_dest_acc_en via matmul-full-fp32 (default).
// With matmul-full-fp32=0, bf16 matmul does not trigger ttl.fp32_dest_acc_en.
// DEFAULT-LABEL: func.func @bf16_matmul_auto_fp32
// DEFAULT-SAME: ttl.fp32_dest_acc_en = true
// OVERRIDE-LABEL: func.func @bf16_matmul_auto_fp32
// OVERRIDE-SAME: ttl.dst_full_sync_en = true
// OVERRIDE-SAME: ttl.fp32_dest_acc_en = true
// NO-MATMUL-FP32-LABEL: func.func @bf16_matmul_auto_fp32
// NO-MATMUL-FP32-NOT: ttl.fp32_dest_acc_en
// NO-REDUCE-FP32-LABEL: func.func @bf16_matmul_auto_fp32
// NO-REDUCE-FP32-SAME: ttl.fp32_dest_acc_en = true
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

// Purpose: bf16 matmul + bcast in the same kernel suppresses matmul-triggered
// ttl.fp32_dest_acc_en. unary_bcast produces incorrect results under fp32 DST
// format with bf16 CBs, so the kernel must stay in bf16 mode.
// DEFAULT-LABEL: func.func @bf16_matmul_bcast_no_fp32
// DEFAULT-NOT: ttl.fp32_dest_acc_en
// OVERRIDE-LABEL: func.func @bf16_matmul_bcast_no_fp32
// OVERRIDE-SAME: ttl.dst_full_sync_en = true
// OVERRIDE-SAME: ttl.fp32_dest_acc_en = true
// NO-MATMUL-FP32-LABEL: func.func @bf16_matmul_bcast_no_fp32
// NO-MATMUL-FP32-NOT: ttl.fp32_dest_acc_en
// NO-REDUCE-FP32-LABEL: func.func @bf16_matmul_bcast_no_fp32
// NO-REDUCE-FP32-NOT: ttl.fp32_dest_acc_en
func.func @bf16_matmul_bcast_no_fp32(
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

#map_unpack = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: f32 typecast reads its input via copy_tile -> DST, so the input CB
// needs UnpackToDestFp32. The pass must set ttl.unpack_to_dest_fp32 = true.
// DEFAULT-LABEL: func.func @f32_typecast_sets_unpack
// DEFAULT-SAME: ttl.fp32_dest_acc_en = true
// OVERRIDE-LABEL: func.func @f32_typecast_sets_unpack
// NO-MATMUL-FP32-LABEL: func.func @f32_typecast_sets_unpack
// NO-REDUCE-FP32-LABEL: func.func @f32_typecast_sets_unpack
// BLACKHOLE-LABEL: func.func @f32_typecast_sets_unpack
// WORMHOLE-LABEL: func.func @f32_typecast_sets_unpack
// UNPACK-LABEL: func.func @f32_typecast_sets_unpack
// UNPACK-SAME: ttl.unpack_to_dest_fp32 = true
// UNPACK-NOFPU-LABEL: func.func @f32_typecast_sets_unpack
// UNPACK-NOFPU-SAME: ttl.unpack_to_dest_fp32 = true
func.func @f32_typecast_sets_unpack(%a: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a_cb = ttl.attach_cb %a, %cb0
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_view = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %res = ttl.compute
      ins(%a_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map_unpack, #map_unpack],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, bf16>):
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      %cast = ttl.tile_typecast %a_tile into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, bf16>
      ttl.tile_store %cast, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  return %res : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

#map_unpack_red = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: f32 reduce reads inputs via SRCA/B (FPU strategy). The pass must
// NOT set ttl.unpack_to_dest_fp32, because UnpackToDestFp32 is incompatible
// with SRCA/B unpack. Regression test for the over-broad classifier that set
// the flag for every f32 input.
// DEFAULT-LABEL: func.func @f32_reduce_no_unpack
// DEFAULT-SAME: ttl.fp32_dest_acc_en = true
// OVERRIDE-LABEL: func.func @f32_reduce_no_unpack
// NO-MATMUL-FP32-LABEL: func.func @f32_reduce_no_unpack
// NO-REDUCE-FP32-LABEL: func.func @f32_reduce_no_unpack
// BLACKHOLE-LABEL: func.func @f32_reduce_no_unpack
// WORMHOLE-LABEL: func.func @f32_reduce_no_unpack
// UNPACK-LABEL: func.func @f32_reduce_no_unpack
// UNPACK-NOT: ttl.unpack_to_dest_fp32
// UNPACK-NOFPU-LABEL: func.func @f32_reduce_no_unpack
// UNPACK-NOFPU-NOT: ttl.unpack_to_dest_fp32
func.func @f32_reduce_no_unpack(
    %a: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %scaler: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a_cb = ttl.attach_cb %a, %cb0
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %s_cb = ttl.attach_cb %scaler, %cb1
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %out_view = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %res = ttl.compute
      ins(%a_cb, %s_cb : tensor<1x1x!ttcore.tile<32x32, f32>>,
                         tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map_unpack_red, #map_unpack_red, #map_unpack_red],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %s_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      %red = ttl.tile_reduce %a_tile, %s_tile, %out_tile 0 : i32 <reduce_dim_col> into dst[%c0] : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      ttl.tile_store %red, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>
  return %res : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

#map_unpack_mm = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: f32 matmul reads inputs via SRCA/B (FPU strategy). The pass must
// NOT set ttl.unpack_to_dest_fp32.
// DEFAULT-LABEL: func.func @f32_matmul_no_unpack
// DEFAULT-SAME: ttl.fp32_dest_acc_en = true
// OVERRIDE-LABEL: func.func @f32_matmul_no_unpack
// NO-MATMUL-FP32-LABEL: func.func @f32_matmul_no_unpack
// NO-REDUCE-FP32-LABEL: func.func @f32_matmul_no_unpack
// BLACKHOLE-LABEL: func.func @f32_matmul_no_unpack
// WORMHOLE-LABEL: func.func @f32_matmul_no_unpack
// UNPACK-LABEL: func.func @f32_matmul_no_unpack
// UNPACK-NOT: ttl.unpack_to_dest_fp32
// UNPACK-NOFPU-LABEL: func.func @f32_matmul_no_unpack
// UNPACK-NOFPU-NOT: ttl.unpack_to_dest_fp32
func.func @f32_matmul_no_unpack(
    %a: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %b: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
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
  %out_view = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %res = ttl.compute
      ins(%a_cb, %b_cb : tensor<1x1x!ttcore.tile<32x32, f32>>,
                         tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map_unpack_mm, #map_unpack_mm, #map_unpack_mm],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
      %mm = ttl.tile_matmul_block %a_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      ttl.tile_store %mm, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>
  return %res : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

#map_unpack_add = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: f32 add with both operands as input block args is marked
// ttl.fpu_binary by TTLMarkFPUBinaries (FPU strategy). The pass must NOT set
// ttl.unpack_to_dest_fp32 — the inputs go to SRCA/B, not DST. Regression for
// the case where TTLSetComputeKernelConfig misclassified pre-marking ops as
// SFPU.
// DEFAULT-LABEL: func.func @f32_fpu_add_no_unpack
// DEFAULT-SAME: ttl.fp32_dest_acc_en = true
// OVERRIDE-LABEL: func.func @f32_fpu_add_no_unpack
// NO-MATMUL-FP32-LABEL: func.func @f32_fpu_add_no_unpack
// NO-REDUCE-FP32-LABEL: func.func @f32_fpu_add_no_unpack
// BLACKHOLE-LABEL: func.func @f32_fpu_add_no_unpack
// WORMHOLE-LABEL: func.func @f32_fpu_add_no_unpack
// UNPACK-LABEL: func.func @f32_fpu_add_no_unpack
// UNPACK-NOT: ttl.unpack_to_dest_fp32
// With FPU binary marking disabled, the tile_add lowers as an SFPU binary op.
// Its f32 operands are input block args, so the pass must set
// ttl.unpack_to_dest_fp32 so copy_tile preserves f32 into DST.
// UNPACK-NOFPU-LABEL: func.func @f32_fpu_add_no_unpack
// UNPACK-NOFPU-SAME: ttl.unpack_to_dest_fp32 = true
func.func @f32_fpu_add_no_unpack(
    %a: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %b: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
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
  %out_view = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %res = ttl.compute
      ins(%a_cb, %b_cb : tensor<1x1x!ttcore.tile<32x32, f32>>,
                         tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map_unpack_add, #map_unpack_add, #map_unpack_add],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
      %add = ttl.tile_add %a_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      ttl.tile_store %add, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>
  return %res : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

#map_mixed = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: f32 fpu_add producing an intermediate, followed by an SFPU op
// (tile_exp) reading that intermediate from DST. The SFPU op does not
// unpack from a CB; its f32 operand is a DST-backed SSA value. The pass
// must NOT set ttl.unpack_to_dest_fp32 -- doing so would mis-configure CB
// 0 (the FPU input) and zero the result.
// DEFAULT-LABEL: func.func @f32_fpu_then_sfpu_no_unpack
// DEFAULT-SAME: ttl.fp32_dest_acc_en = true
// OVERRIDE-LABEL: func.func @f32_fpu_then_sfpu_no_unpack
// NO-MATMUL-FP32-LABEL: func.func @f32_fpu_then_sfpu_no_unpack
// NO-REDUCE-FP32-LABEL: func.func @f32_fpu_then_sfpu_no_unpack
// BLACKHOLE-LABEL: func.func @f32_fpu_then_sfpu_no_unpack
// WORMHOLE-LABEL: func.func @f32_fpu_then_sfpu_no_unpack
// UNPACK-LABEL: func.func @f32_fpu_then_sfpu_no_unpack
// UNPACK-NOT: ttl.unpack_to_dest_fp32
// With FPU binary marking disabled, the add falls back to SFPU and reads
// input block args from CB; the flag is then required.
// UNPACK-NOFPU-LABEL: func.func @f32_fpu_then_sfpu_no_unpack
// UNPACK-NOFPU-SAME: ttl.unpack_to_dest_fp32 = true
func.func @f32_fpu_then_sfpu_no_unpack(
    %a: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %b: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
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
  %out_view = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %res = ttl.compute
      ins(%a_cb, %b_cb : tensor<1x1x!ttcore.tile<32x32, f32>>,
                         tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map_mixed, #map_mixed, #map_mixed],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
      %sum = ttl.tile_add %a_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
      %e = ttl.tile_exp %sum into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      ttl.tile_store %e, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>
  return %res : tensor<1x1x!ttcore.tile<32x32, f32>>
}
