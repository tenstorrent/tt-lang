// Summary: Verify ttl-set-compute-kernel-config sets kernel config on func.func.
// Attributes are per-kernel (set on the function, not individual compute ops).
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config))' --split-input-file | FileCheck %s --check-prefix=DEFAULT
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config{fp32-dest-acc-en=1 dst-full-sync-en=1}))' --split-input-file | FileCheck %s --check-prefix=OVERRIDE
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config{matmul-full-fp32=0}))' --split-input-file | FileCheck %s --check-prefix=NO-MATMUL-FP32

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: f32 tile args enable fp32_dest_acc_en on the function.
// DEFAULT-LABEL: func.func @f32_auto_enable
// DEFAULT-SAME: fp32_dest_acc_en = true
// DEFAULT-NOT: dst_full_sync_en
// OVERRIDE-LABEL: func.func @f32_auto_enable
// OVERRIDE-SAME: dst_full_sync_en = true
// OVERRIDE-SAME: fp32_dest_acc_en = true
// f32 tile args still trigger fp32 even with matmul-full-fp32=0.
// NO-MATMUL-FP32-LABEL: func.func @f32_auto_enable
// NO-MATMUL-FP32-SAME: fp32_dest_acc_en = true
func.func @f32_auto_enable(%a: tensor<1x1x!ttcore.tile<32x32, f32>>,
                           %b: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>

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
      %sum = ttl.tile_add %a_arg, %b_arg : !ttcore.tile<32x32, f32>
      ttl.tile_store %sum, %out_view[%i, %j] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  return %res : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// --split-input-file

// Purpose: bf16 with no special ops -- no fp32_dest_acc_en by default,
// but override enables both.
// DEFAULT-LABEL: func.func @bf16_no_special_ops
// DEFAULT-NOT: fp32_dest_acc_en
// DEFAULT-NOT: dst_full_sync_en
// OVERRIDE-LABEL: func.func @bf16_no_special_ops
// OVERRIDE-SAME: dst_full_sync_en = true
// OVERRIDE-SAME: fp32_dest_acc_en = true
// NO-MATMUL-FP32-LABEL: func.func @bf16_no_special_ops
// NO-MATMUL-FP32-NOT: fp32_dest_acc_en
// NO-MATMUL-FP32-NOT: dst_full_sync_en
func.func @bf16_no_special_ops(%a: tensor<1x1x!ttcore.tile<32x32, bf16>>,
                               %b: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

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
      ttl.tile_store %out, %out_view_0[%i, %j] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  return %res : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// --split-input-file

// Purpose: Existing func-level attributes are preserved (not overwritten).
// DEFAULT-LABEL: func.func @preserve_existing
// DEFAULT-SAME: dst_full_sync_en = false
// DEFAULT-SAME: fp32_dest_acc_en = false
// OVERRIDE-LABEL: func.func @preserve_existing
// OVERRIDE-SAME: dst_full_sync_en = false
// OVERRIDE-SAME: fp32_dest_acc_en = false
// NO-MATMUL-FP32-LABEL: func.func @preserve_existing
// NO-MATMUL-FP32-SAME: dst_full_sync_en = false
// NO-MATMUL-FP32-SAME: fp32_dest_acc_en = false
func.func @preserve_existing(%a: tensor<1x1x!ttcore.tile<32x32, f32>>,
                             %b: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>>
    attributes {dst_full_sync_en = false, fp32_dest_acc_en = false} {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>

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
      ttl.tile_store %out, %out_view_1[%i, %j] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  return %res : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// --split-input-file

#map3 = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: bf16 matmul triggers fp32_dest_acc_en via matmul-full-fp32 (default).
// With matmul-full-fp32=0, bf16 matmul does not trigger fp32_dest_acc_en.
// DEFAULT-LABEL: func.func @bf16_matmul_auto_fp32
// DEFAULT-SAME: fp32_dest_acc_en = true
// OVERRIDE-LABEL: func.func @bf16_matmul_auto_fp32
// OVERRIDE-SAME: dst_full_sync_en = true
// OVERRIDE-SAME: fp32_dest_acc_en = true
// NO-MATMUL-FP32-LABEL: func.func @bf16_matmul_auto_fp32
// NO-MATMUL-FP32-NOT: fp32_dest_acc_en
func.func @bf16_matmul_auto_fp32(
    %a: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %b: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

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
      %mm = ttl.tile_matmul_block %a_tile, %b_tile : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      ttl.tile_store %mm, %out_view_2[%i, %j] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  return %res : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// --split-input-file

#map4 = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: bf16 matmul + bcast in the same kernel suppresses matmul-triggered
// fp32_dest_acc_en. unary_bcast produces incorrect results under fp32 DST
// format with bf16 CBs, so the kernel must stay in bf16 mode.
// DEFAULT-LABEL: func.func @bf16_matmul_bcast_no_fp32
// DEFAULT-NOT: fp32_dest_acc_en
// OVERRIDE-LABEL: func.func @bf16_matmul_bcast_no_fp32
// OVERRIDE-SAME: dst_full_sync_en = true
// OVERRIDE-SAME: fp32_dest_acc_en = true
// NO-MATMUL-FP32-LABEL: func.func @bf16_matmul_bcast_no_fp32
// NO-MATMUL-FP32-NOT: fp32_dest_acc_en
func.func @bf16_matmul_bcast_no_fp32(
    %a: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %b: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %bias: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

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
      %mm = ttl.tile_matmul_block %a_tile, %b_tile : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
      %bc = ttl.tile_bcast %bias_tile, %out_tile 1 : i32 : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      %sum = ttl.tile_add %mm, %bc : !ttcore.tile<32x32, bf16>
      %i0 = ttl.iter_index 0 : index
      %j0 = ttl.iter_index 1 : index
      ttl.tile_store %sum, %out_view_3[%i0, %j0] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  return %res2 : tensor<1x1x!ttcore.tile<32x32, bf16>>
}
