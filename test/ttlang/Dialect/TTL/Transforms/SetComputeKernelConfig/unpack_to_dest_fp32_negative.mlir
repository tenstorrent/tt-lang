// Verify DFB inputs that do not unpack through DST retain default mode.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config))' --split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: tile_reduce consumes f32 input through SRCA/SRCB, so it must not
// request unpack-to-DST mode for the input CB.
// CHECK-LABEL: func.func @f32_reduce_does_not_set_unpack
// CHECK-SAME: ttl.unpack_to_dest_fp32 = array<i32>
func.func @f32_reduce_does_not_set_unpack(
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
  %scaler_cb = ttl.attach_cb %scaler, %cb1
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %out_view = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %res = ttl.compute
      ins(%a_cb, %scaler_cb : tensor<1x1x!ttcore.tile<32x32, f32>>,
                              tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %scaler_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      %red = ttl.tile_reduce %a_tile, %scaler_tile, %out_tile 0 : i32 <reduce_dim_col> into dst[%c0] : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      ttl.tile_store %red, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  return %res : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: An SFPU op consuming an upstream FPU result reads from DST, not from
// a CB, so no unpack-to-DST mode is needed for the original f32 block args.
// CHECK-LABEL: func.func @f32_fpu_result_to_sfpu_does_not_set_unpack
// CHECK-SAME: ttl.unpack_to_dest_fp32 = array<i32>
func.func @f32_fpu_result_to_sfpu_does_not_set_unpack(
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
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      %sum = ttl.tile_add %a_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
      %exp = ttl.tile_exp %sum into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
      ttl.tile_store %exp, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  return %res : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: bf16 -> f32 typecast from CB0 does not need f32 unpack-to-DST mode
// because the CB0 input itself is not f32.
// CHECK-LABEL: func.func @bf16_to_f32_typecast_does_not_set_unpack
// CHECK-SAME: ttl.unpack_to_dest_fp32 = array<i32>
func.func @bf16_to_f32_typecast_does_not_set_unpack(
    %a: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init_cb = ttl.attach_cb %init, %cb1
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %out_view = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %res = ttl.compute
      ins(%a_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_tile: !ttcore.tile<32x32, bf16>, %out_tile: !ttcore.tile<32x32, f32>):
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      %cast = ttl.tile_typecast %a_tile into dst[%c0] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, f32>
      ttl.tile_store %cast, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  return %res : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

#init_map = affine_map<(d0, d1, d2) -> (d0, d1)>
#contrib_map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#out_map = affine_map<(d0, d1, d2) -> (d0, d1)>

// Purpose: tile_accumulate lowers to binary_dest_reuse_tiles, whose
// contribution operand must keep the default f32 unpack mode.
// CHECK-LABEL: func.func @f32_tile_accumulate_does_not_set_unpack
func.func @f32_tile_accumulate_does_not_set_unpack() {
  %c0 = arith.constant 0 : index
  %init_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %contrib_cb = ttl.bind_cb {cb_index = 1, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>
  %out_cb = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %init_wait = ttl.cb_wait %init_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init = ttl.attach_cb %init_wait, %init_cb : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %contrib_wait = ttl.cb_wait %contrib_cb {num_tiles = 3 : i64} : <[1, 1], !ttcore.tile<32x32, f32>, 4> -> tensor<3x1x1x!ttcore.tile<32x32, f32>>
  %contrib = ttl.attach_cb %contrib_wait, %contrib_cb : (tensor<3x1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>) -> tensor<3x1x1x!ttcore.tile<32x32, f32>>
  %out_view = ttl.cb_reserve %out_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %empty = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
  %out = ttl.attach_cb %empty, %out_cb : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%init, %contrib : tensor<1x1x!ttcore.tile<32x32, f32>>,
                             tensor<3x1x1x!ttcore.tile<32x32, f32>>)
      outs(%out : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#init_map, #contrib_map, #out_map],
       iterator_types = ["parallel", "parallel", "reduction"]} {
  ^bb0(%init_tile: !ttcore.tile<32x32, f32>,
       %contrib_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %acc = ttl.tile_accumulate %init_tile, %contrib_tile add into dst[%c0]
        : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>
          -> !ttcore.tile<32x32, f32>
    ttl.tile_store %acc, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>
  func.return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: tile_bcast consuming an f32 CB must NOT enable UnpackToDestFp32
// (tt-llk #1338); with no SFPU consumer on that CB there is also no conflict.
// CHECK-LABEL: func.func @bcast_only_f32_does_not_set_unpack
// CHECK-SAME: ttl.unpack_to_dest_fp32 = array<i32>
func.func @bcast_only_f32_does_not_set_unpack(
    %a: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %out_view = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %res = ttl.compute
      ins(%a_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      %bc = ttl.tile_bcast %a_tile, %out_tile 2 : i32 into dst[%c0] : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      ttl.tile_store %bc, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>
  return %res : tensor<1x1x!ttcore.tile<32x32, f32>>
}
