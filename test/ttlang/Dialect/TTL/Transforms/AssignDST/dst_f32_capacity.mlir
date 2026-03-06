// Summary: Verify f32 compute ops use reduced DST capacity.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst),canonicalize,cse)' --split-input-file | FileCheck %s
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8}),canonicalize,cse)' --split-input-file | FileCheck %s --check-prefix=OVERRIDE

#idx_map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: f32 tiles use dst_idx in [0-3] with default (double-buffered) capacity.
// CHECK-LABEL: func.func @f32_add
// CHECK: ttl.tile_add {{.*}} {dst_idx = [[IDX0:[0-3]]] : i32}
func.func @f32_add(%a: tensor<1x1x!ttcore.tile<32x32, f32>>,
                   %b: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

  %cba = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbb = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cba
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cbb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cbout
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %out_view = ttl.cb_reserve %cbout : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %res = ttl.compute
      ins(%a_cb, %b_cb : tensor<1x1x!ttcore.tile<32x32, f32>>,
                         tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#idx_map, #idx_map, #idx_map],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_arg: !ttcore.tile<32x32, f32>, %b_arg: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
      %c0 = arith.constant 0 : index
      %dtok0, %dtile0 = ttl.copy_tile %a_arg, %c0, %c0 : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
      %dtok1, %dtile1 = ttl.copy_tile %b_arg, %c0, %c0 : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
      %add = ttl.tile_add %dtile0, %dtile1 : !ttcore.tile<32x32, f32>
      ttl.tile_store %add, %out_view : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  return %res : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

#idx_map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: Manual dst-capacity=8 override widens the index range to [0-7].
// OVERRIDE-LABEL: func.func @f32_capacity_override
// OVERRIDE: ttl.tile_add {{.*}} {dst_idx = [[OVRIDX0:[0-7]]] : i32}
func.func @f32_capacity_override(%a: tensor<1x1x!ttcore.tile<32x32, f32>>,
                             %b: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

  %cba = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbb = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cba
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cbb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cbout
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %out_view_0 = ttl.cb_reserve %cbout : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %res = ttl.compute
      ins(%a_cb, %b_cb : tensor<1x1x!ttcore.tile<32x32, f32>>,
                         tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#idx_map, #idx_map, #idx_map],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_arg: !ttcore.tile<32x32, f32>, %b_arg: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
      %c0 = arith.constant 0 : index
      %dtok0, %dtile0 = ttl.copy_tile %a_arg, %c0, %c0 : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
      %dtok1, %dtile1 = ttl.copy_tile %b_arg, %c0, %c0 : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
      %add = ttl.tile_add %dtile0, %dtile1 : !ttcore.tile<32x32, f32>
      ttl.tile_store %add, %out_view_0 : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  return %res : tensor<1x1x!ttcore.tile<32x32, f32>>
}
