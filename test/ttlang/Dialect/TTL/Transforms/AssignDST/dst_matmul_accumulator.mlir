// Summary: Matmul accumulator and output share the same DST register.
// The 3-operand tile_matmul_block(lhs, rhs, accumulator) has its accumulator
// and output intervals merged so they receive the same dst_idx. The actual
// copy_tile to pre-load the accumulator into DST is emitted during TTKernel
// lowering, not here; this test verifies the interval merge only.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8}),canonicalize,cse)' | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @matmul_with_accumulator
// Accumulator and output merged to dst_idx = 0. Three operands preserved.
// CHECK: ^bb0(%[[A:.*]]: !ttcore.tile<32x32, bf16>, %[[B:.*]]: !ttcore.tile<32x32, bf16>, %[[C:.*]]: !ttcore.tile<32x32, bf16>, %{{.*}}: !ttcore.tile<32x32, bf16>):
// CHECK: %[[MM:.*]] = ttl.tile_matmul_block %[[A]], %[[B]], %[[C]] {dst_idx = 0 : i32}
func.func @matmul_with_accumulator(
    %a: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %b: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %c: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c_cb = ttl.attach_cb %c, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init_cb = ttl.attach_cb %init, %cb3 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %out_view = ttl.cb_reserve %cb3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute
      ins(%a_cb, %b_cb, %c_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_t: !ttcore.tile<32x32, bf16>, %b_t: !ttcore.tile<32x32, bf16>, %c_t: !ttcore.tile<32x32, bf16>, %out_t: !ttcore.tile<32x32, bf16>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %mm = ttl.tile_matmul_block %a_t, %b_t, %c_t : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    ttl.tile_store %mm, %out_view[%i, %j] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

// With a post-matmul relu: accumulator, matmul output, and relu all share
// dst_idx via interval merging (accumulator + in-place chain).

#map2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @matmul_accumulator_relu
// CHECK: ^bb0(%[[A2:.*]]: !ttcore.tile<32x32, bf16>, %[[B2:.*]]: !ttcore.tile<32x32, bf16>, %[[C2:.*]]: !ttcore.tile<32x32, bf16>, %{{.*}}: !ttcore.tile<32x32, bf16>):
// CHECK: %[[MM2:.*]] = ttl.tile_matmul_block %[[A2]], %[[B2]], %[[C2]] {dst_idx = 0 : i32}
// CHECK: ttl.tile_relu %[[MM2]] {dst_idx = 0 : i32}
func.func @matmul_accumulator_relu(
    %a: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %b: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %c: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c_cb = ttl.attach_cb %c, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init_cb = ttl.attach_cb %init, %cb3 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %out_view = ttl.cb_reserve %cb3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute
      ins(%a_cb, %b_cb, %c_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map2, #map2, #map2, #map2],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_t: !ttcore.tile<32x32, bf16>, %b_t: !ttcore.tile<32x32, bf16>, %c_t: !ttcore.tile<32x32, bf16>, %out_t: !ttcore.tile<32x32, bf16>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %mm = ttl.tile_matmul_block %a_t, %b_t, %c_t : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    %r = ttl.tile_relu %mm : !ttcore.tile<32x32, bf16>
    ttl.tile_store %r, %out_view[%i, %j] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}
