// Test: DST intermediate reused with in-place unary chain (issue #384).
// Pattern:
//   x = mul(a, b)           -- FPU binary, result in DST
//   abs_x = abs(x)          -- unary (in-place, destructive)
//   rsqrt_x = rsqrt(abs_x)  -- unary (in-place)
//   result = mul(x, rsqrt_x) -- SFPU binary consuming original x and chain result
//
// 'x' has two consumers: the abs chain and the final mul. Since abs is in-place,
// it would destroy x. The allocator must insert copy_dst to preserve x.
//
// Expected DST assignment:
//   x (FPU mul)      -> DST[0]
//   copy_dst(x)      -> DST[1]  (preserve x for abs chain)
//   abs(copy)        -> DST[1]  (in-place on copy)
//   rsqrt(abs)       -> DST[1]  (in-place on abs result)
//   mul(x, rsqrt)    -> DST[0]  (SFPU binary, consumes original x at DST[0])

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8}),canonicalize)' | FileCheck %s

// CHECK-LABEL: func.func @dst_intermediate_reuse_unary_chain
// CHECK:           ttl.compute
// CHECK-NEXT:      ^bb0(%[[A:.*]]: !ttcore.tile<32x32, bf16>, %[[B:.*]]: !ttcore.tile<32x32, bf16>, %[[OUT:.*]]: !ttcore.tile<32x32, bf16>):
// CHECK-NEXT:        %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:        %[[I1:.*]] = ttl.iter_index 1 : index
// FPU binary mul: both operands are block args -> DST[0]
// CHECK-NEXT:      %[[X:.*]] = ttl.tile_mul %[[A]], %[[B]] into dst[%c0] {ttl.fpu_binary} : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
// copy_dst preserves x in DST[1] before destructive abs
// CHECK-NEXT:      %[[COPY:.*]] = ttl.copy_dst %[[X]] into dst[%c1] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
// abs overwrites DST[1] in-place
// CHECK-NEXT:      %[[ABS:.*]] = ttl.tile_abs %[[COPY]] into dst[%c1] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
// rsqrt overwrites DST[1] in-place
// CHECK-NEXT:      %[[RSQRT:.*]] = ttl.tile_rsqrt %[[ABS]] into dst[%c1] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
// SFPU binary mul: x at DST[0], rsqrt at DST[1] -> DST[0]
// CHECK-NEXT:      %[[RESULT:.*]] = ttl.tile_mul %[[X]], %[[RSQRT]] into dst[%c0] : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
// CHECK:           ttl.tile_store %[[RESULT]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:      ttl.yield

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @dst_intermediate_reuse_unary_chain(
    %a: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %b: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %out_view = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute
    ins(%a_cb, %b_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>)
    outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>)
    {indexing_maps = [#map, #map, #map],
     iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, bf16>,
       %b_tile: !ttcore.tile<32x32, bf16>,
       %out_tile: !ttcore.tile<32x32, bf16>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index

    // x = a * b (FPU binary, result is DST intermediate)
    %c0 = arith.constant 0 : index
    %x = ttl.tile_mul %a_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    // Unary chain on x: abs then rsqrt (both in-place, destructive)
    %abs_x = ttl.tile_abs %x into dst[%c0] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    %rsqrt_x = ttl.tile_rsqrt %abs_x into dst[%c0] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    // Final SFPU binary consuming original x and chain result
    %final = ttl.tile_mul %x, %rsqrt_x into dst[%c0] : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    ttl.tile_store %final, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>

    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  func.return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}
