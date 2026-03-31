// Verify that per-subblock reserve/push is NOT applied when the output CB has
// multiple reserves or multiple pushes. The subblock tiling itself should still
// happen, but the original reserve/push ops must be preserved.
//
// RUN: ttlang-opt %s \
// RUN:   --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=4},ttl-subblock-compute-for-dst{subblock-sync=true}))' \
// RUN:   --split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// Multiple cb_reserve ops for the same output CB: per-subblock sync skipped.
// The compute is 4x4 with unary exp (dstPerIteration=1, unroll=4, subblock=[1,4]).
// Outer loop over dim0 with step 1. Original reserves and push remain outside the loop.

// CHECK-LABEL: func.func @multi_reserve_skip
// The two original reserves must survive (not be replaced by per-subblock ones).
// CHECK-DAG:     ttl.cb_reserve %[[CB:.*]] : {{.*}} -> tensor<4x4x
// CHECK-DAG:     ttl.cb_reserve %[[CB]] : {{.*}} -> tensor<4x4x
// Subblock loop is still generated.
// CHECK:         scf.for
// CHECK:           ttl.compute
// CHECK-SAME:      tensor<1x4x!ttcore.tile<32x32, f32>>
// The original push must survive outside the loop.
// CHECK:         ttl.cb_push

func.func @multi_reserve_skip(%a: tensor<4x4x!ttcore.tile<32x32, f32>>)
    -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2}
      : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, buffer_factor = 2}
      : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0
      : (tensor<4x4x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>)
      -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1
      : (tensor<4x4x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>)
      -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // Two reserves on the same output CB — triggers the skip.
  %reserve1 = ttl.cb_reserve %cb1
      : <[4, 4], !ttcore.tile<32x32, f32>, 2>
      -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %reserve2 = ttl.cb_reserve %cb1
      : <[4, 4], !ttcore.tile<32x32, f32>, 2>
      -> tensor<4x4x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb : tensor<4x4x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<4x4x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %exp = ttl.tile_exp %a_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %reserve1[%i, %j]
        : !ttcore.tile<32x32, f32>, tensor<4x4x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<4x4x!ttcore.tile<32x32, f32>>

  ttl.cb_push %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2>

  func.return %result : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Multiple cb_push ops for the same output CB: per-subblock sync skipped.

// CHECK-LABEL: func.func @multi_push_skip
// The original reserve must survive.
// CHECK:         ttl.cb_reserve
// Subblock loop is still generated.
// CHECK:         scf.for
// CHECK:           ttl.compute
// CHECK-SAME:      tensor<1x4x!ttcore.tile<32x32, f32>>
// Both original pushes must survive outside the loop.
// CHECK:         ttl.cb_push
// CHECK:         ttl.cb_push

func.func @multi_push_skip(%a: tensor<4x4x!ttcore.tile<32x32, f32>>)
    -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2}
      : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, buffer_factor = 2}
      : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0
      : (tensor<4x4x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>)
      -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1
      : (tensor<4x4x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>)
      -> tensor<4x4x!ttcore.tile<32x32, f32>>

  %reserve = ttl.cb_reserve %cb1
      : <[4, 4], !ttcore.tile<32x32, f32>, 2>
      -> tensor<4x4x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb : tensor<4x4x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<4x4x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %exp = ttl.tile_exp %a_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %reserve[%i, %j]
        : !ttcore.tile<32x32, f32>, tensor<4x4x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // Two pushes on the same output CB — triggers the skip.
  ttl.cb_push %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2>
  ttl.cb_push %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2>

  func.return %result : tensor<4x4x!ttcore.tile<32x32, f32>>
}
