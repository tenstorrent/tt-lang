#layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  ttcore.global @a = tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> [0]
  ttcore.global @b = tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> [1]
  ttcore.global @c = tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> [2]
  func.func @test_chained_add(%arg0: tensor<32x32xf32> {d2m.stream = true}, %arg1: tensor<32x32xf32> {d2m.stream = true}, %arg2: tensor<32x32xf32> {d2m.stream = true}, %arg3: tensor<32x32xf32> {d2m.stream = false}) -> tensor<32x32xf32> {
    %0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %1 = d2m.to_layout %arg0, %0 : tensor<32x32xf32> into tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %2 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %stream = "d2m.stream_layout"(%1, %2) : (tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %3 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %4 = d2m.to_layout %arg1, %3 : tensor<32x32xf32> into tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %5 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %stream_0 = "d2m.stream_layout"(%4, %5) : (tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %6 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %7 = d2m.to_layout %arg2, %6 : tensor<32x32xf32> into tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %8 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %stream_1 = "d2m.stream_layout"(%7, %8) : (tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %9 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %10 = d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%stream, %stream_0, %stream_1 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
        outs(%9 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)  {
    ^datamovement0(%cb0: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb2: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb3: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
      %12 = ttcore.get_global @a : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
      %13 = d2m.reserve %cb0 : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %c0 = arith.constant 0 : index
      %c0_2 = arith.constant 0 : index
      %tx = d2m.dma %12 [%c0, %c0_2], %13 : (tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !d2m.mem_tx
      d2m.dma_wait %tx
    }, {
    ^datamovement1(%cb0: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb2: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb3: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
      %12 = ttcore.get_global @b : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
      %13 = d2m.reserve %cb1 : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %c0 = arith.constant 0 : index
      %c0_2 = arith.constant 0 : index
      %tx = d2m.dma %12 [%c0, %c0_2], %13 : (tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !d2m.mem_tx
      d2m.dma_wait %tx
    }, {
    ^datamovement2(%cb0: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb2: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb3: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
      %12 = ttcore.get_global @c : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
      %13 = d2m.reserve %cb2 : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %c0 = arith.constant 0 : index
      %c0_2 = arith.constant 0 : index
      %tx = d2m.dma %12 [%c0, %c0_2], %13 : (tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !d2m.mem_tx
      d2m.dma_wait %tx
    }, {
    ^compute0(%cb0: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb2: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb3: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
      %12 = d2m.wait %cb0 : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %13 = d2m.wait %cb1 : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %14 = d2m.wait %cb2 : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %15 = d2m.reserve %cb3 : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %16 = d2m.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
      %17 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%12, %13 : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>) outs(%16 : tensor<1x1x!ttcore.tile<32x32, f32>>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %in_2: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %21 = "d2m.tile_add"(%in, %in_2) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %21 : !ttcore.tile<32x32, f32>
      } -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %18 = d2m.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
      %19 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%17, %14 : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>) outs(%18 : tensor<1x1x!ttcore.tile<32x32, f32>>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %in_2: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %21 = "d2m.tile_add"(%in, %in_2) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %21 : !ttcore.tile<32x32, f32>
      } -> tensor<1x1x!ttcore.tile<32x32, f32>>
      d2m.store %15, %19 : tensor<1x1x!ttcore.tile<32x32, f32>>
      %20 = d2m.wait %cb3 : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    } : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %11 = d2m.to_layout %10, %arg3 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> into tensor<32x32xf32> -> tensor<32x32xf32>
    return %11 : tensor<32x32xf32>
  }
}

