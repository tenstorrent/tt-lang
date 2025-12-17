// RUN: ttlang-opt %s -split-input-file
// Purpose: positive coverage for ttl.compute with CB operands, including CB reuse.

// Simple compute with distinct CBs.
func.func @compute_with_cbs(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                            %b: tensor<2x2x!ttcore.tile<32x32, f32>>,
                            %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
                            %cbb: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
                            %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute
      ins(%a, %b : tensor<2x2x!ttcore.tile<32x32, f32>>,
                     tensor<2x2x!ttcore.tile<32x32, f32>>)
      in_cbs(%cba, %cbb : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
                         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      outs(%init : tensor<2x2x!ttcore.tile<32x32, f32>>)
      out_cbs(%cbout : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%at: !ttcore.tile<32x32, f32>,
       %bt: !ttcore.tile<32x32, f32>,
       %ct: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %at, %bt : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// CB reuse when the same tensor accessor is used twice.
func.func @compute_with_cbs_reuse(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                  %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
                                  %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute
      ins(%a, %a : tensor<2x2x!ttcore.tile<32x32, f32>>,
                     tensor<2x2x!ttcore.tile<32x32, f32>>)
      in_cbs(%cba, %cba : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
                         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      outs(%init : tensor<2x2x!ttcore.tile<32x32, f32>>)
      out_cbs(%cbout : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%at0: !ttcore.tile<32x32, f32>,
       %at1: !ttcore.tile<32x32, f32>,
       %ct: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %at0, %at1 : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}
