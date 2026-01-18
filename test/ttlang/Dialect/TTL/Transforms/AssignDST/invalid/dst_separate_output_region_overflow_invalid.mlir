// Summary: separate-output-region=1 should fail when outputs exceed available region.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=4 separate-output-region=1}))' --split-input-file --verify-diagnostics
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=4}))' --split-input-file | FileCheck %s --check-prefix=CHECK

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: With separate-output-region=1, inputs/intermediates use 2 registers,
// leaving only 2 registers for outputs. Three outputs that need to be live
// simultaneously exceed this capacity. Without separate-output-region, outputs
// can reuse input registers and the allocation succeeds.

func.func @separate_output_region_overflow(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                           %b: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                           %c: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> (tensor<2x2x!ttcore.tile<32x32, f32>>,
        tensor<2x2x!ttcore.tile<32x32, f32>>,
        tensor<2x2x!ttcore.tile<32x32, f32>>) {
  %init0 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init2 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb4 = ttl.bind_cb {cb_index = 17, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb5 = ttl.bind_cb {cb_index = 18, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init0_cb = ttl.attach_cb %init0, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1_cb = ttl.attach_cb %init1, %cb4 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init2_cb = ttl.attach_cb %init2, %cb5 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK-LABEL: func.func @separate_output_region_overflow
  // Note: The expected-error below is only for the first RUN line (with separate-output-region=1).
  // The second RUN line (without separate-output-region) should succeed.
  // expected-error @+1 {{insufficient DST registers for outputs: all 4 registers in use (spilling not yet implemented)}}
  %result:3 = ttl.compute
      ins(%a_cb, %b_cb, %c_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                                tensor<2x2x!ttcore.tile<32x32, f32>>,
                                tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init0_cb, %init1_cb, %init2_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                                          tensor<2x2x!ttcore.tile<32x32, f32>>,
                                          tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map, #map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  // CHECK: ttl.compute
  // CHECK: ^bb0
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %c_tile: !ttcore.tile<32x32, f32>,
       %out0_tile: !ttcore.tile<32x32, f32>,
       %out1_tile: !ttcore.tile<32x32, f32>,
       %out2_tile: !ttcore.tile<32x32, f32>):
    // Inputs a and b need to be live together (use 2 registers in Phase 3).
    // Intermediate operation keeps them live.
    // CHECK-DAG: ttl.tile_add {{.*}} {dst_idx = 0 : i32}
    %intermediate = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>

    // Three outputs that need to be live simultaneously.
    // With separate-output-region, outputs start at inputsFootprint (2),
    // so only registers 2 and 3 are available. Three outputs exceed this.
    // CHECK-DAG: ttl.tile_add {{.*}} {dst_idx = 2 : i32}
    %out0 = ttl.tile_add %intermediate, %c_tile : !ttcore.tile<32x32, f32>
    // CHECK-DAG: ttl.tile_mul {{.*}} {dst_idx = 0 : i32}
    %out1 = ttl.tile_mul %intermediate, %c_tile : !ttcore.tile<32x32, f32>
    // CHECK-DAG: ttl.tile_add {{.*}} {dst_idx = 1 : i32}
    %out2 = ttl.tile_add %out0, %out1 : !ttcore.tile<32x32, f32>

    // CHECK: ttl.yield
    ttl.yield %out0, %out1, %out2 : !ttcore.tile<32x32, f32>,
                                    !ttcore.tile<32x32, f32>,
                                    !ttcore.tile<32x32, f32>
  } -> (tensor<2x2x!ttcore.tile<32x32, f32>>,
        tensor<2x2x!ttcore.tile<32x32, f32>>,
        tensor<2x2x!ttcore.tile<32x32, f32>>)

  func.return %result#0, %result#1, %result#2 : tensor<2x2x!ttcore.tile<32x32, f32>>,
                                                  tensor<2x2x!ttcore.tile<32x32, f32>>,
                                                  tensor<2x2x!ttcore.tile<32x32, f32>>
}
