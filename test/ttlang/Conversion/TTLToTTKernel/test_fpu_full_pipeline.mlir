// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-assign-dst,ttl-lower-to-loops),convert-ttl-to-ttkernel)' | FileCheck %s
// Summary: Full pipeline test for FPU optimization with actual ttl.compute regions.
//
// This test verifies that when binary operations receive operands directly from
// circular buffers (via block arguments in ttl.compute), the FPU pattern generates
// ttkernel.add_tiles (FPU) instead of ttkernel.add_binary_tile (SFPU).

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @fpu_binary_add
func.func @fpu_binary_add(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>>, %arg1: tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  // Bind circular buffers
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>

  %a = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // The FPU optimization should activate here
  // CHECK: ttkernel.add_tiles_init
  // CHECK: ttkernel.add_tiles
  // CHECK-NOT: ttkernel.add_binary_tile_init
  // CHECK-NOT: ttkernel.add_binary_tile(%c
  %result = ttl.add %a, %b : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<1x1x!ttcore.tile<32x32, f32>>
}
