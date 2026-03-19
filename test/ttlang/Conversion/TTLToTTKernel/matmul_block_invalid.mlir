// RUN: not ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(convert-ttl-to-compute, ttl-assign-dst{enable-fpu-binary-ops=0}, ttl-insert-tile-regs-sync, ttl-lower-matmul-block))' \
// RUN:   --split-input-file 2>&1 | FileCheck %s

// bf16 DST capacity exceeded.
// CHECK: matmul output 3x3 = 9 tiles exceeds DST capacity of 8
func.func @matmul_3x3_bf16_dst_overflow(
    %arg0: tensor<3x1x!ttcore.tile<32x32, bf16>>,
    %arg1: tensor<1x3x!ttcore.tile<32x32, bf16>>) -> tensor<3x3x!ttcore.tile<32x32, bf16>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[3, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 3], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[3, 3], !ttcore.tile<32x32, bf16>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<3x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[3, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<3x1x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<1x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 3], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x3x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[3, 3], !ttcore.tile<32x32, bf16>, 2> -> tensor<3x3x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b : tensor<3x1x!ttcore.tile<32x32, bf16>>, tensor<1x3x!ttcore.tile<32x32, bf16>> -> tensor<3x3x!ttcore.tile<32x32, bf16>>
  ttl.store %mm, %reserve : tensor<3x3x!ttcore.tile<32x32, bf16>>, tensor<3x3x!ttcore.tile<32x32, bf16>>
  func.return %mm : tensor<3x3x!ttcore.tile<32x32, bf16>>
}

// -----

// f32 DST capacity exceeded.
// CHECK: matmul output 2x3 = 6 tiles exceeds DST capacity of 4
func.func @matmul_2x3_f32_dst_overflow(
    %arg0: tensor<2x1x!ttcore.tile<32x32, f32>>,
    %arg1: tensor<1x3x!ttcore.tile<32x32, f32>>) -> tensor<2x3x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 3], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<2x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<1x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 3], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x3x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb2 : <[2, 3], !ttcore.tile<32x32, f32>, 2> -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %mm = ttl.matmul %a, %b : tensor<2x1x!ttcore.tile<32x32, f32>>, tensor<1x3x!ttcore.tile<32x32, f32>> -> tensor<2x3x!ttcore.tile<32x32, f32>>
  ttl.store %mm, %reserve : tensor<2x3x!ttcore.tile<32x32, f32>>, tensor<2x3x!ttcore.tile<32x32, f32>>
  func.return %mm : tensor<2x3x!ttcore.tile<32x32, f32>>
}
