// Tests for ttl-subblock-compute-for-dst with matmul computes.
// Matmul K (reduction) accumulates in-place in DST, so only M*N parallel
// tiles count toward the DST budget. When the parallel output exceeds DST,
// subblocking partitions M*N AND tiles K to 1 for L1 accumulation.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute, ttl-set-compute-kernel-config, ttl-assign-dst{enable-fpu-binary-ops=0}, ttl-subblock-compute-for-dst))' --split-input-file | FileCheck %s

// -----

// Purpose: M*N=16 exceeds f32 DST capacity (4). Subblocking partitions the
// 4x4 output into 1x4 strips AND tiles K from 3 to 1. The K loop is
// annotated with ttl.reduction_loop for L1 accumulation.
// Loops: M (dim 0) 0..4 step 1, K (dim 2) 0..3 step 1.

// CHECK-LABEL: func.func @matmul_subblock_k_tiled
// CHECK-SAME:  fp32_dest_acc_en = true
// Outer subblock loop over M dimension.
// CHECK:       scf.for %[[MIV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// Inner K reduction loop.
// CHECK:         scf.for %[[KIV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// A sliced on M and K: [miv, kiv] [1, 1].
// CHECK:           tensor.extract_slice {{.*}}[%[[MIV]], %[[KIV]]] [1, 1] [1, 1]
// B sliced on K: [kiv, 0] [1, 4].
// CHECK:           tensor.extract_slice {{.*}}[%[[KIV]], 0] [1, 4] [1, 1]
// Output sliced on M: [miv, 0] [1, 4].
// CHECK:           tensor.extract_slice {{.*}}[%[[MIV]], 0] [1, 4] [1, 1]
// Inner compute on subblock [1, 4, 1] (M=1, N=4, K=1).
// CHECK:           ttl.compute
// CHECK-SAME:        tensor<1x1x!ttcore.tile<32x32, bf16>>
// CHECK-SAME:        tensor<1x4x!ttcore.tile<32x32, bf16>>
// CHECK-SAME:        tensor<1x4x!ttcore.tile<32x32, bf16>>
// CHECK-SAME:        iterator_types = ["parallel", "parallel", "reduction"]
// CHECK:             ttl.tile_matmul_block
// K loop annotated for L1 accumulation.
// CHECK:         } {{{.*}}ttl.reduction_loop{{.*}}}
// CHECK:       }
func.func @matmul_subblock_k_tiled(
    %arg0: tensor<4x3x!ttcore.tile<32x32, bf16>>,
    %arg1: tensor<3x4x!ttcore.tile<32x32, bf16>>) -> tensor<4x4x!ttcore.tile<32x32, bf16>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 3], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[3, 4], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 3], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x3x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<3x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[3, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<3x4x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[4, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b : tensor<4x3x!ttcore.tile<32x32, bf16>>, tensor<3x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  ttl.store %mm, %reserve : tensor<4x4x!ttcore.tile<32x32, bf16>>, tensor<4x4x!ttcore.tile<32x32, bf16>>
  func.return %mm : tensor<4x4x!ttcore.tile<32x32, bf16>>
}

// -----

// Purpose: M*N=4 fits in f32 DST capacity (4) with K=3. No subblock loop
// needed -- the entire matmul fits in one DST sync region.

// CHECK-LABEL: func.func @matmul_fits_in_dst
// CHECK-SAME:  fp32_dest_acc_en = true
// No subblock loop.
// CHECK-NOT:   scf.for
// CHECK:       ttl.compute
// CHECK-SAME:    tensor<2x3x!ttcore.tile<32x32, bf16>>
// CHECK-SAME:    tensor<3x2x!ttcore.tile<32x32, bf16>>
// CHECK-SAME:    tensor<2x2x!ttcore.tile<32x32, bf16>>
// CHECK-SAME:    ttl.full_linearization_strides
// CHECK:         ttl.tile_matmul_block
func.func @matmul_fits_in_dst(
    %arg0: tensor<2x3x!ttcore.tile<32x32, bf16>>,
    %arg1: tensor<3x2x!ttcore.tile<32x32, bf16>>) -> tensor<2x2x!ttcore.tile<32x32, bf16>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[3, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<2x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<3x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[3, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<3x2x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b : tensor<2x3x!ttcore.tile<32x32, bf16>>, tensor<3x2x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.store %mm, %reserve : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>
  func.return %mm : tensor<2x2x!ttcore.tile<32x32, bf16>>
}
