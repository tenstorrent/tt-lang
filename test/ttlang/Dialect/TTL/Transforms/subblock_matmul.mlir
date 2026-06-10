// Tests for ttl-subblock-compute-for-dst with matmul computes.
// Matmul K (reduction) accumulates in-place in DST, so only M*N parallel
// tiles count toward the DST budget. Subblocking partitions the M*N output
// space while keeping K whole in each subblock.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute, ttl-set-compute-kernel-config, ttl-set-compute-kernel-config{enable-fpu-binary-ops=0 matmul-full-fp32=0 reduce-full-fp32=0}, ttl-assign-dst, ttl-subblock-compute-for-dst))' --split-input-file | FileCheck %s

// -----

// Purpose: M*N=16 exceeds f32 DST capacity (4). K=3 is excluded from the
// budget, so subblocking partitions the 4x4 output into 1x4 strips.
// Loop on M (dim 0): 0 to 4 step 1. K (dim 2) stays at 3 in each subblock.

// CHECK-LABEL: func.func @matmul_subblock_k_excluded
// CHECK-SAME:  fp32_dest_acc_en = true
// Outer subblock loop over M dimension.
// CHECK:       scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// A sliced on M, K kept whole: [iv, 0] [1, 3].
// CHECK:         tensor.extract_slice {{.*}}[%[[IV]], 0] [1, 3] [1, 1]
// B not sliced (full [3, 4]).
// CHECK:         tensor.extract_slice {{.*}}[0, 0] [3, 4] [1, 1]
// Output sliced on M: [iv, 0] [1, 4].
// CHECK:         tensor.extract_slice {{.*}}[%[[IV]], 0] [1, 4] [1, 1]
// Inner compute on subblock [1, 4, 3] (M=1, N=4, K=3).
// CHECK:         ttl.compute
// CHECK-SAME:      tensor<1x3x!ttcore.tile<32x32, bf16>>
// CHECK-SAME:      tensor<3x4x!ttcore.tile<32x32, bf16>>
// CHECK-SAME:      tensor<1x4x!ttcore.tile<32x32, bf16>>
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
// CHECK:           ttl.tile_matmul_block
// CHECK:       }
func.func @matmul_subblock_k_excluded(
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

// -----

// Purpose: A block matmul with a scaled accumulator uses one output slot plus
// two scratch slots per output tile in the current lowering. f32 DST capacity
// is 4, so only one output tile fits even though the output has two tiles.
// Subblocking must still emit a loop when the chosen output subblock product
// is one.

// CHECK-LABEL: func.func @scaled_acc_matmul_one_output_tile_subblock
// CHECK-SAME:  fp32_dest_acc_en = true
// CHECK:       scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:         tensor.extract_slice {{.*}}[0, %[[IV]]] [1, 1] [1, 1]
// CHECK:         ttl.compute
// CHECK-SAME:      tensor<1x1x!ttcore.tile<32x32, f32>>
// CHECK-SAME:      tensor<1x1x!ttcore.tile<32x32, f32>>
// CHECK-SAME:      tensor<1x1x!ttcore.tile<32x32, f32>>
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
// CHECK:           ttl.tile_matmul_block
// CHECK:       }
func.func @scaled_acc_matmul_one_output_tile_subblock(
    %alpha: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %o_old: tensor<1x2x!ttcore.tile<32x32, f32>>,
    %exp_scores: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %v: tensor<1x2x!ttcore.tile<32x32, f32>>) -> tensor<1x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
  %cb4 = ttl.bind_cb {cb_index = 4, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
  %scores_attached = ttl.attach_cb %exp_scores, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %v_attached = ttl.attach_cb %v, %cb1 : (tensor<1x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x2x!ttcore.tile<32x32, f32>>
  %alpha_attached = ttl.attach_cb %alpha, %cb2 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %old_attached = ttl.attach_cb %o_old, %cb3 : (tensor<1x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x2x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb4 : <[1, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<1x2x!ttcore.tile<32x32, f32>>
  %alpha_bcast = ttl.block.broadcast %alpha_attached dims = [-1], shape = [1, 2] : tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x2x!ttcore.tile<32x32, f32>>
  %scaled = ttl.mul %alpha_bcast, %old_attached : tensor<1x2x!ttcore.tile<32x32, f32>>, tensor<1x2x!ttcore.tile<32x32, f32>> -> tensor<1x2x!ttcore.tile<32x32, f32>>
  %mm = ttl.matmul %scores_attached, %v_attached : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x2x!ttcore.tile<32x32, f32>> -> tensor<1x2x!ttcore.tile<32x32, f32>>
  %out = ttl.add %scaled, %mm : tensor<1x2x!ttcore.tile<32x32, f32>>, tensor<1x2x!ttcore.tile<32x32, f32>> -> tensor<1x2x!ttcore.tile<32x32, f32>>
  ttl.store %out, %reserve : tensor<1x2x!ttcore.tile<32x32, f32>>, tensor<1x2x!ttcore.tile<32x32, f32>>
  func.return %out : tensor<1x2x!ttcore.tile<32x32, f32>>
}
