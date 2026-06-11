// Matmul outputs exceeding DST capacity are rejected when the subblock pass
// is not in the pipeline. With subblocking enabled, these cases compile
// successfully (tested by simple_matmul_subblock.py).
// RUN: not ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(convert-ttl-to-compute, ttl-set-compute-kernel-config{enable-fpu-binary-ops=0 matmul-full-fp32=0 reduce-full-fp32=0}, ttl-assign-dst, ttl-lower-to-loops))' \
// RUN:   --split-input-file 2>&1 | FileCheck %s

// bf16 DST capacity exceeded.
// CHECK: output 3x3 with 1 DST slots per tile = 9 total slots exceeds DST capacity of 8
func.func @matmul_3x3_bf16_dst_overflow(
    %arg0: tensor<3x1x!ttcore.tile<32x32, bf16>>,
    %arg1: tensor<1x3x!ttcore.tile<32x32, bf16>>) -> tensor<3x3x!ttcore.tile<32x32, bf16>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[3, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 3], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[3, 3], !ttcore.tile<32x32, bf16>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<3x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[3, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<3x1x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<1x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 3], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x3x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[3, 3], !ttcore.tile<32x32, bf16>, 2> -> tensor<3x3x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b : tensor<3x1x!ttcore.tile<32x32, bf16>>, tensor<1x3x!ttcore.tile<32x32, bf16>> -> tensor<3x3x!ttcore.tile<32x32, bf16>>
  ttl.store %mm, %reserve : tensor<3x3x!ttcore.tile<32x32, bf16>>, tensor<3x3x!ttcore.tile<32x32, bf16>>
  func.return %mm : tensor<3x3x!ttcore.tile<32x32, bf16>>
}

// -----

// f32 DST capacity exceeded.
// CHECK: output 2x3 with 1 DST slots per tile = 6 total slots exceeds DST capacity of 4
func.func @matmul_2x3_f32_dst_overflow(
    %arg0: tensor<2x1x!ttcore.tile<32x32, f32>>,
    %arg1: tensor<1x3x!ttcore.tile<32x32, f32>>) -> tensor<2x3x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 3], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<2x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<1x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 3], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x3x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb2 : <[2, 3], !ttcore.tile<32x32, f32>, 2> -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %mm = ttl.matmul %a, %b : tensor<2x1x!ttcore.tile<32x32, f32>>, tensor<1x3x!ttcore.tile<32x32, f32>> -> tensor<2x3x!ttcore.tile<32x32, f32>>
  ttl.store %mm, %reserve : tensor<2x3x!ttcore.tile<32x32, f32>>, tensor<2x3x!ttcore.tile<32x32, f32>>
  func.return %mm : tensor<2x3x!ttcore.tile<32x32, f32>>
}

// -----

// Scaled accumulator exceeding DST capacity exercises the >1-slot-per-tile
// accounting: each of the two output tiles needs its output slot plus scratch
// for the broadcasted scale tile and old-state copy, so 2x3 = 6 slots > f32
// capacity 4.
// CHECK: output 1x2 with 3 DST slots per tile = 6 total slots exceeds DST capacity of 4
func.func @scaled_acc_2slot_f32_dst_overflow(
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
