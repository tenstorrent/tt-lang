// Summary: Regression coverage for scaled accumulator expressions folded into
// matmul_block. Verifies that scale * acc is computed into DST before
// matmul_block accumulates A @ B into the same output slots.

// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(convert-ttl-to-compute, ttl-set-compute-kernel-config{enable-fpu-binary-ops=0 matmul-full-fp32=0 reduce-full-fp32=0}, ttl-assign-dst, ttl-lower-to-loops))' \
// RUN:   --split-input-file | FileCheck %s --check-prefix=TTL
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(convert-ttl-to-compute, ttl-set-compute-kernel-config{enable-fpu-binary-ops=0 matmul-full-fp32=0 reduce-full-fp32=0}, ttl-assign-dst, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   --split-input-file | FileCheck %s --check-prefix=TTK

// Simple scaled accumulator: out = scale * acc + (a @ b). The store must use
// the indexed post-matmul DST slot, not the raw ranged matmul result.

// TTL-LABEL: func.func @scaled_acc_matmul
// TTL-DAG: %[[C0:.*]] = arith.constant 0 : index
// TTL:     %{{.*}}, %[[SCALE_COPY:.*]] = ttl.copy_tile
// TTL:     %{{.*}}, %[[ACC_COPY:.*]] = ttl.copy_tile
// TTL:     %[[SCALED:.*]] = ttl.tile_mul %[[SCALE_COPY]], %[[ACC_COPY]] into dst[%[[C0]]]
// TTL:     %[[MM:.*]] = ttl.tile_matmul_block {{.*}} into dst[%[[C0]]]
// TTL-NOT: ttl.tile_add
// TTL:     %[[FINAL:.*]] = ttl.dst_index %[[MM]][%[[C0]]]
// TTL:     ttl.tile_store %[[FINAL]],
//
// TTK-LABEL: func.func @scaled_acc_matmul
// TTK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
// TTK-DAG: %[[C1_I32:.*]] = arith.constant 1 : i32
// TTK-DAG: %[[C0:.*]] = arith.constant 0 : index
// TTK-DAG: %[[CB_A:.*]] = ttkernel.get_compile_time_arg_val(0)
// TTK-DAG: %[[CB_B:.*]] = ttkernel.get_compile_time_arg_val(1)
// TTK-DAG: %[[CB_SCALE:.*]] = ttkernel.get_compile_time_arg_val(2)
// TTK-DAG: %[[CB_ACC:.*]] = ttkernel.get_compile_time_arg_val(3)
// TTK-DAG: %[[CB_OUT:.*]] = ttkernel.get_compile_time_arg_val(4)
// TTK:     ttkernel.tile_regs_acquire
// TTK:     ttkernel.copy_tile_init(%[[CB_SCALE]])
// TTK:     ttkernel.copy_tile(%[[CB_SCALE]], %[[C0]], {{.*}})
// TTK:     ttkernel.copy_tile_init(%[[CB_ACC]])
// TTK:     ttkernel.copy_tile(%[[CB_ACC]], %[[C0]], {{.*}})
// TTK:     ttkernel.mul_binary_tile({{.*}}, {{.*}}, %[[C0]])
// TTK:     ttkernel.matmul_block(%[[CB_A]], %[[CB_B]], %[[C0]], %[[C0]], %[[C0]], %[[C0_I32]], %[[C1_I32]], %[[C1_I32]], %[[C1_I32]])
// TTK:     ttkernel.tile_regs_commit
// TTK-NEXT: ttkernel.tile_regs_wait
// TTK-NEXT: ttkernel.pack_tile(%[[C0]], %[[CB_OUT]], %[[C0]]
// TTK-NEXT: ttkernel.tile_regs_release
// TTK-NOT: ttl.dst_index
func.func @scaled_acc_matmul(
    %scale: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %acc: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %a: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %b: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb4 = ttl.bind_cb {cb_index = 4, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a_attached = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b_attached = ttl.attach_cb %b, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scale_attached = ttl.attach_cb %scale, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %acc_attached = ttl.attach_cb %acc, %cb3 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaled = ttl.mul %scale_attached, %acc_attached : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a_attached, %b_attached : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out = ttl.add %scaled, %mm : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %out, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return %out : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

// Flash-shaped variant: alpha is broadcast across the V/output dimension and
// V has two tiles. The two scaled accumulator tiles must prefill DST[0] and
// DST[1] before one matmul_block accumulates both output tiles.

// TTL-LABEL: func.func @flash_broadcast_scaled_acc_multi_v
// TTL-DAG: %[[C0:.*]] = arith.constant 0 : index
// TTL-DAG: %[[C1:.*]] = arith.constant 1 : index
// TTL:     ttl.tile_mul {{.*}} into dst[%[[C0]]]
// TTL:     ttl.tile_mul {{.*}} into dst[%[[C1]]]
// TTL:     %[[MM:.*]] = ttl.tile_matmul_block {{.*}} into dst[%[[C0]]]
// TTL:     %[[FINAL0:.*]] = ttl.dst_index %[[MM]][%[[C0]]]
// TTL:     %[[FINAL1:.*]] = ttl.dst_index %[[MM]][%[[C1]]]
// TTL:     ttl.tile_store %[[FINAL0]],
// TTL:     ttl.tile_store %[[FINAL1]],
//
// TTK-LABEL: func.func @flash_broadcast_scaled_acc_multi_v
// TTK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
// TTK-DAG: %[[C1_I32:.*]] = arith.constant 1 : i32
// TTK-DAG: %[[C2_I32:.*]] = arith.constant 2 : i32
// TTK-DAG: %[[C0:.*]] = arith.constant 0 : index
// TTK-DAG: %[[C1:.*]] = arith.constant 1 : index
// TTK-DAG: %[[CB_SCORES:.*]] = ttkernel.get_compile_time_arg_val(0)
// TTK-DAG: %[[CB_V:.*]] = ttkernel.get_compile_time_arg_val(1)
// TTK-DAG: %[[CB_ALPHA:.*]] = ttkernel.get_compile_time_arg_val(2)
// TTK-DAG: %[[CB_OLD:.*]] = ttkernel.get_compile_time_arg_val(3)
// TTK-DAG: %[[CB_OUT:.*]] = ttkernel.get_compile_time_arg_val(4)
// TTK:     ttkernel.tile_regs_acquire
// TTK:     ttkernel.mul_binary_tile({{.*}}, {{.*}}, %[[C0]])
// TTK:     ttkernel.mul_binary_tile({{.*}}, {{.*}}, %[[C1]])
// TTK:     ttkernel.matmul_block(%[[CB_SCORES]], %[[CB_V]], %[[C0]], %[[C0]], %[[C0]], %[[C0_I32]], %[[C2_I32]], %[[C1_I32]], %[[C1_I32]])
// TTK:     ttkernel.tile_regs_commit
// TTK-NEXT: ttkernel.tile_regs_wait
// TTK-NEXT: ttkernel.pack_tile(%[[C0]], %[[CB_OUT]], %[[C0]]
// TTK-NEXT: ttkernel.pack_tile(%[[C1]], %[[CB_OUT]], %[[C1]]
// TTK-NEXT: ttkernel.tile_regs_release
// TTK-NOT: ttl.dst_index
func.func @flash_broadcast_scaled_acc_multi_v(
    %alpha: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %o_old: tensor<1x2x!ttcore.tile<32x32, bf16>>,
    %exp_scores: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %v: tensor<1x2x!ttcore.tile<32x32, bf16>>) -> tensor<1x2x!ttcore.tile<32x32, bf16>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb4 = ttl.bind_cb {cb_index = 4, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %scores_attached = ttl.attach_cb %exp_scores, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %v_attached = ttl.attach_cb %v, %cb1 : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %alpha_attached = ttl.attach_cb %alpha, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %old_attached = ttl.attach_cb %o_old, %cb3 : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb4 : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %alpha_bcast = ttl.block.broadcast %alpha_attached dims = [-1], shape = [1, 2] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %scaled = ttl.mul %alpha_bcast, %old_attached : tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, bf16>> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %scores_attached, %v_attached : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, bf16>> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %out = ttl.add %scaled, %mm : tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, bf16>> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.store %out, %reserve : tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, bf16>>
  func.return %out : tensor<1x2x!ttcore.tile<32x32, bf16>>
}

// -----

// f32 variant of the simple scaled accumulator. The prefill-then-accumulate
// folding is identical to bf16, and the 3 DST slots used (1 output + 2 scratch
// copies) fit within the halved f32 DST capacity of 4.

// TTL-LABEL: func.func @scaled_acc_matmul_f32
// TTL-DAG: %[[C0:.*]] = arith.constant 0 : index
// TTL:     %{{.*}}, %[[SCALE_COPY:.*]] = ttl.copy_tile
// TTL:     %{{.*}}, %[[ACC_COPY:.*]] = ttl.copy_tile
// TTL:     %[[SCALED:.*]] = ttl.tile_mul %[[SCALE_COPY]], %[[ACC_COPY]] into dst[%[[C0]]]
// TTL:     %[[MM:.*]] = ttl.tile_matmul_block {{.*}} into dst[%[[C0]]]
// TTL-NOT: ttl.tile_add
// TTL:     %[[FINAL:.*]] = ttl.dst_index %[[MM]][%[[C0]]]
// TTL:     ttl.tile_store %[[FINAL]],
//
// TTK-LABEL: func.func @scaled_acc_matmul_f32
// TTK-DAG: %[[C0:.*]] = arith.constant 0 : index
// TTK-DAG: %[[CB_A:.*]] = ttkernel.get_compile_time_arg_val(0)
// TTK-DAG: %[[CB_B:.*]] = ttkernel.get_compile_time_arg_val(1)
// TTK-DAG: %[[CB_SCALE:.*]] = ttkernel.get_compile_time_arg_val(2)
// TTK-DAG: %[[CB_ACC:.*]] = ttkernel.get_compile_time_arg_val(3)
// TTK-DAG: %[[CB_OUT:.*]] = ttkernel.get_compile_time_arg_val(4)
// TTK:     ttkernel.tile_regs_acquire
// TTK:     ttkernel.copy_tile_init(%[[CB_SCALE]])
// TTK:     ttkernel.copy_tile(%[[CB_SCALE]], %[[C0]], {{.*}})
// TTK:     ttkernel.copy_tile_init(%[[CB_ACC]])
// TTK:     ttkernel.copy_tile(%[[CB_ACC]], %[[C0]], {{.*}})
// TTK:     ttkernel.mul_binary_tile({{.*}}, {{.*}}, %[[C0]])
// TTK:     ttkernel.matmul_block(%[[CB_A]], %[[CB_B]], %[[C0]], %[[C0]], %[[C0]]
// TTK:     ttkernel.tile_regs_commit
// TTK-NEXT: ttkernel.tile_regs_wait
// TTK-NEXT: ttkernel.pack_tile(%[[C0]], %[[CB_OUT]], %[[C0]]
// TTK-NEXT: ttkernel.tile_regs_release
// TTK-NOT: ttl.dst_index
func.func @scaled_acc_matmul_f32(
    %scale: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %acc: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %a: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %b: tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb4 = ttl.bind_cb {cb_index = 4, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a_attached = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b_attached = ttl.attach_cb %b, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %scale_attached = ttl.attach_cb %scale, %cb2 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %acc_attached = ttl.attach_cb %acc, %cb3 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb4 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %scaled = ttl.mul %scale_attached, %acc_attached : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %mm = ttl.matmul %a_attached, %b_attached : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %out = ttl.add %scaled, %mm : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.store %out, %reserve : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>
  func.return %out : tensor<1x1x!ttcore.tile<32x32, f32>>
}
