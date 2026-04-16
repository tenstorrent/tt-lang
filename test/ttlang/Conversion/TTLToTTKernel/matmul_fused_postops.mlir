// Regression test for #476: elementwise ops fused with matmul in a single
// store must not be dropped. Verifies that scale * (A @ B) + bias produces
// matmul_block followed by mul_binary_tile and add_binary_tile.

// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(convert-ttl-to-compute, ttl-assign-dst{enable-fpu-binary-ops=0}, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @matmul_scale_bias
// CHECK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[CB_A:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[CB_B:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK-DAG: %[[CB_SC:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK-DAG: %[[CB_BI:.*]] = ttkernel.get_compile_time_arg_val(3)
// CHECK-DAG: %[[CB_OUT:.*]] = ttkernel.get_compile_time_arg_val(4)
//
// Matmul block.
// CHECK:      ttkernel.tile_regs_acquire
// CHECK:      ttkernel.matmul_block(%[[CB_A]], %[[CB_B]],
//
// Scale: copy_tile loads scale from CB into DST, mul_binary_tile applies it.
// CHECK:      ttkernel.copy_tile_init(%[[CB_SC]])
// CHECK-NEXT: ttkernel.copy_tile(%[[CB_SC]], %[[C0]], %[[C1]])
// CHECK-NEXT: ttkernel.mul_binary_tile_init()
// CHECK-NEXT: ttkernel.mul_binary_tile(%[[C1]], %[[C0]], %[[C0]])
//
// Bias: copy_tile loads bias from CB into DST, add_binary_tile applies it.
// CHECK:      ttkernel.copy_tile_init(%[[CB_BI]])
// CHECK-NEXT: ttkernel.copy_tile(%[[CB_BI]], %[[C0]], %[[C1]])
// CHECK-NEXT: ttkernel.add_binary_tile_init()
// CHECK-NEXT: ttkernel.add_binary_tile(%[[C0]], %[[C1]], %[[C0]])
//
// Pack result.
// CHECK:      ttkernel.tile_regs_commit
// CHECK-NEXT: ttkernel.tile_regs_wait
// CHECK-NEXT: ttkernel.pack_tile(%[[C0]], %[[CB_OUT]], %[[C0]]
// CHECK-NEXT: ttkernel.tile_regs_release

func.func @matmul_scale_bias(
    %sc: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %a: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %b: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %bi: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb3 = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb4 = ttl.bind_cb {cb_index = 4, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %sc_a = ttl.attach_cb %sc, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a_a = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b_a = ttl.attach_cb %b, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %bi_a = ttl.attach_cb %bi, %cb3 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a_a, %b_a : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %mul = ttl.mul %sc_a, %mm : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %add = ttl.add %mul, %bi_a : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %add, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return %add : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

// Multi-output matmul: scale * (A @ B) stored to two output CBs.
// Verifies that generateMatmulCompute handles multiple output views and
// produces the correct number of replacement tensors for the compute op.

// CHECK-LABEL: func.func @matmul_scale_dual_output
// CHECK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[CB_A:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[CB_B:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK-DAG: %[[CB_SC:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK-DAG: %[[CB_OUT0:.*]] = ttkernel.get_compile_time_arg_val(3)
// CHECK-DAG: %[[CB_OUT1:.*]] = ttkernel.get_compile_time_arg_val(4)
//
// Matmul block.
// CHECK:      ttkernel.tile_regs_acquire
// CHECK:      ttkernel.matmul_block(%[[CB_A]], %[[CB_B]],
//
// Scale post-op.
// CHECK:      ttkernel.copy_tile_init(%[[CB_SC]])
// CHECK-NEXT: ttkernel.copy_tile(%[[CB_SC]], %[[C0]], %[[C1]])
// CHECK-NEXT: ttkernel.mul_binary_tile_init()
// CHECK-NEXT: ttkernel.mul_binary_tile(%[[C1]], %[[C0]], %[[C0]])
//
// Pack to both output CBs.
// CHECK:      ttkernel.tile_regs_commit
// CHECK-NEXT: ttkernel.tile_regs_wait
// CHECK-NEXT: ttkernel.pack_tile(%[[C0]], %[[CB_OUT1]], %[[C0]]
// CHECK-NEXT: ttkernel.pack_tile(%[[C0]], %[[CB_OUT0]], %[[C0]]
// CHECK-NEXT: ttkernel.tile_regs_release

func.func @matmul_scale_dual_output(
    %sc: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %a: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %b: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb3 = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb4 = ttl.bind_cb {cb_index = 4, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %sc_a = ttl.attach_cb %sc, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a_a = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b_a = ttl.attach_cb %b, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve0 = ttl.cb_reserve %cb3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve1 = ttl.cb_reserve %cb4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a_a, %b_a : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %mul = ttl.mul %sc_a, %mm : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %mul, %reserve0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %mul, %reserve1 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return %mul : tensor<1x1x!ttcore.tile<32x32, bf16>>
}
