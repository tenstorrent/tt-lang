// Regression test for fused broadcast + elementwise through full pipeline.
//
// The Python pattern `a_bcast * b_bcast + c_bcast` with different-shaped
// inputs (col-bcast Nx1, row-bcast 1xM, scalar 1x1) triggers two bugs:
//
// Bug 1 (fixed): buildFusedCompute used identity indexing maps for all inputs,
//   causing tensor.extract_slice out-of-bounds when TilingInterface tried to
//   index a 2x1 tensor at column 1.
//
// Bug 2 (fixed): computeBcastShapeExpansionIndex generated arith.divui/remui
//   to decompose linearized index into row/col. ConvertTTKernelToEmitC cannot
//   lower these ops. Fix: compute row/col components directly from constant
//   strides (compile-time integer division).
//
// This test runs the full pipeline and verifies:
// - Pipeline completes without crash (from Bug 1)
// - No arith.divui or arith.remui in output (from Bug 2)
// - Correct bcast/mul/add operations appear

// Full pipeline test: input is pre-convert-ttl-to-compute MLIR
// RUN: ttlang-opt %s --split-input-file \
// RUN:   -pass-pipeline='builtin.module(func.func(convert-ttl-to-compute, ttl-set-compute-kernel-config, ttl-assign-dst, ttl-subblock-compute-for-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @fused_bcast_mul_add
// CHECK: ttkernel.tile_regs_acquire
//
// Must not contain divui/remui (Bug 2):
// CHECK-NOT: arith.divui
// CHECK-NOT: arith.remui
//
// Col bcast group (a: 2x1 -> 2x2):
// CHECK: ttkernel.unary_bcast_init({{.*}}, <col>)
// CHECK: ttkernel.unary_bcast({{.*}}, <col>)
//
// Row bcast group (b: 1x2 -> 2x2):
// CHECK: ttkernel.unary_bcast_init({{.*}}, <row>)
// CHECK: ttkernel.unary_bcast({{.*}}, <row>)
//
// Mul group:
// CHECK: ttkernel.mul_binary_tile_init
// CHECK: ttkernel.mul_binary_tile
//
// Scalar bcast group (c: 1x1 -> 2x2):
// CHECK: ttkernel.unary_bcast_init({{.*}}, <scalar>)
// CHECK: ttkernel.unary_bcast({{.*}}, <scalar>)
//
// Add group:
// CHECK: ttkernel.add_binary_tile_init
// CHECK: ttkernel.add_binary_tile
//
// no divui/remui anywhere
// CHECK-NOT: arith.divui
// CHECK-NOT: arith.remui
// CHECK: ttkernel.tile_regs_commit
func.func @fused_bcast_mul_add()
    attributes {ttl.base_cta_index = 4 : i32, ttl.crta_indices = [],
                ttl.kernel_thread = #ttkernel.thread<compute>} {

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>

  // a: col vector (2x1), c: scalar (1x1) - waited outside loop
  %a_ready = ttl.cb_wait %cb0 : <[2, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %a_cb = ttl.attach_cb %a_ready, %cb0 : (tensor<2x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>

  %c_ready = ttl.cb_wait %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c_cb = ttl.attach_cb %c_ready, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  // b: row vector (1x2)
  %b_ready = ttl.cb_wait %cb1 : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %b_cb = ttl.attach_cb %b_ready, %cb1 : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x2x!ttcore.tile<32x32, bf16>>

  // output (2x2)
  %out = ttl.cb_reserve %cb3 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %out_cb = ttl.attach_cb %out, %cb3 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  // Fused compute: bcast(a) * bcast(b) + bcast(c)
  // Col bcast: 2x1 -> 2x2 (BcastType::Col = 1)
  %a_bcast = ttl.bcast %a_cb, %out_cb 1 : i32 : (tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  // Row bcast: 1x2 -> 2x2 (BcastType::Row = 2)
  %b_bcast = ttl.bcast %b_cb, %out_cb 2 : i32 : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  // Scalar bcast: 1x1 -> 2x2 (BcastType::Scalar = 3)
  %c_bcast = ttl.bcast %c_cb, %out_cb 3 : i32 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %prod = ttl.mul %a_bcast, %b_bcast : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %sum = ttl.add %prod, %c_bcast : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  ttl.store %sum, %out_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>

  %result_cb = ttl.attach_cb %sum, %cb3 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb3 : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb1 : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : <[2, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>

  return
}
