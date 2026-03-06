// RUN: ttlang-opt %s --split-input-file \
// RUN:   -pass-pipeline='builtin.module(func.func(convert-ttl-to-compute, ttl-set-compute-kernel-config, ttl-assign-dst, ttl-subblock-compute-for-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s

// Purpose: Test that the scheduler groups bcast ops by type (COL/ROW/SCALAR)
// and the consolidation pass emits exactly one init per group.
//
// Without init-affinity scheduling, the interleaved per-tile pattern
//   col_init, col(0), row_init, row(0), col_init, col(1), row_init, row(1)...
// requires N*K inits (N tiles * K bcast types). With grouping:
//   col_init, col(0), col(1), ..., row_init, row(0), row(1), ...
// requires only K inits (one per bcast type).
//
// The test uses a fused bcast(a) * bcast(b) + bcast(c) pattern with
// inputs a(2x1), b(1x2), c(1x1) -> output(2x2) = 4 output tiles.

// CHECK-LABEL: func.func @bcast_init_grouping
// CHECK:       ttkernel.tile_regs_acquire
//
// Group 1: COL bcasts - one init, then all 4 COL bcast ops
// CHECK:       ttkernel.unary_bcast_init({{.*}}, <col>)
// CHECK-NEXT:  ttkernel.unary_bcast({{.*}}, <col>)
// CHECK-NEXT:  ttkernel.unary_bcast({{.*}}, <col>)
// CHECK-NEXT:  ttkernel.unary_bcast({{.*}}, <col>)
// CHECK-NEXT:  ttkernel.unary_bcast({{.*}}, <col>)
//
// Group 2: ROW bcasts - one init, then all 4 ROW bcast ops
// CHECK-NEXT:  ttkernel.unary_bcast_init({{.*}}, <row>)
// CHECK-NEXT:  ttkernel.unary_bcast({{.*}}, <row>)
// CHECK-NEXT:  ttkernel.unary_bcast({{.*}}, <row>)
// CHECK-NEXT:  ttkernel.unary_bcast({{.*}}, <row>)
// CHECK-NEXT:  ttkernel.unary_bcast({{.*}}, <row>)
//
// Group 3: MUL - one init, then all 4 mul ops
// CHECK-NEXT:  ttkernel.mul_binary_tile_init
// CHECK-NEXT:  ttkernel.mul_binary_tile
// CHECK-NEXT:  ttkernel.mul_binary_tile
// CHECK-NEXT:  ttkernel.mul_binary_tile
// CHECK-NEXT:  ttkernel.mul_binary_tile
//
// Group 4: SCALAR bcasts - one init, then all 4 scalar bcast ops
// CHECK-NEXT:  ttkernel.unary_bcast_init({{.*}}, <scalar>)
// CHECK-NEXT:  ttkernel.unary_bcast({{.*}}, <scalar>)
// CHECK-NEXT:  ttkernel.unary_bcast({{.*}}, <scalar>)
// CHECK-NEXT:  ttkernel.unary_bcast({{.*}}, <scalar>)
// CHECK-NEXT:  ttkernel.unary_bcast({{.*}}, <scalar>)
//
// Group 5: ADD - one init, then all 4 add ops
// CHECK-NEXT:  ttkernel.add_binary_tile_init
// CHECK-NEXT:  ttkernel.add_binary_tile
// CHECK-NEXT:  ttkernel.add_binary_tile
// CHECK-NEXT:  ttkernel.add_binary_tile
// CHECK-NEXT:  ttkernel.add_binary_tile
//
// CHECK:       ttkernel.tile_regs_commit
func.func @bcast_init_grouping()
    attributes {ttl.base_cta_index = 4 : i32, ttl.crta_indices = [],
                ttl.kernel_thread = #ttkernel.thread<compute>} {

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>

  %a_ready = ttl.cb_wait %cb0 : <[2, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %a_cb = ttl.attach_cb %a_ready, %cb0 : (tensor<2x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %c_ready = ttl.cb_wait %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c_cb = ttl.attach_cb %c_ready, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b_ready = ttl.cb_wait %cb1 : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %b_cb = ttl.attach_cb %b_ready, %cb1 : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %out = ttl.cb_reserve %cb3 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %out_cb = ttl.attach_cb %out, %cb3 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %a_bcast = ttl.bcast %a_cb, %out_cb 1 : i32 : (tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %b_bcast = ttl.bcast %b_cb, %out_cb 2 : i32 : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %c_bcast = ttl.bcast %c_cb, %out_cb 3 : i32 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %prod = ttl.mul %a_bcast, %b_bcast : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %sum = ttl.add %prod, %c_bcast : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %out : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>

  %result_cb = ttl.attach_cb %sum, %cb3 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb3 : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb1 : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : <[2, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>

  return
}
