// Tests for scalar-constant reduce scaler handling in ttl-insert-intermediate-dfbs.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-insert-intermediate-dfbs))' | FileCheck %s

// -----

// Two reduces consuming the same FillOp scaler share one compiler-allocated
// scaler DFB. Each reduce result is wrapped in its own ttl.mul_unary_const
// (the post-reduce scaler multiply) and gets its own intermediate DFB.

// CHECK-LABEL: func.func @shared_scalar_scaler_dedup
// CHECK-COUNT-1: ttl.fill 1.000000e+00
// CHECK-NOT: ttl.fill
// CHECK-COUNT-2: ttl.mul_unary_const {{.*}}, 5.000000e-01
// CHECK-NOT: ttl.mul_unary_const
// CHECK-NOT: ttl.reduce_scalar_multiplier
func.func @shared_scalar_scaler_dedup()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb_in_a = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_in_b = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a_wait = ttl.cb_wait %cb_in_a : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a = ttl.attach_cb %a_wait, %cb_in_a : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b_wait = ttl.cb_wait %cb_in_b : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %b_wait, %cb_in_b : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %fill = ttl.fill 5.000000e-01 : tensor<1x1x!ttcore.tile<32x32, bf16>>

  %r1 = ttl.reduce %a, %fill 0 : i32 [1] : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %r2 = ttl.reduce %b, %fill 0 : i32 [1] : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// After the structural rewrite, the only remaining `ttl.fill` is the
// internally-materialized 1.0 neutral fill. The original 0.5 fill must
// be erased because its only consumer was the rewritten reduce scaler.

// CHECK-LABEL: func.func @orphan_fill_erased
// CHECK-NOT: ttl.fill 5.000000e-01
// CHECK: ttl.fill 1.000000e+00
// CHECK-NOT: ttl.fill 5.000000e-01
// CHECK: ttl.mul_unary_const {{.*}}, 5.000000e-01
func.func @orphan_fill_erased()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb_in = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a_wait = ttl.cb_wait %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a = ttl.attach_cb %a_wait, %cb_in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %fill = ttl.fill 5.000000e-01 : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %r = ttl.reduce %a, %fill 0 : i32 [1] : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// FillOp shared between a reduce scaler AND a non-reduce consumer must NOT
// be erased — the non-reduce consumer still needs the actual fill value.

// CHECK-LABEL: func.func @fill_kept_when_other_users
// CHECK: ttl.fill 5.000000e-01
// CHECK: ttl.mul_unary_const {{.*}}, 5.000000e-01
func.func @fill_kept_when_other_users()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb_in = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_out = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a_wait = ttl.cb_wait %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a = ttl.attach_cb %a_wait, %cb_in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %fill = ttl.fill 5.000000e-01 : tensor<1x1x!ttcore.tile<32x32, bf16>>

  // Non-reduce consumer of the fill: a store to a regular output CB.
  %reserve = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %fill, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>

  %r = ttl.reduce %a, %fill 0 : i32 [1] : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}
