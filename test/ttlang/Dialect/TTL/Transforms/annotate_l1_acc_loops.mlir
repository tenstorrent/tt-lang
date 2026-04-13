// Verifies ttl-annotate-l1-acc-loops: user-written scf.for loops that store
// to a CB reserved outside the loop are annotated with ttl.l1_acc_loop.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-annotate-l1-acc-loops))' --split-input-file | FileCheck %s

// Loop storing to an externally reserved CB should be annotated.

// CHECK-LABEL: func.func @external_reserve
// CHECK: scf.for
// CHECK: } {ttl.l1_acc_loop}
func.func @external_reserve(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %arg1: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.for %iv = %c0 to %c4 step %c1 {
    %mm = ttl.matmul %a, %b : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %mm, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  func.return %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

// Loop where cb_reserve is INSIDE the loop should NOT be annotated.

// CHECK-LABEL: func.func @internal_reserve
// CHECK: scf.for
// CHECK-NOT: ttl.l1_acc_loop
// CHECK: }
func.func @internal_reserve(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %arg1: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %arg0) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %mm = ttl.matmul %a, %b : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %mm, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield %mm : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  func.return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

// Loops already annotated with compiler-generated attributes should be skipped.

// CHECK-LABEL: func.func @skip_tile_loop
// CHECK: scf.for
// CHECK: } {ttl.tile_loop_stride
// CHECK-NOT: ttl.l1_acc_loop
func.func @skip_tile_loop(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.for %iv = %c0 to %c4 step %c1 {
    ttl.store %arg0, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  } {ttl.tile_loop_stride = array<i64: 1>}
  func.return %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

// Loops already annotated with ttl.reduction_loop should be skipped.

// CHECK-LABEL: func.func @skip_reduction_loop
// CHECK: scf.for
// CHECK: } {ttl.reduction_loop
// CHECK-NOT: ttl.l1_acc_loop
func.func @skip_reduction_loop(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.for %iv = %c0 to %c4 step %c1 {
    ttl.store %arg0, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  } {ttl.reduction_loop}
  func.return %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

// Loop without any store should NOT be annotated.

// CHECK-LABEL: func.func @no_store
// CHECK: scf.for
// CHECK-NOT: ttl.l1_acc_loop
// CHECK: }
func.func @no_store(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.for %iv = %c0 to %c4 step %c1 {
    // No ttl.store in the loop body.
  }
  func.return %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>
}
