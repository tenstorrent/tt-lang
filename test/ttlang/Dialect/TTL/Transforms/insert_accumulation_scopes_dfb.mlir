// Verifies ttl-insert-accumulation-scopes{kind=dfb} wraps user-written
// accumulating stores in ttl.accumulation_scope and records the initial mode.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-insert-accumulation-scopes{kind=dfb}))' --split-input-file | FileCheck %s

// A loop whose first iteration creates the output value uses overwrite mode.

// CHECK-LABEL: func.func @dfb_overwrite_scope
// CHECK: ttl.accumulation_scope outs(%{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
// CHECK: ^bb0(%{{.*}}: tensor<1x1x!ttcore.tile<32x32, bf16>>):
// CHECK:   scf.for
// CHECK:     ttl.store
// CHECK:   ttl.yield %{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>
// CHECK: } initial_modes([overwrite])
func.func @dfb_overwrite_scope(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %input = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.for %iv = %c0 to %c4 step %c1 {
    ttl.store %input, %reserve {accumulate} : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  func.return %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

// A preceding overwrite store to the same view seeds the L1 value.

// CHECK-LABEL: func.func @dfb_accumulate_existing_scope
// CHECK: ttl.store
// CHECK: ttl.accumulation_scope outs(%{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
// CHECK: ^bb0(%{{.*}}: tensor<1x1x!ttcore.tile<32x32, bf16>>):
// CHECK:   scf.for
// CHECK:     ttl.store
// CHECK:   ttl.yield %{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>
// CHECK: } initial_modes([accumulate_existing])
func.func @dfb_accumulate_existing_scope(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %arg1: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %init = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %arg1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %init, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.for %iv = %c0 to %c4 step %c1 {
    ttl.store %input, %reserve {accumulate} : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  func.return %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

// A seed store in an enclosing block is visible to nested accumulation loops.

// CHECK-LABEL: func.func @dfb_nested_seed_accumulate_existing_scope
// CHECK: ttl.store
// CHECK: scf.for
// CHECK:   ttl.accumulation_scope outs(%{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
// CHECK:     scf.for
// CHECK:       ttl.store
// CHECK:     ttl.yield
// CHECK:   } initial_modes([accumulate_existing])
func.func @dfb_nested_seed_accumulate_existing_scope(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %arg1: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %init = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %arg1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %init, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.for %outer = %c0 to %c4 step %c1 {
    scf.for %inner = %c0 to %c4 step %c1 {
      ttl.store %input, %reserve {accumulate} : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
  }
  func.return %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

// A seed store in a guarded acquire region is visible to a later same-guard
// accumulation loop.

// CHECK-LABEL: func.func @guarded_seed_accumulate_existing_scope
// CHECK: %[[COND:.*]] = arith.constant true
// CHECK: scf.if %[[COND]]
// CHECK: %[[RESERVE:.*]] = ttl.cb_reserve
// CHECK:   ttl.store
// CHECK: ttl.accumulation_scope outs(%[[RESERVE]]
// CHECK:   scf.for
// CHECK:     ttl.store
// CHECK: } initial_modes([accumulate_existing])
func.func @guarded_seed_accumulate_existing_scope(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %arg1: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %condition = arith.constant true
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %init = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %arg1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.if %condition {
    %reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %init, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.for %iv = %c0 to %c4 step %c1 {
      ttl.store %input, %reserve {accumulate} : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
  }
  func.return
}
