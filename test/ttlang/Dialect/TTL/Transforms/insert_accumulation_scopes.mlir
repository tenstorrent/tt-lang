// Verifies ttl-insert-accumulation-scopes wraps matched tensor recurrences in a
// semantic accumulation region without selecting a storage strategy.
//
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-insert-accumulation-scopes{kind=tensor}))' --split-input-file | FileCheck %s
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-insert-accumulation-scopes{kind=tensor}), func.func(ttl-insert-accumulation-scopes{kind=tensor}))' --split-input-file | FileCheck %s

// Tensor recurrence with a final store inserts one init accumulation
// scope. The output reserve is moved before the loop so the output slot spans
// all accumulation iterations.
// CHECK-LABEL: func.func @tensor_recurrence_scope
func.func @tensor_recurrence_scope(
    %init: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %iter = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %contribution = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sum = ttl.add %acc, %contribution : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield %sum : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  %reserve = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %attached = ttl.attach_cb %reserve, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %loop, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}
// CHECK: %[[RESERVE:.*]] = ttl.cb_reserve %{{.*}} :
// CHECK-NEXT: ttl.accumulation_scope outs(%[[RESERVE]] : tensor<1x1x!ttcore.tile<32x32, bf16>>) inits(%{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
// CHECK-NEXT: ^bb0(%[[STATE:.*]]: tensor<1x1x!ttcore.tile<32x32, bf16>>):
// CHECK-NEXT:   %[[LOOP:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[STATE]])
// CHECK:        ttl.add
// CHECK:        scf.yield
// CHECK-NEXT:   }
// CHECK-NEXT:   ttl.store %[[LOOP]], %[[RESERVE]]
// CHECK-NEXT:   ttl.yield %[[LOOP]] : tensor<1x1x!ttcore.tile<32x32, bf16>>
// CHECK-NEXT: } initial_modes([init])
// CHECK-NEXT: return
// CHECK-NOT: ttl.attach_cb

// -----

// Mixed loop-carried state is not wrapped yet because strategy lowering does
// not own the required non-accumulation state materialization.
// CHECK-LABEL: func.func @mixed_tensor_state_not_inserted
func.func @mixed_tensor_state_not_inserted(
    %init0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %init1: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %loop:2 = scf.for %iter = %c0 to %c4 step %c1 iter_args(%acc = %init0, %state = %init1) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %contribution = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sum = ttl.add %acc, %contribution : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %next_state = ttl.relu %state : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield %sum, %next_state : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  %reserve = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %loop#0, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}
// CHECK-NOT: ttl.accumulation_scope
// CHECK: scf.for
// CHECK: ttl.store

// -----

// Intervening side effects between the loop and final store keep the original
// structure. Insertion only wraps ranges whose operation order is explicit.
// CHECK-LABEL: func.func @intervening_store_not_inserted
func.func @intervening_store_not_inserted(
    %init: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %iter = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %contribution = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sum = ttl.add %acc, %contribution : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield %sum : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  %other_reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %init, %other_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %loop, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}
// CHECK-NOT: ttl.accumulation_scope
// CHECK: ttl.store
// CHECK: ttl.store

// -----

// A tensor recurrence loop with a loop-local store is not wrapped. The scope
// would not own that store's side effect.
// CHECK-LABEL: func.func @loop_local_store_not_inserted
func.func @loop_local_store_not_inserted(
    %init: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %other_reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %loop = scf.for %iter = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %contribution = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sum = ttl.add %acc, %contribution : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %init, %other_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield %sum : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  %reserve = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %loop, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}
// CHECK-NOT: ttl.accumulation_scope
// CHECK: scf.for
// CHECK: ttl.store
