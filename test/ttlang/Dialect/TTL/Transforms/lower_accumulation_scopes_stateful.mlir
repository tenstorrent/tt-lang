// Verifies stateful multi-output tensor scopes lower through explicit stores
// and then use compiler-allocated DFB state for remaining tensor iter_args.
//
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-lower-accumulation-scopes{kind=tensor strategy=auto}))' | FileCheck %s --check-prefix=LOWER
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-lower-accumulation-scopes{kind=tensor strategy=auto},ttl-materialize-loop-state))' | FileCheck %s --check-prefix=MATERIALIZE

// Auto strategy converts yielded single-output state to a final store and leaves
// the tensor loop state for ttl-materialize-loop-state.
// LOWER-LABEL: func.func @stateful_single_output
// LOWER: %[[OUT:.*]] = ttl.cb_reserve
// LOWER: %[[LOOP:.*]] = scf.for
// LOWER: %[[NEXT:.*]] = ttl.relu
// LOWER: ttl.store %[[LOOP]], %[[OUT]]
// LOWER-NOT: ttl.accumulation_scope
//
// MATERIALIZE-LABEL: func.func @stateful_single_output
// MATERIALIZE: ttl.compiler_allocated
// MATERIALIZE: scf.for
// MATERIALIZE-NOT: iter_args
// MATERIALIZE: ttl.relu
// MATERIALIZE: ttl.store
func.func @stateful_single_output() {
  %cb_init = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_out = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %init_wait = ttl.cb_wait %cb_init : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init = ttl.attach_cb %init_wait, %cb_init : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  ttl.accumulation_scope outs(%reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%state: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    %loop = scf.for %iter = %c0 to %c4 step %c1 iter_args(%acc = %state) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
      %next = ttl.relu %acc : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      scf.yield %next : tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    ttl.yield %loop : tensor<1x1x!ttcore.tile<32x32, bf16>>
  } initial_modes([init])
  func.return
}

// Auto strategy converts the yielded multi-output state to final stores and
// leaves the tensor loop state for ttl-materialize-loop-state.
// LOWER-LABEL: func.func @stateful_dependent_accumulators
// LOWER: %[[OUT0:.*]] = ttl.cb_reserve
// LOWER: %[[OUT1:.*]] = ttl.cb_reserve
// LOWER: %[[LOOP:.*]]:2 = scf.for {{.*}} iter_args
// LOWER: %[[NEXT0:.*]] = ttl.add
// LOWER: %[[NEXT1:.*]] = ttl.add %{{.*}}, %[[NEXT0]]
// LOWER: ttl.store %[[LOOP]]#0, %[[OUT0]]
// LOWER: ttl.store %[[LOOP]]#1, %[[OUT1]]
// LOWER-NOT: ttl.accumulation_scope
//
// Remaining tensor loop state receives one compiler-allocated DFB per carried
// tensor value.
// MATERIALIZE-LABEL: func.func @stateful_dependent_accumulators
// MATERIALIZE-COUNT-2: ttl.compiler_allocated
// MATERIALIZE-NOT: iter_args
// MATERIALIZE: ttl.add
// MATERIALIZE: ttl.add
// MATERIALIZE: ttl.store
// MATERIALIZE: ttl.store
func.func @stateful_dependent_accumulators() {
  %cb_init0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_init1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_delta = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_out0 = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_out1 = ttl.bind_cb {cb_index = 4, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %init0_wait = ttl.cb_wait %cb_init0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init0 = ttl.attach_cb %init0_wait, %cb_init0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init1_wait = ttl.cb_wait %cb_init1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init1 = ttl.attach_cb %init1_wait, %cb_init1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve0 = ttl.cb_reserve %cb_out0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve1 = ttl.cb_reserve %cb_out1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  ttl.accumulation_scope outs(%reserve0, %reserve1 : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                                     tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init0, %init1 : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                              tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%state0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
       %state1: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    %loop:2 = scf.for %iter = %c0 to %c4 step %c1 iter_args(%acc0 = %state0, %acc1 = %state1) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) {
      %delta_wait = ttl.cb_wait %cb_delta : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %delta = ttl.attach_cb %delta_wait, %cb_delta : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %next0 = ttl.add %acc0, %delta : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %next1 = ttl.add %acc1, %next0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      scf.yield %next0, %next1 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    ttl.yield %loop#0, %loop#1 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  } initial_modes([init, init])
  func.return
}
