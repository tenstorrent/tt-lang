// Verifies auto strategy falls back to L1 packer accumulation when a tensor
// accumulation scope is not DST-compatible.
//
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-lower-accumulation-scopes{kind=tensor strategy=auto}))' | FileCheck %s

// The contribution dataflow buffer cannot hold the coalesced three-tile wait
// required by DST, so auto selects the L1 packer strategy.
// CHECK-LABEL: func.func @auto_l1_fallback
// CHECK: %[[RESERVE:.*]] = ttl.cb_reserve
// CHECK: ttl.store %{{.*}}, %[[RESERVE]]
// CHECK: scf.for {{.*}} {
// CHECK: ttl.store %{{.*}}, %[[RESERVE]] {accumulate}
// CHECK: } {ttl.l1_acc_initial = 1 : i32, ttl.l1_acc_loop, ttl.l1_acc_scope_id = 0 : i64}
// CHECK-NOT: ttl.accumulation_scope
// CHECK-NOT: ttl.compute
func.func @auto_l1_fallback() {
  %cb_init = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_delta = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_out = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %init_wait = ttl.cb_wait %cb_init : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init = ttl.attach_cb %init_wait, %cb_init : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  ttl.accumulation_scope outs(%reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%state: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    %loop = scf.for %iter = %c0 to %c3 step %c1 iter_args(%acc = %state) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
      %delta_wait = ttl.cb_wait %cb_delta : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %delta = ttl.attach_cb %delta_wait, %cb_delta : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %sum = ttl.add %acc, %delta : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      scf.yield %sum : tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    ttl.store %loop, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield %loop : tensor<1x1x!ttcore.tile<32x32, bf16>>
  } initial_modes([init])
  func.return
}
