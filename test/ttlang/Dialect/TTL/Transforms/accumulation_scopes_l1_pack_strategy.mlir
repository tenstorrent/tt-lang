// Verifies tensor accumulation scope formation honors the requested strategy.
//
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-form-accumulation-scopes{strategy=l1-pack}, ttl-lower-accumulation-scopes{strategy=l1-pack}))' | FileCheck %s --check-prefix=L1
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-form-accumulation-scopes{strategy=auto}, ttl-lower-accumulation-scopes{strategy=auto}))' | FileCheck %s --check-prefix=AUTO

// Purpose: L1 packer accumulation forms for a recurrence that is not legal for
// DST lowering because the contribution release is left for auto sync.
// L1-LABEL: func.func @l1_pack_formable_recurrence
// L1: %[[RESERVE:.*]] = ttl.cb_reserve
// L1: ttl.store %{{.*}}, %[[RESERVE]]
// L1: scf.for
// L1: ttl.store %{{.*}}, %[[RESERVE]] {accumulate}
// L1: } {ttl.l1_acc_initial = 1 : i32, ttl.l1_acc_loop, ttl.l1_acc_scope_id = 0 : i64}
// L1-NOT: ttl.accumulation_scope
// AUTO-LABEL: func.func @l1_pack_formable_recurrence
// AUTO: %[[RESERVE:.*]] = ttl.cb_reserve
// AUTO: ttl.store %{{.*}}, %[[RESERVE]]
// AUTO: scf.for
// AUTO: ttl.store %{{.*}}, %[[RESERVE]] {accumulate}
// AUTO: } {ttl.l1_acc_initial = 1 : i32, ttl.l1_acc_loop, ttl.l1_acc_scope_id = 0 : i64}
// AUTO-NOT: ttl.accumulation_scope
func.func @l1_pack_formable_recurrence() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %init_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %contrib_cb = ttl.bind_cb {cb_index = 1, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %out_cb = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %init_wait = ttl.cb_wait %init_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init = ttl.attach_cb %init_wait, %init_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out = ttl.cb_reserve %out_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %result = scf.for %iv = %c0 to %c3 step %c1
      iter_args(%acc = %init)
      -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %contrib_wait = ttl.cb_wait %contrib_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %contrib = ttl.attach_cb %contrib_wait, %contrib_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %next = ttl.add %acc, %contrib : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield %next : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  ttl.store %result, %out : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}
