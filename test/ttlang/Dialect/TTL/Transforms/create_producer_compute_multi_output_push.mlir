// Verify that producer `ComputeOp` creation preserves the operation order of
// releases when the release order differs from the output store order.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute))' | FileCheck %s

// CHECK-LABEL: func.func @add_stored_to_three_dfbs
// CHECK: %[[OUT_A:.*]] = ttl.bind_cb{cb_index = 2
// CHECK: %[[OUT_B:.*]] = ttl.bind_cb{cb_index = 3
// CHECK: %[[OUT_C:.*]] = ttl.bind_cb{cb_index = 4
// No producer release may precede the compute that writes the DFB slots.
// CHECK-NOT: ttl.cb_push
// CHECK: %{{.*}}:3 = ttl.compute
// CHECK-COUNT-3: ttl.tile_store
// CHECK: ttl.cb_push %[[OUT_B]]
// CHECK-NEXT: ttl.cb_push %[[OUT_A]]
// CHECK-NEXT: ttl.cb_push %[[OUT_C]]
func.func @add_stored_to_three_dfbs()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_outA = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_outB = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_outC = ttl.bind_cb {cb_index = 4, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %a_wait = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a = ttl.attach_cb %a_wait, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b_wait = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %b_wait, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %sum = ttl.add %a, %b : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %rA = ttl.cb_reserve %cb_outA : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %rA : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>

  %rB = ttl.cb_reserve %cb_outB : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %rB : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb_outB : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_push %cb_outA : <[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %rC = ttl.cb_reserve %cb_outC : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %rC : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb_outC : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}
