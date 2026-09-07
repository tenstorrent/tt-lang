// A planned reciprocal output store requires storage for its reduction input.
// RUN: ttlang-opt %s --ttl-insert-intermediate-dfbs | FileCheck %s
// CHECK-LABEL: func.func @softmax
// CHECK: %[[REDUCED:.*]] = ttl.reduce
// CHECK: %[[RESERVED:.*]] = ttl.cb_reserve %[[REDUCTION_STORAGE:.*]] :
// CHECK-NEXT: ttl.store %[[REDUCED]], %[[RESERVED]]
// CHECK-NEXT: %[[WAITED:.*]] = ttl.cb_wait %[[REDUCTION_STORAGE]]
// CHECK-NEXT: %[[ATTACHED:.*]] = ttl.attach_cb %[[WAITED]], %[[REDUCTION_STORAGE]]
// CHECK-NEXT: %{{.*}} = ttl.recip %[[ATTACHED]]
func.func @softmax() attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 2 : i32} {
  %input = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %output = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %input_block = ttl.cb_wait %input : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %attached = ttl.attach_cb %input_block, %input : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output_block = ttl.cb_reserve %output : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %numerator = ttl.exp %attached : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.fill 1.0 : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %denominator = ttl.reduce %numerator, %scaler 0 : i32 [1] : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %inverse = ttl.recip %denominator : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %broadcast = ttl.block.broadcast %inverse dims = [1], shape = [1, 1] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %normalized = ttl.mul %numerator, %broadcast : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %normalized, %output_block : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %output : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  ttl.cb_pop %input : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  return
}
