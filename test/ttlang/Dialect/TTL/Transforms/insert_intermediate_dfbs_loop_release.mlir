// Verifies that a DFB-attached value used inside a loop is materialized before
// its original DFB is released.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-materialize-loop-state,ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute,ttl-auto-sync))' | FileCheck %s

// CHECK-LABEL: func.func @preserve_attached_value_across_loop
// CHECK: %[[DELTA_DFB:.*]] = ttl.bind_cb{{.*}}cb_index = 1
// CHECK: %[[LOOP_STATE_DFB:.*]] = ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK: %[[PRESERVED_DFB:.*]] = ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK: %[[DELTA_WAIT:.*]] = ttl.cb_wait %[[DELTA_DFB]]
// CHECK: %[[DELTA:.*]] = ttl.attach_cb %[[DELTA_WAIT]], %[[DELTA_DFB]]
// CHECK: %[[PRESERVED_RESERVE:.*]] = ttl.cb_reserve %[[PRESERVED_DFB]]
// CHECK: ttl.compute ins(%[[DELTA]]
// CHECK: ttl.cb_push %[[PRESERVED_DFB]]
// CHECK: %[[PRESERVED_WAIT:.*]] = ttl.cb_wait %[[PRESERVED_DFB]]
// CHECK: %[[PRESERVED:.*]] = ttl.attach_cb %[[PRESERVED_WAIT]], %[[PRESERVED_DFB]]
// CHECK: ttl.cb_pop %[[DELTA_DFB]]
// CHECK: scf.for
// CHECK: ttl.compute ins({{.*}}%[[PRESERVED]]
// CHECK: ttl.cb_pop %[[PRESERVED_DFB]]
func.func @preserve_attached_value_across_loop()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %initial_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %delta_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %delta_wait = ttl.cb_wait %delta_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %delta = ttl.attach_cb %delta_wait, %delta_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %delta_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %initial_wait = ttl.cb_wait %initial_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %initial = ttl.attach_cb %initial_wait, %initial_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %zero = arith.constant 0 : index
  %limit = arith.constant 4 : index
  %one = arith.constant 1 : index
  %result = scf.for %iteration = %zero to %limit step %one
      iter_args(%accumulator = %initial)
      -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
    %updated = ttl.add %accumulator, %delta
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield %updated : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}
