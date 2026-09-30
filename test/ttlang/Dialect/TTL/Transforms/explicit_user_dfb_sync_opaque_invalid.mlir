// Opaque consumers are accepted with explicit user synchronization; default inference still diagnoses ambiguous access order.
// RUN: ttlang-opt %s --ttl-insert-cb-sync --split-input-file --verify-diagnostics
// RUN: ttlang-opt %s --ttl-insert-cb-sync='sync-user-dfbs=false' --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @conditional_external_consumer
// CHECK: scf.if
// CHECK-NEXT: %[[SLOT:.*]] = ttl.cb_reserve %[[DFB:[a-zA-Z0-9_]+]] :
// CHECK-NEXT: ttl.store {{.*}}, %[[SLOT]]
// CHECK-NEXT: ttl.cb_push %[[DFB]]
// CHECK-NEXT: ttl.opaque_call "consume" (%[[DFB]])
// CHECK-NOT: ttl.cb_push
// CHECK-NOT: ttl.cb_pop
// CHECK: return
func.func @conditional_external_consumer(
    %input_value: tensor<1x1x!ttcore.tile<32x32, bf16>>, %condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %user_dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  scf.if %condition {
    %slot = ttl.cb_reserve %user_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %input_value, %slot : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-error @below {{guarded local dataflow buffer push must follow all uses in its acquiring region}}
    ttl.cb_push %user_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.opaque_call "consume"(%user_dfb) {header = "consume.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> ()
  }
  return
}
