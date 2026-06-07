// Verifies ttl-insert-cb-sync leaves resident acquires alone. A cb_reserve /
// cb_wait carrying the `resident` attr (emitted by a DFB's store() / read())
// is an in-place access: the pass inserts no cb_push / cb_pop, so the slot's
// write/read pointer never advances. Non-resident acquires still get their
// releases. Running the pass twice is idempotent.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-insert-cb-sync))' --split-input-file | FileCheck %s
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-insert-cb-sync,ttl-insert-cb-sync))' --split-input-file | FileCheck %s

// A resident reserve (store side) gets no cb_push.

// CHECK-LABEL: func.func @resident_store
// CHECK: ttl.cb_reserve %{{.+}} {resident}
// CHECK: ttl.store
// CHECK-NOT: ttl.cb_push
// CHECK: return
func.func @resident_store(%arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %r = ttl.cb_reserve %cb0 {resident} : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %arg0, %r : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}

// -----

// A resident wait (read side) gets no cb_pop.

// CHECK-LABEL: func.func @resident_read
// CHECK: ttl.cb_wait %{{.+}} {resident}
// CHECK: ttl.add
// CHECK-NOT: ttl.cb_pop
// CHECK: return
func.func @resident_read(%arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %w = ttl.cb_wait %cb0 {resident} : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %w, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %res = ttl.add %b, %arg0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}

// -----

// Mixing is allowed: a non-resident reserve on the same buffer still gets a
// cb_push, while the resident reserve does not.

// CHECK-LABEL: func.func @mixed_store
// CHECK: ttl.cb_reserve %[[CB:.+]] {resident}
// CHECK: ttl.store
// CHECK-NOT: ttl.cb_push
// CHECK: ttl.cb_reserve %[[CB]] :
// CHECK: ttl.store
// CHECK: ttl.cb_push %[[CB]]
func.func @mixed_store(%arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %r0 = ttl.cb_reserve %cb0 {resident} : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %arg0, %r0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  %r1 = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %arg0, %r1 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}
