// Verifies that convert-ttl-to-compute preserves cb_push ordering when a
// single CB is written across multiple compute blocks (fixes #519) and
// when a single reserved slot is written by multiple stores absorbed
// into one compute block (regression guard for PR #523 / #524).

// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute))' | FileCheck %s

// Test 1: two back-to-back reduce+store sequences sharing red_dfb (cb2).
// Each store has a cb_push immediately after. After lowering, each
// cb_push must remain after its own ttl.compute block; they must not be
// grouped together after the second compute (#519).

// CHECK-LABEL: func.func @two_reduces_shared_dfb
// CHECK:       ttl.cb_reserve %[[RED:.*]] :
//
// First compute: reduce -> tile_store, then push for pass 1.
// CHECK:       ttl.compute
// CHECK:       ttl.cb_push %[[RED]]
//
// Intervening wait/reserve/store/pop on other CBs.
// CHECK:       ttl.cb_pop
// CHECK:       ttl.cb_wait %[[RED]]
//
// Second compute: reduce -> tile_store, then push for pass 2.
// CHECK:       ttl.cb_reserve %[[RED]]
// CHECK:       ttl.compute
// CHECK:       ttl.cb_push %[[RED]]
// CHECK:       ttl.cb_pop
// CHECK:       ttl.cb_wait %[[RED]]
func.func @two_reduces_shared_dfb()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %acc = ttl.bind_cb{cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %red = ttl.bind_cb{cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %sc  = ttl.bind_cb{cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %x   = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %s0 = ttl.cb_wait %sc : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %s  = ttl.attach_cb %s0, %sc : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  // Pass 1: reduce into red.
  %x0_t = ttl.cb_wait %x : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %x0   = ttl.attach_cb %x0_t, %x : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %r1_t = ttl.cb_reserve %red : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %r1   = ttl.attach_cb %r1_t, %red : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %v1   = ttl.reduce %x0, %s 0 : i32 [1] : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %v1, %r1_t : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %red : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop  %x   : <[1, 1], !ttcore.tile<32x32, bf16>, 2>

  // Consume red, produce acc (intervening ops).
  %w1_t = ttl.cb_wait %red : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %w1   = ttl.attach_cb %w1_t, %red : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a1_t = ttl.cb_reserve %acc : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a1   = ttl.attach_cb %a1_t, %acc : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %w1, %a1_t : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %acc : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop  %red : <[1, 1], !ttcore.tile<32x32, bf16>, 2>

  // Pass 2: reduce into red again.
  %x1_t = ttl.cb_wait %x : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %x1   = ttl.attach_cb %x1_t, %x : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %r2_t = ttl.cb_reserve %red : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %r2   = ttl.attach_cb %r2_t, %red : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %v2   = ttl.reduce %x1, %s 0 : i32 [1] : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %v2, %r2_t : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %red : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop  %x   : <[1, 1], !ttcore.tile<32x32, bf16>, 2>

  // Consume red + acc.
  %w2_t = ttl.cb_wait %red : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %w2   = ttl.attach_cb %w2_t, %red : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %red : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %sc  : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  return
}

// -----

// Test 2: one reserved slot written by TWO stores, followed by a SINGLE
// cb_push at end of the scope (the layernorm `with reserve as v:
// v.store(init); for _: v += ...; v.store(final)` pattern). Both stores
// fuse into one ttl.compute. The cb_push is already positioned after
// the generated compute and must NOT be relocated forward — doing so
// would commit the slot before later stores pack their data. Regression
// guard for the #523 fix over-reaching into this pattern (#524).

// CHECK-LABEL: func.func @multi_store_single_reserve
// CHECK:       %[[OUT:.*]] = ttl.bind_cb{cb_index = 1
// CHECK:       ttl.cb_reserve %[[OUT]]
// CHECK:       ttl.compute
// CHECK:         ttl.tile_store
// CHECK:         ttl.tile_store
// CHECK:         ttl.yield
// CHECK:       ttl.cb_push %[[OUT]]
// CHECK-NOT:   ttl.cb_push
// CHECK:       return
func.func @multi_store_single_reserve()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %x   = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %out = ttl.bind_cb{cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %a_t = ttl.cb_wait %x : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a   = ttl.attach_cb %a_t, %x : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %r_t = ttl.cb_reserve %out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %r   = ttl.attach_cb %r_t, %out : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  // First store into the reserved slot.
  %v1 = ttl.add %a, %a : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %v1, %r_t : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>

  // Second store into the SAME reserved slot — mirrors a user-written
  // `with reserve: ... multi-store ...` block_expr scope.
  %v2 = ttl.mul %a, %a : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %v2, %r_t : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>

  // Single cb_push at end of the scope, already after where the pass
  // will materialize the ttl.compute. Must not be moved forward past
  // the later store.
  ttl.cb_push %out : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop  %x   : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Test 3: two reserves on two different CBs with stores interleaved,
// cb_push ops in REVERSE order (push for y, then push for x). Both
// pushes are already past the generated compute ops, so the push-move
// guard (`isBeforeInBlock(computeOp)`) correctly leaves them in place.
// Each CB is still pushed exactly once and each push follows the
// corresponding pack, so the emitted ordering is functionally correct
// even though the pushes are not "tightly" paired with their compute
// op.

// CHECK-LABEL: func.func @interleaved_pushes_wrong_order
// CHECK:       %[[IN:.*]] = ttl.bind_cb{cb_index = 0
// CHECK:       %[[X:.*]] = ttl.bind_cb{cb_index = 1
// CHECK:       %[[Y:.*]] = ttl.bind_cb{cb_index = 2
//
// Two separate compute ops (one per store), followed by the two pushes
// in their original order.
// CHECK:       ttl.compute
// CHECK:         ttl.tile_add
// CHECK:       ttl.compute
// CHECK:         ttl.tile_mul
// CHECK:       ttl.cb_push %[[Y]]
// CHECK:       ttl.cb_push %[[X]]
// CHECK:       ttl.cb_pop %[[IN]]
// CHECK:       return
func.func @interleaved_pushes_wrong_order()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %in = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %x  = ttl.bind_cb{cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %y  = ttl.bind_cb{cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %a_t = ttl.cb_wait %in : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a   = ttl.attach_cb %a_t, %in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %rx_t = ttl.cb_reserve %x : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %rx   = ttl.attach_cb %rx_t, %x : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %ry_t = ttl.cb_reserve %y : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %ry   = ttl.attach_cb %ry_t, %y : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %vx = ttl.add %a, %a : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %vx, %rx_t : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  %vy = ttl.mul %a, %a : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %vy, %ry_t : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>

  // Pushes in the OPPOSITE order of the stores — the walk must still
  // pair each store with the push whose CB matches its view.
  ttl.cb_push %y : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_push %x : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop  %in : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}
