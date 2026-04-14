// Verifies ttl-insert-cb-sync: missing cb_push/cb_pop are inserted after
// the last transitive use of the CB data.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-insert-cb-sync))' --split-input-file | FileCheck %s

// --- Test 1: compute thread, reserve without push ---
// The pass should insert cb_push after the store (last use of reserve view).

// CHECK-LABEL: func.func @compute_reserve_no_push
// CHECK: %[[CB:.+]] = ttl.bind_cb {cb_index = 0
// CHECK: %[[R:.+]] = ttl.cb_reserve %[[CB]]
// CHECK: ttl.store
// CHECK-NEXT: ttl.cb_push %[[CB]]
// CHECK: return
func.func @compute_reserve_no_push(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %arg0, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}

// -----

// --- Test 2: compute thread, wait without pop ---
// The pass should insert cb_pop after the last use of the waited block.

// CHECK-LABEL: func.func @compute_wait_no_pop
// CHECK: %[[CB:.+]] = ttl.bind_cb {cb_index = 0
// CHECK: %[[W:.+]] = ttl.cb_wait %[[CB]]
// CHECK: %[[A:.+]] = ttl.attach_cb %[[W]], %[[CB]]
// CHECK: ttl.add %[[A]]
// CHECK-NEXT: ttl.cb_pop %[[CB]]
// CHECK: return
func.func @compute_wait_no_pop(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %block = ttl.attach_cb %w, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.add %block, %arg0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}

// -----

// --- Test 3: DM thread, reserve + copy + wait chain ---
// The pass should chase copy -> transfer_handle -> wait and insert push
// after ttl.wait (not after copy).

// CHECK-LABEL: func.func @dm_reserve_copy_chain
// CHECK: %[[CB:.+]] = ttl.bind_cb {cb_index = 0
// CHECK: ttl.cb_reserve %[[CB]]
// CHECK: ttl.copy
// CHECK: ttl.wait
// CHECK-NEXT: ttl.cb_push %[[CB]]
// CHECK: return
func.func @dm_reserve_copy_chain(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %slice = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>
  %tx = ttl.copy %slice, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
  ttl.wait %tx : !ttl.transfer_handle<read>
  func.return
}

// -----

// --- Test 4: explicit push/pop should be preserved (no double-insert) ---

// CHECK-LABEL: func.func @explicit_push_preserved
// CHECK: ttl.cb_reserve
// CHECK: ttl.store
// CHECK-NEXT: ttl.cb_push
// CHECK-NOT: ttl.cb_push
// CHECK: return
func.func @explicit_push_preserved(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %arg0, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// --- Test 5: mixed explicit and implicit ---
// First CB has explicit push, second has no push (should be auto-inserted).

// CHECK-LABEL: func.func @mixed_explicit_implicit
// CHECK: %[[CB0:.+]] = ttl.bind_cb {cb_index = 0
// CHECK: %[[CB1:.+]] = ttl.bind_cb {cb_index = 1
// CHECK: ttl.cb_reserve %[[CB0]]
// CHECK: ttl.store
// CHECK: ttl.cb_push %[[CB0]]
// CHECK: ttl.cb_reserve %[[CB1]]
// CHECK: ttl.store
// CHECK-NEXT: ttl.cb_push %[[CB1]]
// CHECK: return
func.func @mixed_explicit_implicit(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %r0 = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %arg0, %r0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %r1 = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %arg0, %r1 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}

// -----

// --- Test 6: multiple waits, pops needed for each ---

// CHECK-LABEL: func.func @multiple_waits
// CHECK: %[[CB0:.+]] = ttl.bind_cb {cb_index = 0
// CHECK: %[[CB1:.+]] = ttl.bind_cb {cb_index = 1
// CHECK: ttl.cb_wait %[[CB0]]
// CHECK: ttl.attach_cb
// CHECK: ttl.cb_wait %[[CB1]]
// CHECK: ttl.attach_cb
// CHECK: ttl.add
// CHECK: ttl.cb_pop %[[CB1]]
// CHECK: ttl.cb_pop %[[CB0]]
// CHECK: return
func.func @multiple_waits(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w0 = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b0 = ttl.attach_cb %w0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %w1 = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b1 = ttl.attach_cb %w1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.add %b0, %b1 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}
