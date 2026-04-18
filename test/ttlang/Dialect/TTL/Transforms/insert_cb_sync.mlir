// Verifies ttl-insert-cb-sync: missing cb_push/cb_pop are inserted after
// the last transitive use of the CB data.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-insert-cb-sync))' --split-input-file | FileCheck %s

// Test 1: compute reserve without push, auto-insert after store.

// CHECK-LABEL: func.func @compute_reserve_no_push
// CHECK: %[[CB:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: ttl.cb_reserve %[[CB]]
// CHECK: ttl.store
// CHECK-NEXT: ttl.cb_push %[[CB]]
// CHECK-NOT: ttl.cb_push
// CHECK: return
func.func @compute_reserve_no_push(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %arg0, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}

// -----

// Test 2: compute wait without pop, auto-insert after add.

// CHECK-LABEL: func.func @compute_wait_no_pop
// CHECK: %[[CB:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: ttl.cb_wait %[[CB]]
// CHECK: ttl.attach_cb
// CHECK: ttl.add
// CHECK-NEXT: ttl.cb_pop %[[CB]]
// CHECK-NOT: ttl.cb_pop
// CHECK: return
func.func @compute_wait_no_pop(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %block = ttl.attach_cb %w, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.add %block, %arg0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}

// -----

// Test 3: DM thread, chase copy -> transfer_handle -> wait chain.

// CHECK-LABEL: func.func @dm_reserve_copy_chain
// CHECK: %[[CB:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: ttl.cb_reserve %[[CB]]
// CHECK: ttl.copy
// CHECK: ttl.wait
// CHECK-NEXT: ttl.cb_push %[[CB]]
// CHECK-NOT: ttl.cb_push
// CHECK: return
func.func @dm_reserve_copy_chain(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %slice = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>
  %tx = ttl.copy %slice, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
  ttl.wait %tx : !ttl.transfer_handle<read>
  func.return
}

// -----

// Test 4: explicit push preserved, no double-insert.

// CHECK-LABEL: func.func @explicit_push_preserved
// CHECK: ttl.cb_reserve
// CHECK: ttl.store
// CHECK-NEXT: ttl.cb_push
// CHECK-NOT: ttl.cb_push
// CHECK: return
func.func @explicit_push_preserved(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %arg0, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Test 5: mixed explicit and implicit across different CBs.

// CHECK-LABEL: func.func @mixed_explicit_implicit
// CHECK: %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
// CHECK: ttl.cb_reserve %[[CB0]]
// CHECK: ttl.store
// CHECK: ttl.cb_push %[[CB0]]
// CHECK: ttl.cb_reserve %[[CB1]]
// CHECK: ttl.store
// CHECK-NEXT: ttl.cb_push %[[CB1]]
// CHECK-NOT: ttl.cb_push
// CHECK: return
func.func @mixed_explicit_implicit(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb{cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %r0 = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %arg0, %r0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %r1 = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %arg0, %r1 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}

// -----

// Test 6: two waits on different CBs, both need auto-pop.

// CHECK-LABEL: func.func @multiple_waits
// CHECK: %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
// CHECK: ttl.cb_wait %[[CB0]]
// CHECK: ttl.cb_wait %[[CB1]]
// CHECK: ttl.add
// CHECK: ttl.cb_pop %[[CB1]]
// CHECK: ttl.cb_pop %[[CB0]]
// CHECK-NOT: ttl.cb_pop
// CHECK: return
func.func @multiple_waits(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb{cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w0 = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b0 = ttl.attach_cb %w0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %w1 = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b1 = ttl.attach_cb %w1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.add %b0, %b1 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}

// -----

// Test 7: same CB double wait, first explicit pop, second implicit.

// CHECK-LABEL: func.func @same_cb_double_wait_mixed
// CHECK: %[[CB:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: ttl.cb_wait %[[CB]]
// CHECK: ttl.add
// CHECK: ttl.cb_pop %[[CB]]
// CHECK: ttl.cb_wait %[[CB]]
// CHECK: ttl.add
// CHECK-NEXT: ttl.cb_pop %[[CB]]
// CHECK-NOT: ttl.cb_pop
// CHECK: return
func.func @same_cb_double_wait_mixed(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w0 = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b0 = ttl.attach_cb %w0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %r0 = ttl.add %b0, %arg0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w1 = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b1 = ttl.attach_cb %w1, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %r1 = ttl.add %b1, %arg0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}

// -----

// Test 8: wait + use inside scf.for loop body, pop auto-inserted in body.

// CHECK-LABEL: func.func @wait_inside_loop
// CHECK: %[[CB:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: scf.for
// CHECK:   ttl.cb_wait %[[CB]]
// CHECK:   ttl.add
// CHECK-NEXT: ttl.cb_pop %[[CB]]
// CHECK: }
// CHECK-NOT: ttl.cb_pop
// CHECK: return
func.func @wait_inside_loop(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %iv = %c0 to %c4 step %c1 {
    %w = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b = ttl.attach_cb %w, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %r = ttl.add %b, %arg0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  func.return
}

// -----

// Test 9: pop in one scf.if branch gets hoisted after the scf.if.

// CHECK-LABEL: func.func @pop_hoisted_from_branch
// CHECK: %[[CB:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: ttl.cb_wait %[[CB]]
// CHECK: scf.if
// CHECK:   ttl.add
// CHECK-NOT: ttl.cb_pop
// CHECK: } else {
// CHECK:   ttl.mul
// CHECK-NOT: ttl.cb_pop
// CHECK: }
// CHECK-NEXT: ttl.cb_pop %[[CB]]
// CHECK-NOT: ttl.cb_pop
// CHECK: return
func.func @pop_hoisted_from_branch(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %cond: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %w, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.if %cond {
    %r = ttl.add %b, %arg0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  } else {
    %r = ttl.mul %b, %arg0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  func.return
}

// -----

// Test 10: reserve with no uses at all. Push should go right after reserve.

// CHECK-LABEL: func.func @reserve_no_uses
// CHECK: %[[CB:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: ttl.cb_reserve %[[CB]]
// CHECK-NEXT: ttl.cb_push %[[CB]]
// CHECK-NOT: ttl.cb_push
// CHECK: return
func.func @reserve_no_uses()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}

// -----

// Test 11: attach_cb with dead result. Push after attach_cb.

// CHECK-LABEL: func.func @attach_cb_dead_result
// CHECK: %[[CB:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: ttl.cb_reserve %[[CB]]
// CHECK: ttl.attach_cb
// CHECK-NEXT: ttl.cb_push %[[CB]]
// CHECK-NOT: ttl.cb_push
// CHECK: return
func.func @attach_cb_dead_result()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %block = ttl.attach_cb %reserve, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}

// -----

// Test 12: two copies on same CB in DM thread. Push after last wait.

// CHECK-LABEL: func.func @two_copies_same_cb
// CHECK: %[[CB:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: ttl.cb_reserve %[[CB]]
// CHECK: ttl.copy
// CHECK: ttl.wait
// CHECK: ttl.copy
// CHECK: ttl.wait
// CHECK-NEXT: ttl.cb_push %[[CB]]
// CHECK-NOT: ttl.cb_push
// CHECK: return
func.func @two_copies_same_cb(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %slice0 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>
  %tx0 = ttl.copy %slice0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
  ttl.wait %tx0 : !ttl.transfer_handle<read>
  %slice1 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>
  %tx1 = ttl.copy %slice1, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
  ttl.wait %tx1 : !ttl.transfer_handle<read>
  func.return
}

// -----

// Test 13: pops in BOTH branches of scf.if. Both hoisted, single pop after.

// CHECK-LABEL: func.func @pops_both_branches
// CHECK: %[[CB:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: ttl.cb_wait %[[CB]]
// CHECK: scf.if
// CHECK:   ttl.add
// CHECK-NOT: ttl.cb_pop
// CHECK: } else {
// CHECK:   ttl.mul
// CHECK-NOT: ttl.cb_pop
// CHECK: }
// CHECK-NEXT: ttl.cb_pop %[[CB]]
// CHECK-NOT: ttl.cb_pop
// CHECK: return
func.func @pops_both_branches(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %cond: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %w, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.if %cond {
    %r = ttl.add %b, %arg0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  } else {
    %r = ttl.mul %b, %arg0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  }
  func.return
}

// -----

// Test 14: reserve outside loop, store inside. Push after the loop.

// CHECK-LABEL: func.func @reserve_outside_loop
// CHECK: %[[CB:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: ttl.cb_reserve %[[CB]]
// CHECK: scf.for
// CHECK:   ttl.store
// CHECK-NOT: ttl.cb_push
// CHECK: }
// CHECK-NEXT: ttl.cb_push %[[CB]]
// CHECK-NOT: ttl.cb_push
// CHECK: return
func.func @reserve_outside_loop(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %iv = %c0 to %c4 step %c1 {
    ttl.store %arg0, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  func.return
}

// -----

// Test 15: scf.if nested inside scf.for, pop inside if branch.
// Pop hoisted out of if, stays inside the for body.

// CHECK-LABEL: func.func @if_inside_for
// CHECK: %[[CB:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: scf.for
// CHECK:   ttl.cb_wait %[[CB]]
// CHECK:   scf.if
// CHECK:     ttl.add
// CHECK-NOT: ttl.cb_pop
// CHECK:   } else {
// CHECK:     ttl.mul
// CHECK-NOT: ttl.cb_pop
// CHECK:   }
// CHECK-NEXT: ttl.cb_pop %[[CB]]
// CHECK: }
// CHECK-NOT: ttl.cb_pop
// CHECK: return
func.func @if_inside_for(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %cond: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %iv = %c0 to %c4 step %c1 {
    %w = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b = ttl.attach_cb %w, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.if %cond {
      %r = ttl.add %b, %arg0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    } else {
      %r = ttl.mul %b, %arg0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
  }
  func.return
}

// -----

// Test 16: nested for loops, wait in inner loop.

// CHECK-LABEL: func.func @nested_for_loops
// CHECK: %[[CB:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     ttl.cb_wait %[[CB]]
// CHECK:     ttl.add
// CHECK-NEXT: ttl.cb_pop %[[CB]]
// CHECK:   }
// CHECK: }
// CHECK-NOT: ttl.cb_pop
// CHECK: return
func.func @nested_for_loops(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      %w = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %b = ttl.attach_cb %w, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %r = ttl.add %b, %arg0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
  }
  func.return
}

// -----

// Test 17: intra-thread reserve + store OUTSIDE the loop, cb_wait on the
// same CB INSIDE the loop body. Regression for #524: the outer push must
// be placed before the scf.for so the first loop iteration's wait sees
// the slot. The nested wait's pop lands inside the loop body after the
// last tensor use (ttl.add), as usual.

// CHECK-LABEL: func.func @reserve_before_loop_with_nested_wait
// CHECK: %[[CB:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: ttl.cb_reserve %[[CB]]
// CHECK: ttl.store
// CHECK-NEXT: ttl.cb_push %[[CB]]
// CHECK: scf.for
// CHECK:   ttl.cb_wait %[[CB]]
// CHECK:   ttl.add
// CHECK-NEXT: ttl.cb_pop %[[CB]]
// CHECK: }
// CHECK-NOT: ttl.cb_push
// CHECK-NOT: ttl.cb_pop
// CHECK: return
func.func @reserve_before_loop_with_nested_wait(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %arg0, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %iv = %c0 to %c4 step %c1 {
    %w = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b = ttl.attach_cb %w, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %r = ttl.add %b, %arg0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  func.return
}

// -----

// Test 18: outer cb_wait whose attached-CB tensor value is consumed
// inside a subsequent scf.for body. The pop must be placed AFTER the
// scf.for — the live interval of the waited slot extends through the
// loop because the value is still used on every iteration.

// CHECK-LABEL: func.func @wait_before_loop_use_inside_loop
// CHECK: %[[CB:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: ttl.cb_wait %[[CB]]
// CHECK: ttl.attach_cb
// CHECK: scf.for
// CHECK:   ttl.add
// CHECK-NOT: ttl.cb_pop
// CHECK: }
// CHECK-NEXT: ttl.cb_pop %[[CB]]
// CHECK-NOT: ttl.cb_pop
// CHECK: return
func.func @wait_before_loop_use_inside_loop(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %w, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %iv = %c0 to %c4 step %c1 {
    %r = ttl.add %b, %arg0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  func.return
}

// -----

// Test 19: DM thread reserve outside scf.for, with ttl.copy writing to
// the CB directly (not via attach_cb) INSIDE the loop body. The CB-use
// walk must pick up the copy's ancestor (the scf.for) as the live-
// interval endpoint and place the push after the loop.

// CHECK-LABEL: func.func @dm_reserve_copy_inside_loop
// CHECK: %[[CB:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: ttl.cb_reserve %[[CB]]
// CHECK: scf.for
// CHECK:   ttl.copy
// CHECK:   ttl.wait
// CHECK-NOT: ttl.cb_push
// CHECK: }
// CHECK-NEXT: ttl.cb_push %[[CB]]
// CHECK-NOT: ttl.cb_push
// CHECK: return
func.func @dm_reserve_copy_inside_loop(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.for %iv = %c0 to %c4 step %c1 {
    %slice = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>
    %tx = ttl.copy %slice, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %tx : !ttl.transfer_handle<read>
  }
  func.return
}

// -----

// Test 20: outer cb_reserve + store with a cb_wait on the same CB
// consuming the slot in only the ELSE branch of a subsequent scf.if
// (the then branch does nothing with this CB). The scf.if is a sibling
// region — not a descendant of the reserve's block in the structured-
// control-flow sense the bug originally exercised. Regardless, the
// push must land before scf.if so the else branch's wait is satisfied
// on the control-flow paths that reach it.

// CHECK-LABEL: func.func @reserve_then_if_else_branch_wait
// CHECK: %[[CB:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: ttl.cb_reserve %[[CB]]
// CHECK: ttl.store
// CHECK-NEXT: ttl.cb_push %[[CB]]
// CHECK: scf.if
// CHECK: } else {
// CHECK:   ttl.cb_wait %[[CB]]
// CHECK:   ttl.add
// CHECK-NEXT: ttl.cb_pop %[[CB]]
// CHECK: }
// CHECK-NOT: ttl.cb_push
// CHECK: return
func.func @reserve_then_if_else_branch_wait(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %cond: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %arg0, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.if %cond {
  } else {
    %w = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b = ttl.attach_cb %w, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %r = ttl.add %b, %arg0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  func.return
}
