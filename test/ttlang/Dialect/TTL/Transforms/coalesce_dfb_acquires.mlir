// Verifies ttl-coalesce-dfb-acquires: strictly-consecutive same-DFB
// acquires collapse into a single multi-tile acquire plus per-block
// extract_slice views, with N matching releases collapsing into one
// carrying num_tiles=N*k. See issue #556.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-coalesce-dfb-acquires))' --split-input-file | FileCheck %s
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-coalesce-dfb-acquires,ttl-coalesce-dfb-acquires))' --split-input-file | FileCheck %s

// Test 1: three consecutive cb_wait + three pops -> one cb_wait{num_tiles=3}
// + three extract_slices + one cb_pop{num_tiles=3}.

// CHECK-LABEL: func.func @three_waits_consumer
// CHECK: %[[CBIN:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: %[[CBOUT:.+]] = ttl.bind_cb{cb_index = 1
// CHECK: %[[GROUP:.+]] = ttl.cb_wait %[[CBIN]] {num_tiles = 3 : i64}
// CHECK-SAME: tensor<1x3x!ttcore.tile<32x32, bf16>>
// CHECK-NEXT: %[[S0:.+]] = tensor.extract_slice %[[GROUP]][0, 0] [1, 1] [1, 1]
// CHECK-NEXT: ttl.attach_cb %[[S0]]
// CHECK-NEXT: %[[S1:.+]] = tensor.extract_slice %[[GROUP]][0, 1] [1, 1] [1, 1]
// CHECK-NEXT: ttl.attach_cb %[[S1]]
// CHECK-NEXT: %[[S2:.+]] = tensor.extract_slice %[[GROUP]][0, 2] [1, 1] [1, 1]
// CHECK-NEXT: ttl.attach_cb %[[S2]]
// CHECK: ttl.cb_pop %[[CBIN]] {num_tiles = 3 : i64}
// CHECK-NOT: ttl.cb_wait
// CHECK-NOT: ttl.cb_pop
// CHECK: return
func.func @three_waits_consumer()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb_in = ttl.bind_cb{cb_index = 0, block_count = 3} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
  %cb_out = ttl.bind_cb{cb_index = 1, block_count = 3} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
  %w0 = ttl.cb_wait %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a0 = ttl.attach_cb %w0, %cb_in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %w1 = ttl.cb_wait %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a1 = ttl.attach_cb %w1, %cb_in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %w2 = ttl.cb_wait %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a2 = ttl.attach_cb %w2, %cb_in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %r0 = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %a0, %r0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
  ttl.cb_push %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
  %r1 = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %a1, %r1 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
  ttl.cb_push %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
  %r2 = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %a2, %r2 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
  ttl.cb_push %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
  func.return
}

// -----

// Test 2: producer-side analog. Three consecutive cb_reserve + three pushes
// collapse to one cb_reserve{num_tiles=3} + three extract_slices routed to
// stores + one cb_push{num_tiles=3}.

// CHECK-LABEL: func.func @three_reserves_producer
// CHECK: %[[CB:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: %[[GROUP:.+]] = ttl.cb_reserve %[[CB]] {num_tiles = 3 : i64}
// CHECK-SAME: tensor<1x3x!ttcore.tile<32x32, bf16>>
// CHECK-NEXT: tensor.extract_slice %[[GROUP]][0, 0] [1, 1] [1, 1]
// CHECK-NEXT: tensor.extract_slice %[[GROUP]][0, 1] [1, 1] [1, 1]
// CHECK-NEXT: tensor.extract_slice %[[GROUP]][0, 2] [1, 1] [1, 1]
// CHECK: ttl.cb_push %[[CB]] {num_tiles = 3 : i64}
// CHECK-NOT: ttl.cb_reserve
// CHECK-NOT: ttl.cb_push
// CHECK: return
func.func @three_reserves_producer(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb{cb_index = 0, block_count = 3} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
  %r0 = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %r1 = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %r2 = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %arg0, %r0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
  ttl.store %arg0, %r1 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
  ttl.store %arg0, %r2 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
  func.return
}

// -----

// Test 3: four consecutive waits inside scf.for body coalesce per iteration.

// CHECK-LABEL: func.func @four_waits_in_loop
// CHECK: scf.for
// CHECK: ttl.cb_wait {{.*}} {num_tiles = 4 : i64}
// CHECK-SAME: tensor<1x4x!ttcore.tile<32x32, bf16>>
// CHECK-COUNT-4: tensor.extract_slice
// CHECK: ttl.cb_pop {{.*}} {num_tiles = 4 : i64}
// CHECK: }
// CHECK-NOT: ttl.cb_wait
// CHECK-NOT: ttl.cb_pop
// CHECK: return
func.func @four_waits_in_loop()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cb_in = ttl.bind_cb{cb_index = 0, block_count = 12} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 12>
  %cb_out = ttl.bind_cb{cb_index = 1, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  scf.for %i = %c0 to %c3 step %c1 {
    %w0 = ttl.cb_wait %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 12> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %a0 = ttl.attach_cb %w0, %cb_in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 12>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %w1 = ttl.cb_wait %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 12> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %a1 = ttl.attach_cb %w1, %cb_in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 12>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %w2 = ttl.cb_wait %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 12> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %a2 = ttl.attach_cb %w2, %cb_in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 12>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %w3 = ttl.cb_wait %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 12> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %a3 = ttl.attach_cb %w3, %cb_in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 12>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %r0 = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %a0, %r0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 12>
    ttl.cb_push %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %r1 = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %a1, %r1 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 12>
    ttl.cb_push %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %r2 = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %a2, %r2 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 12>
    ttl.cb_push %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %r3 = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %a3, %r3 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 12>
    ttl.cb_push %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  }
  func.return
}

// -----

// Test 4 (negative): wait, use, wait, use — non-consecutive acquires.
// The use of %w0 between the waits breaks the run; nothing coalesces.

// CHECK-LABEL: func.func @interleaved_consume_not_coalesced
// CHECK-NOT: num_tiles
// CHECK-NOT: tensor.extract_slice
func.func @interleaved_consume_not_coalesced()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb_in = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_out = ttl.bind_cb{cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w0 = ttl.cb_wait %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a0 = ttl.attach_cb %w0, %cb_in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %r0 = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %a0, %r0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_push %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w1 = ttl.cb_wait %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a1 = ttl.attach_cb %w1, %cb_in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %r1 = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %a1, %r1 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_push %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Test 5 (negative): waits on different CBs alternating — neither group
// is "strictly consecutive on the same DFB". No coalescing.

// CHECK-LABEL: func.func @alternating_cbs_not_coalesced
// CHECK-NOT: num_tiles
// CHECK-NOT: tensor.extract_slice
//
// Note: this test verifies the SINGLE-acquire-per-CB pattern is left
// alone. Multi-acquire-per-CB interleaved across CBs (matmul-style) IS
// coalesced and is covered by the next test.
func.func @alternating_cbs_not_coalesced()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb_a = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_b = ttl.bind_cb{cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_out = ttl.bind_cb{cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %wa = ttl.cb_wait %cb_a : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %wb = ttl.cb_wait %cb_b : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %aa = ttl.attach_cb %wa, %cb_a : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %ab = ttl.attach_cb %wb, %cb_b : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %r0 = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %aa, %r0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %cb_a : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb_b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_push %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Test 6 (negative): cb_reserve already carrying num_tiles (e.g. set by
// ttl-subblock-compute-for-dst) is left untouched.

// CHECK-LABEL: func.func @existing_num_tiles_untouched
// CHECK: ttl.cb_reserve %{{.*}} {num_tiles = 2 : i64}
// CHECK-NOT: tensor.extract_slice
// CHECK: return
func.func @existing_num_tiles_untouched(
    %arg0: tensor<1x2x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %r = ttl.cb_reserve %cb {num_tiles = 2 : i64} : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.store %arg0, %r : tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb {num_tiles = 2 : i64} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Test 7: single cb_wait without a sibling on the same DFB stays
// unchanged (no group of >= 2 to coalesce).

// CHECK-LABEL: func.func @single_wait_unchanged
// CHECK: ttl.cb_wait
// CHECK-NOT: num_tiles
// CHECK: ttl.cb_pop
// CHECK-NOT: num_tiles
// CHECK: return
func.func @single_wait_unchanged()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb_in = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_out = ttl.bind_cb{cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w = ttl.cb_wait %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a = ttl.attach_cb %w, %cb_in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %r = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %a, %r : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_push %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Test 8: matmul-style pattern. Two waits on cb_a interleaved with two
// waits on cb_b. Each CB independently has a coalescable group; the
// other-CB acquire between same-CB acquires does not touch our CB or our
// group's results, so it does not break the run.

// CHECK-LABEL: func.func @matmul_style_two_cb_interleaved
// CHECK: %[[CBA:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: %[[CBB:.+]] = ttl.bind_cb{cb_index = 1
// CHECK: %[[GA:.+]] = ttl.cb_wait %[[CBA]] {num_tiles = 2 : i64}
// CHECK-SAME: tensor<1x2x!ttcore.tile<32x32, bf16>>
// CHECK: %[[GB:.+]] = ttl.cb_wait %[[CBB]] {num_tiles = 2 : i64}
// CHECK-SAME: tensor<1x2x!ttcore.tile<32x32, bf16>>
// CHECK-DAG: ttl.cb_pop %[[CBA]] {num_tiles = 2 : i64}
// CHECK-DAG: ttl.cb_pop %[[CBB]] {num_tiles = 2 : i64}
// CHECK-NOT: ttl.cb_wait
// CHECK-NOT: ttl.cb_pop
// CHECK: return
func.func @matmul_style_two_cb_interleaved()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb_a = ttl.bind_cb{cb_index = 0, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %cb_b = ttl.bind_cb{cb_index = 1, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %a1 = ttl.cb_wait %cb_a : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %aa1 = ttl.attach_cb %a1, %cb_a : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b1 = ttl.cb_wait %cb_b : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %ab1 = ttl.attach_cb %b1, %cb_b : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a2 = ttl.cb_wait %cb_a : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %aa2 = ttl.attach_cb %a2, %cb_a : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b2 = ttl.cb_wait %cb_b : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %ab2 = ttl.attach_cb %b2, %cb_b : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %cb_a : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  ttl.cb_pop %cb_b : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  ttl.cb_pop %cb_a : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  ttl.cb_pop %cb_b : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  func.return
}

// -----

// Test 9: a region-bearing op (scf.if) between two same-DFB acquires
// terminates the candidate group, even when the region's body is empty.
// `mayReleaseDFB` treats any op with regions as opaque because the body
// might contain a release on the DFB.

// CHECK-LABEL: func.func @region_op_between_acquires_not_coalesced
// CHECK-NOT: num_tiles
// CHECK-NOT: tensor.extract_slice
func.func @region_op_between_acquires_not_coalesced(%cond: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb_in = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_out = ttl.bind_cb{cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w0 = ttl.cb_wait %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a0 = ttl.attach_cb %w0, %cb_in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.if %cond {
  }
  %w1 = ttl.cb_wait %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a1 = ttl.attach_cb %w1, %cb_in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %r0 = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %a0, %r0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_push %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %r1 = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %a1, %r1 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_push %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Test 10: stray `attach_cb` on an unrelated tensor between two same-DFB
// acquires does NOT terminate the group. `attach_cb` is an SSA identity
// erased at lowering and cannot release the DFB; `mayReleaseDFB`
// allow-lists it explicitly.

// CHECK-LABEL: func.func @attach_cb_unrelated_tensor_between_waits
// CHECK: %[[CBIN:.+]] = ttl.bind_cb{cb_index = 0
// CHECK: ttl.cb_wait %[[CBIN]] {num_tiles = 2 : i64}
// CHECK-SAME: tensor<1x2x!ttcore.tile<32x32, bf16>>
// CHECK-COUNT-2: tensor.extract_slice
// CHECK: ttl.cb_pop %[[CBIN]] {num_tiles = 2 : i64}
// CHECK-NOT: ttl.cb_wait
// CHECK-NOT: ttl.cb_pop
// CHECK: return
func.func @attach_cb_unrelated_tensor_between_waits(
    %unrelated: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb_in = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_out = ttl.bind_cb{cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w0 = ttl.cb_wait %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a0 = ttl.attach_cb %w0, %cb_in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %stray = ttl.attach_cb %unrelated, %cb_in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %w1 = ttl.cb_wait %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a1 = ttl.attach_cb %w1, %cb_in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %r0 = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %a0, %r0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_push %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %r1 = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %a1, %r1 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_push %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}
