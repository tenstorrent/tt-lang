// RUN: ttlang-opt %s --split-input-file -ttl-insert-pipenet-active-guards | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -ttl-insert-pipenet-active-guards -ttl-insert-pipenet-active-guards | FileCheck %s --check-prefix=IDEMPOTENT

// Verifies the active-set guard pass:
//   * Wraps every kernel-thread function in scf.if when any ttl.create_pipe
//     exists in the module.
//   * Skips functions when no pipes are present.
//   * Skips empty bodies that contain only a terminator.
//   * Is idempotent: running twice still produces a single guard per func.
//   * Coalesces rectangles fully contained in another (e.g. loopback pipes
//     where the source unit cell sits inside the destination range).

// Single multicast pipe: src=(0,0), dst range x in [0,3], y=0.
// The source unit cell (0,1)x(0,1) is contained in the destination
// rectangle (0,4)x(0,1), so coalescing drops the source rect and the
// predicate has a single conjunction (no arith.ori).

// CHECK-LABEL: func.func @dm_thread_single_pipe
// CHECK: ttl.core_x : index
// CHECK: ttl.core_y : index
// CHECK: arith.cmpi sge
// CHECK: arith.cmpi slt
// CHECK-NOT: arith.ori
// CHECK: scf.if {{.*}} {
// CHECK:   ttl.create_pipe
// CHECK:   ttl.if_src
// CHECK: }
// CHECK: return

// IDEMPOTENT-LABEL: func.func @dm_thread_single_pipe
// IDEMPOTENT: scf.if
// IDEMPOTENT-NOT: scf.if
// IDEMPOTENT: return

func.func @dm_thread_single_pipe() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %p = ttl.create_pipe src(0, 0) dst(0, 0) to(3, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(3, 0) net 0>
  ttl.if_src %p : !ttl.pipe<src(0, 0) dst(0, 0) to(3, 0) net 0> {
  }
  func.return
}

// Compute thread in the same module is also wrapped, even with no direct
// pipe reference.

// CHECK-LABEL: func.func @compute_thread_with_module_pipe
// CHECK: ttl.core_x
// CHECK: ttl.core_y
// CHECK: scf.if
// CHECK: return

// IDEMPOTENT-LABEL: func.func @compute_thread_with_module_pipe
// IDEMPOTENT: scf.if
// IDEMPOTENT-NOT: scf.if
// IDEMPOTENT: return

func.func @compute_thread_with_module_pipe() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Multi-pipe case: two pipes whose sources are each contained in their
// own destination ranges. Coalescing drops both source rects, leaving
// two destination rectangles combined by a single arith.ori.

// CHECK-LABEL: func.func @dm_thread_multi_pipe
// CHECK: ttl.core_x
// CHECK: ttl.core_y
// CHECK: arith.andi
// CHECK: arith.ori
// CHECK-NOT: arith.ori
// CHECK: scf.if
// CHECK: return

func.func @dm_thread_multi_pipe() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %p0 = ttl.create_pipe src(0, 0) dst(0, 0) to(3, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(3, 0) net 0>
  %p1 = ttl.create_pipe src(2, 1) dst(2, 1) to(2, 4) net 1
      : !ttl.pipe<src(2, 1) dst(2, 1) to(2, 4) net 1>
  ttl.if_src %p0 : !ttl.pipe<src(0, 0) dst(0, 0) to(3, 0) net 0> {
  }
  ttl.if_dst %p1 : !ttl.pipe<src(2, 1) dst(2, 1) to(2, 4) net 1> {
  }
  func.return
}

// -----

// No-pipe case: kernel-thread function with no ttl.create_pipe anywhere in
// the module is left untouched.

// CHECK-LABEL: func.func @compute_thread_no_pipes
// CHECK-NOT: scf.if
// CHECK-NOT: ttl.core_x
// CHECK-NOT: ttl.core_y
// CHECK: ttl.bind_cb
// CHECK: return

func.func @compute_thread_no_pipes() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Empty-body case: kernel-thread with only a terminator should be skipped
// (no scf.if, no core_x/core_y) even if pipes exist elsewhere in the module.

// CHECK-LABEL: func.func @dm_thread_empty_body
// CHECK-NEXT: return
// CHECK-NOT: scf.if
// CHECK-LABEL: func.func @dm_thread_carries_pipe

func.func @dm_thread_empty_body() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  func.return
}

func.func @dm_thread_carries_pipe() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  ttl.if_src %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
  }
  func.return
}

// -----

// Functions without ttl.kernel_thread are not touched even if pipes exist.

// CHECK-LABEL: func.func @host_helper
// CHECK-NOT: scf.if
// CHECK-NOT: ttl.core_x
// CHECK: return
// CHECK-LABEL: func.func @dm_thread_in_same_module

func.func @host_helper() {
  func.return
}

func.func @dm_thread_in_same_module() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  ttl.if_src %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
  }
  func.return
}

// -----

// Unicast pipe: dst_start == dst_end, so the destination rectangle is a
// unit cell. Combined with the source unit cell, the predicate is two
// disjoint single-core checks.

// CHECK-LABEL: func.func @unicast_pipe
// Source rectangle constants: src=(0,0) -> [0,1) x [0,1)
// CHECK: arith.constant 0 : index
// CHECK: arith.constant 1 : index
// CHECK: arith.constant 0 : index
// CHECK: arith.constant 1 : index
// Destination rectangle constants: dst=(2,3) to (2,3) -> [2,3) x [3,4)
// CHECK: arith.constant 2 : index
// CHECK: arith.constant 3 : index
// CHECK: arith.constant 3 : index
// CHECK: arith.constant 4 : index
// CHECK: arith.ori
// CHECK: scf.if
// CHECK: return

func.func @unicast_pipe() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %p = ttl.create_pipe src(0, 0) dst(2, 3) to(2, 3) net 0
      : !ttl.pipe<src(0, 0) dst(2, 3) to(2, 3) net 0>
  ttl.if_src %p : !ttl.pipe<src(0, 0) dst(2, 3) to(2, 3) net 0> {
  }
  func.return
}

// -----

// Inverted destination range: dst_start > dst_end on x. The pass must
// normalize via min/max so the rectangle is [0, 4) x [0, 1). The
// source (3, 0) lies inside the normalized destination, so coalescing
// drops the source rect and only the destination predicate remains.

// CHECK-LABEL: func.func @inverted_dst_range
// CHECK: arith.constant 0 : index
// CHECK: arith.constant 4 : index
// CHECK: arith.constant 0 : index
// CHECK: arith.constant 1 : index
// CHECK-NOT: arith.ori
// CHECK: scf.if
// CHECK: return

func.func @inverted_dst_range() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %p = ttl.create_pipe src(3, 0) dst(3, 0) to(0, 0) net 0
      : !ttl.pipe<src(3, 0) dst(3, 0) to(0, 0) net 0>
  ttl.if_dst %p : !ttl.pipe<src(3, 0) dst(3, 0) to(0, 0) net 0> {
  }
  func.return
}

// -----

// Loopback multicast: source coordinate sits inside the destination
// range. Without coalescing the predicate would be the OR of a unit
// cell and a strictly larger rectangle that already covers it; the
// pass drops the redundant source rect, leaving a single destination
// rectangle predicate.

// CHECK-LABEL: func.func @loopback_pipe_coalesces
// CHECK: ttl.core_x
// CHECK: ttl.core_y
// CHECK: arith.cmpi sge
// CHECK: arith.cmpi slt
// CHECK-NOT: arith.ori
// CHECK: scf.if
// CHECK: return

func.func @loopback_pipe_coalesces() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  // src=(0,0) is in column 0 row 0; dst is column 0 rows 0..3 (includes src).
  %p = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 3) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>
  ttl.if_src %p : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0> {
  }
  func.return
}
