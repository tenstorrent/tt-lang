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
// The source unit cell [0,1) x [0,1) is contained in the destination
// rectangle [0,4) x [0,1), so coalescing drops the source rect. The
// surviving predicate is exactly the dst rect, with constants pinned.

// CHECK-LABEL: func.func @dm_thread_single_pipe
// CHECK: %[[X:.+]] = ttl.core_x : index
// CHECK: %[[Y:.+]] = ttl.core_y : index
// Surviving rect: x in [0, 4), y in [0, 1).
// CHECK: %[[XLO:.+]] = arith.constant 0 : index
// CHECK: %[[XHI:.+]] = arith.constant 4 : index
// CHECK: arith.cmpi sge, %[[X]], %[[XLO]]
// CHECK: arith.cmpi slt, %[[X]], %[[XHI]]
// CHECK: %[[YLO:.+]] = arith.constant 0 : index
// CHECK: %[[YHI:.+]] = arith.constant 1 : index
// CHECK: arith.cmpi sge, %[[Y]], %[[YLO]]
// CHECK: arith.cmpi slt, %[[Y]], %[[YHI]]
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

// Compute thread in the same module: no direct pipe reference, but the
// pass still wraps it with the SAME predicate computed module-wide
// from `dm_thread_single_pipe`'s pipe (rect [0,4) x [0,1)). The
// previously bound bind_cb must move inside the scf.if then-region.

// CHECK-LABEL: func.func @compute_thread_with_module_pipe
// CHECK: %[[X2:.+]] = ttl.core_x
// CHECK: %[[Y2:.+]] = ttl.core_y
// Same surviving rect as the dm thread above.
// CHECK: arith.constant 0 : index
// CHECK: arith.constant 4 : index
// CHECK: arith.cmpi sge, %[[X2]], %{{.*}}
// CHECK: arith.cmpi slt, %[[X2]], %{{.*}}
// CHECK: arith.constant 0 : index
// CHECK: arith.constant 1 : index
// CHECK: arith.cmpi sge, %[[Y2]], %{{.*}}
// CHECK: arith.cmpi slt, %[[Y2]], %{{.*}}
// CHECK-NOT: arith.ori
// The bind_cb must end up inside the scf.if then-region, not before it.
// CHECK: scf.if {{.*}} {
// CHECK:   ttl.bind_cb
// CHECK: }
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
// own destination ranges (pipe0 src=(0,0) in dst [0,4)x[0,1); pipe1
// src=(2,1) in dst [2,3)x[1,5)). Coalescing drops both source rects,
// leaving two destination rectangles combined by exactly one arith.ori.

// CHECK-LABEL: func.func @dm_thread_multi_pipe
// CHECK: %[[X:.+]] = ttl.core_x
// CHECK: %[[Y:.+]] = ttl.core_y
// First surviving rect: pipe0 dst [0,4) x [0,1).
// CHECK: arith.constant 0 : index
// CHECK: arith.constant 4 : index
// CHECK: arith.cmpi sge, %[[X]], %{{.*}}
// CHECK: arith.cmpi slt, %[[X]], %{{.*}}
// CHECK: arith.constant 0 : index
// CHECK: arith.constant 1 : index
// CHECK: arith.cmpi sge, %[[Y]], %{{.*}}
// CHECK: arith.cmpi slt, %[[Y]], %{{.*}}
// CHECK: arith.andi
// Second surviving rect: pipe1 dst [2,3) x [1,5).
// CHECK: arith.constant 2 : index
// CHECK: arith.constant 3 : index
// CHECK: arith.cmpi sge, %[[X]], %{{.*}}
// CHECK: arith.cmpi slt, %[[X]], %{{.*}}
// CHECK: arith.constant 1 : index
// CHECK: arith.constant 5 : index
// CHECK: arith.cmpi sge, %[[Y]], %{{.*}}
// CHECK: arith.cmpi slt, %[[Y]], %{{.*}}
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

// Inverted destination range: dst_start_x > dst_end_x. The pass must
// normalize via min/max so the rectangle becomes [0, 4) x [0, 1). The
// source (3, 0) lies inside the normalized destination, so coalescing
// drops the source rect. Constants pin the normalized bounds (not the
// raw 3 / 0 / 0 / 0 attributes on the pipe).

// CHECK-LABEL: func.func @inverted_dst_range
// CHECK: %[[X:.+]] = ttl.core_x
// CHECK: %[[Y:.+]] = ttl.core_y
// Surviving rect after min/max normalization: x in [0, 4), y in [0, 1).
// CHECK: %[[XLO:.+]] = arith.constant 0 : index
// CHECK: %[[XHI:.+]] = arith.constant 4 : index
// CHECK: arith.cmpi sge, %[[X]], %[[XLO]]
// CHECK: arith.cmpi slt, %[[X]], %[[XHI]]
// CHECK: %[[YLO:.+]] = arith.constant 0 : index
// CHECK: %[[YHI:.+]] = arith.constant 1 : index
// CHECK: arith.cmpi sge, %[[Y]], %[[YLO]]
// CHECK: arith.cmpi slt, %[[Y]], %[[YHI]]
// CHECK-NOT: arith.ori
// Raw inverted-bound constant 3 must not leak into the predicate
// (would mean min/max normalization didn't run).
// CHECK-NOT: arith.constant 3 : index
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

// Partial overlap: source cell at (2,0) sits inside the destination
// rectangle [0,4) x [0,1) after normalization. The src unit cell is
// strictly contained in the dst rect, so coalescing drops it and the
// predicate is exactly the dst rect (no arith.ori; constants pin the
// surviving rectangle bounds).

// CHECK-LABEL: func.func @partial_src_overlap_in_dst
// CHECK: %[[X:.+]] = ttl.core_x
// CHECK: %[[Y:.+]] = ttl.core_y
// Surviving dst rect: x in [0, 4), y in [0, 1).
// CHECK: %[[XLO:.+]] = arith.constant 0 : index
// CHECK: %[[XHI:.+]] = arith.constant 4 : index
// CHECK: %[[XGE:.+]] = arith.cmpi sge, %[[X]], %[[XLO]]
// CHECK: %[[XLT:.+]] = arith.cmpi slt, %[[X]], %[[XHI]]
// CHECK: %[[YLO:.+]] = arith.constant 0 : index
// CHECK: %[[YHI:.+]] = arith.constant 1 : index
// CHECK: %[[YGE:.+]] = arith.cmpi sge, %[[Y]], %[[YLO]]
// CHECK: %[[YLT:.+]] = arith.cmpi slt, %[[Y]], %[[YHI]]
// CHECK-NOT: arith.ori
// No second rectangle should appear: src cell constants 2 / 3 must not
// leak into the predicate.
// CHECK-NOT: arith.constant 2 : index
// CHECK-NOT: arith.constant 3 : index
// CHECK: scf.if
// CHECK:   ttl.create_pipe
// CHECK: }
// CHECK: return

func.func @partial_src_overlap_in_dst() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %p = ttl.create_pipe src(2, 0) dst(0, 0) to(3, 0) net 0
      : !ttl.pipe<src(2, 0) dst(0, 0) to(3, 0) net 0>
  ttl.if_src %p : !ttl.pipe<src(2, 0) dst(0, 0) to(3, 0) net 0> {
  }
  func.return
}

// -----

// Two pipes whose source cells and destination rectangles are pairwise
// equal. Coalescing keeps one src rect and one dst rect, not four. The
// surviving rects are src=[1,2) x [1,2) and dst=[2,3) x [2,3); they are
// disjoint, so the predicate has exactly one arith.ori. CHECK pins
// constants so a regression that drops the wrong rectangle (e.g., keeps
// src and src) fails loudly.

// CHECK-LABEL: func.func @two_equal_pipes
// CHECK: %[[X:.+]] = ttl.core_x
// CHECK: %[[Y:.+]] = ttl.core_y
// Surviving src cell: [1,2) x [1,2).
// CHECK: arith.constant 1 : index
// CHECK: arith.constant 2 : index
// CHECK: arith.cmpi sge, %[[X]], %{{.*}}
// CHECK: arith.cmpi slt, %[[X]], %{{.*}}
// CHECK: arith.constant 1 : index
// CHECK: arith.constant 2 : index
// CHECK: arith.cmpi sge, %[[Y]], %{{.*}}
// CHECK: arith.cmpi slt, %[[Y]], %{{.*}}
// Surviving dst cell: [2,3) x [2,3).
// CHECK: arith.constant 2 : index
// CHECK: arith.constant 3 : index
// CHECK: arith.cmpi sge, %[[X]], %{{.*}}
// CHECK: arith.cmpi slt, %[[X]], %{{.*}}
// CHECK: arith.constant 2 : index
// CHECK: arith.constant 3 : index
// CHECK: arith.cmpi sge, %[[Y]], %{{.*}}
// CHECK: arith.cmpi slt, %[[Y]], %{{.*}}
// Exactly one ori (two surviving rects), then no further rect.
// CHECK: arith.ori
// CHECK-NOT: arith.ori
// CHECK-NOT: arith.constant {{[04-9]}} : index
// CHECK: scf.if
// CHECK: return

func.func @two_equal_pipes() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %p0 = ttl.create_pipe src(1, 1) dst(2, 2) to(2, 2) net 0
      : !ttl.pipe<src(1, 1) dst(2, 2) to(2, 2) net 0>
  %p1 = ttl.create_pipe src(1, 1) dst(2, 2) to(2, 2) net 1
      : !ttl.pipe<src(1, 1) dst(2, 2) to(2, 2) net 1>
  ttl.if_src %p0 : !ttl.pipe<src(1, 1) dst(2, 2) to(2, 2) net 0> {
  }
  ttl.if_dst %p1 : !ttl.pipe<src(1, 1) dst(2, 2) to(2, 2) net 1> {
  }
  func.return
}

// -----

// Loopback multicast: source coordinate sits inside the destination
// range. src cell [0,1) x [0,1) is contained in dst rect [0,1) x [0,4),
// so coalescing drops the source. Surviving predicate is exactly the
// dst rect.

// CHECK-LABEL: func.func @loopback_pipe_coalesces
// CHECK: %[[X:.+]] = ttl.core_x
// CHECK: %[[Y:.+]] = ttl.core_y
// Surviving rect: x in [0, 1), y in [0, 4).
// CHECK: %[[XLO:.+]] = arith.constant 0 : index
// CHECK: %[[XHI:.+]] = arith.constant 1 : index
// CHECK: arith.cmpi sge, %[[X]], %[[XLO]]
// CHECK: arith.cmpi slt, %[[X]], %[[XHI]]
// CHECK: %[[YLO:.+]] = arith.constant 0 : index
// CHECK: %[[YHI:.+]] = arith.constant 4 : index
// CHECK: arith.cmpi sge, %[[Y]], %[[YLO]]
// CHECK: arith.cmpi slt, %[[Y]], %[[YHI]]
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
