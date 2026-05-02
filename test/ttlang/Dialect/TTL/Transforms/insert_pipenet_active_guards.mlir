// RUN: ttlang-opt %s --split-input-file -ttl-insert-pipenet-active-guards | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -ttl-insert-pipenet-active-guards -ttl-insert-pipenet-active-guards | FileCheck %s --check-prefix=IDEMPOTENT

// Verifies the active-set guard pass:
//   * Wraps every kernel-thread function in scf.if when any ttl.create_pipe
//     exists in the module.
//   * Skips functions when no pipes are present.
//   * Skips empty bodies that contain only a terminator.
//   * Is idempotent: running twice still produces a single guard per func.

// Single multicast pipe: src=(0,0), dst range x in [0,3], y=0.

// CHECK-LABEL: func.func @dm_thread_single_pipe
// CHECK: ttl.core_x : index
// CHECK: ttl.core_y : index
// CHECK: arith.cmpi sge
// CHECK: arith.cmpi slt
// CHECK: arith.ori
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

// Multi-pipe case: two pipes contribute distinct rectangles, predicate is
// OR over all rectangles (4 total: 2 src + 2 dst).

// CHECK-LABEL: func.func @dm_thread_multi_pipe
// CHECK: ttl.core_x
// CHECK: ttl.core_y
// CHECK: arith.andi
// CHECK: arith.ori
// CHECK: arith.ori
// CHECK: arith.ori
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
