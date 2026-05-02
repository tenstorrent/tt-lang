// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-insert-pipenet-active-guards,convert-ttl-to-ttkernel)' | FileCheck %s

// Verifies that the active-set guard pass runs before convert-ttl-to-ttkernel:
//   * ttl.create_pipe is consumed by pipe lowering inside the guard.
//   * ttl.core_x / ttl.core_y in the guard predicate lower to TTKernel
//     coordinate ops.
//   * The outer scf.if from the active-set guard is preserved alongside the
//     inner if_src/if_dst guards inserted by pipe lowering.
//   * The predicate constants survive the pipeline: src cell at (0,0) is
//     contained in dst rect [0,4) x [0,1), so coalescing leaves a single
//     rect with bounds 0/4/0/1 (no arith.ori from the active-set guard).

// Constants get hoisted to the function entry by convert-ttl-to-ttkernel,
// and the inner if_dst lowering shares the same SSA constants via CSE
// (so two distinct lo=0 bounds bind to one %c0). Use CHECK-DAG for
// constant ops; the cmpi sequence is order-stable and uses sequential
// CHECKs.

// CHECK-LABEL: func.func @dm_thread_active_guard_lowered
// Surviving active-set rect: x in [0, 4), y in [0, 1).
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
// CHECK: %[[X:.+]] = ttkernel.my_logical_x_
// CHECK: %[[Y:.+]] = ttkernel.my_logical_y_
// CHECK: arith.cmpi sge, %[[X]], %[[C0]]
// CHECK: arith.cmpi slt, %[[X]], %[[C4]]
// CHECK: arith.cmpi sge, %[[Y]], %[[C0]]
// CHECK: arith.cmpi slt, %[[Y]], %[[C1]]
// CHECK-NOT: arith.ori
// CHECK: scf.if {{.*}} {
// Inner if_dst lowering uses its own coord ops + predicate, then the
// noc_async_read_barrier sits inside that inner scf.if.
// CHECK:   ttkernel.my_logical_x_
// CHECK:   ttkernel.my_logical_y_
// CHECK:   arith.cmpi
// CHECK:   scf.if
// CHECK:     ttkernel.noc_async_read_barrier
// CHECK:   }
// CHECK: }
// CHECK: return
func.func @dm_thread_active_guard_lowered() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %p = ttl.create_pipe src(0, 0) dst(0, 0) to(3, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(3, 0) net 0>
  ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(0, 0) to(3, 0) net 0> {
    "ttkernel.noc_async_read_barrier"() : () -> ()
  }
  func.return
}
