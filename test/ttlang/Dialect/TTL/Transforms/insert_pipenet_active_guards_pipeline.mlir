// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-insert-pipenet-active-guards,convert-ttl-to-ttkernel)' | FileCheck %s

// Verifies that the active-set guard pass runs before convert-ttl-to-ttkernel:
//   * ttl.create_pipe is consumed by pipe lowering inside the guard.
//   * ttl.core_x / ttl.core_y in the guard predicate lower to TTKernel
//     coordinate ops.
//   * The outer scf.if from the active-set guard is preserved alongside the
//     inner if_src/if_dst guards inserted by pipe lowering.

// CHECK-LABEL: func.func @dm_thread_active_guard_lowered
// CHECK: ttkernel.my_logical_x_
// CHECK: ttkernel.my_logical_y_
// CHECK: arith.cmpi sge
// CHECK: arith.cmpi slt
// CHECK: scf.if {{.*}} {
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
