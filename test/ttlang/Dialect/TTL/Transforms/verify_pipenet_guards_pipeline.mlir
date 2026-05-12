// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-verify-pipenet-guards,ttl-erase-pipenet-scopes,convert-ttl-to-ttkernel)' | FileCheck %s

// Summary: Verifies that ttl.pipenet_scope is erased before TTL-to-TTKernel
// lowering.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @scope_erased_before_ttkernel_lowering
  // CHECK-NOT: ttl.pipenet_scope
  // CHECK: ttkernel.my_logical_x_
  // CHECK: return
  func.func @scope_erased_before_ttkernel_lowering() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %x = ttl.core_x : index
    %c1 = arith.constant 1 : index
    %is_src = arith.cmpi slt, %x, %c1 : index
    scf.if %is_src {
      ttl.pipenet_scope attributes {ttl.pipe_net_ids = [0 : i64], ttl.pipe_net_roles = [0 : i64]} {
        ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
          "ttkernel.noc_async_read_barrier"() : () -> ()
        }
      }
    }
    func.return
  }
}
