// Summary: ttl.tile_regs_acquire lowers to ttkernel.tile_regs_acquire.
// RUN: ttlang-opt %s --convert-ttl-to-ttkernel | FileCheck %s

// CHECK-LABEL: func.func @acquire_lower
// Purpose: confirm tile_regs_acquire is preserved and rewritten to the TTKernel lock op.
// CHECK: ttkernel.tile_regs_acquire
// CHECK-NOT: ttl.acquire_dst
// CHECK-NOT: ttl.tile_regs_wait
func.func @acquire_lower() {
  ttl.tile_regs_acquire()
  func.return
}
