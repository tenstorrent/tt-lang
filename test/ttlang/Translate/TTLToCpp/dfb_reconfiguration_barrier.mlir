// Summary: Verifies generated C++ contains the DFB reconfiguration protocol.
// RUN: ttlang-opt --convert-ttl-to-ttkernel %s -o %t.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// CHECK: FORCE_INLINE void synchronizeParticipants(
// CHECK: __atomic_fetch_or(&synchronizationState[0], arrivalBit, __ATOMIC_RELEASE);
// CHECK: releaseBit) == 0
// CHECK: __atomic_fetch_and(&synchronizationState[0], ~arrivalBit, __ATOMIC_RELEASE);
// CHECK: releaseBit) != 0
// CHECK: __atomic_fetch_or(&synchronizationState[0], releaseBit, __ATOMIC_RELEASE);
// CHECK: allArrivalBits) != 0
// CHECK: __atomic_store_n(&synchronizationState[0], 0, __ATOMIC_RELEASE);
// CHECK: FORCE_INLINE void exit(volatile uint32_t tt_l1_ptr *synchronizationState) {
// CHECK-NEXT: synchronizeParticipants(&synchronizationState[1]);

module attributes {
  ttl.target_arch = #ttcore.arch<blackhole>,
  ttl.dfb_reconfiguration_plan = {
    boundary_ordinals = array<i64: 0>,
    dfbs = []
  }
} {
  func.func @boundary() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>
  } {
    ttl.dfb_reconfiguration #ttl.dfb_reconfiguration<0, participants[#ttl.logical_kernel<kind = compute>, #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">, #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">]>
    return
  }
}
