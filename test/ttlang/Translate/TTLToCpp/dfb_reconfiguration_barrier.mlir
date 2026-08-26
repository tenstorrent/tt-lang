// Verifies that generated C++ contains reconfiguration entry and exit barriers.
// RUN: ttlang-opt --convert-ttl-to-ttkernel %s -o %t.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp
// RUN: FileCheck %s --check-prefix=NO-ATOMICS --input-file=%t.cpp

// CHECK: constexpr uint32_t participantCount = 3;
// CHECK: constexpr uint32_t completionMarker = 0xD1FB;
// CHECK: FORCE_INLINE void drainComputeEngine() {
// CHECK: TTI_STALLWAIT(p_stall::STALL_TDMA, waitResources);
// CHECK-NEXT: TTI_SETDMAREG(0, completionMarker, 0, LO_16(completionGpr));
// CHECK-NEXT: sync_regfile_write(completionGpr);
// CHECK: FORCE_INLINE void enter(volatile uint32_t tt_l1_ptr *synchronizationState) {
// CHECK: storeSynchronizationWord(&synchronizationState[arrivalWord], entryComplete);
// CHECK: while (!participantsHaveState(synchronizationState, entryComplete)) {
// CHECK: FORCE_INLINE void exit(volatile uint32_t tt_l1_ptr *synchronizationState) {
// CHECK: storeSynchronizationWord(&synchronizationState[arrivalWord], exitComplete);
// CHECK: while (!participantsHaveState(synchronizationState, exitComplete)) {
// CHECK: dfb_reconfiguration_detail::enter(synchronizationState);
// CHECK: dfb_reconfiguration_detail::exit(synchronizationState);
// CHECK: get_arg_val<uint32_t>(get_compile_time_arg_val(0))
// NO-ATOMICS-NOT: __atomic_

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
