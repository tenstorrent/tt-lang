// Verifies reconfiguration header emission and its hardware entry/exit barriers.
// RUN: ttlang-opt --convert-ttl-to-ttkernel %s -o %t.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp --implicit-check-not="namespace dfb_reconfiguration_detail" --implicit-check-not="asm volatile"
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="__atomic_" --input-file=%S/../../../../include/ttlang/Target/TTKernel/LLKs/experimental_dfb_reconfiguration.h

// HEADER: constexpr uint32_t participantCount = 3;
// HEADER: constexpr uint32_t completionMarker = 0xD1FB;
// HEADER: FORCE_INLINE void drainComputeEngine() {
// HEADER: TTI_STALLWAIT(p_stall::STALL_TDMA, waitResources);
// HEADER-NEXT: TTI_SETDMAREG(0, completionMarker, 0, LO_16(completionGpr));
// HEADER-NEXT: sync_regfile_write(completionGpr);
// HEADER: FORCE_INLINE void enter(volatile uint32_t tt_l1_ptr *synchronizationState) {
// HEADER: storeSynchronizationWord(&synchronizationState[arrivalWord], entryComplete);
// HEADER: while (!participantsHaveState(synchronizationState, entryComplete)) {
// HEADER: FORCE_INLINE void exit(volatile uint32_t tt_l1_ptr *synchronizationState) {
// HEADER: storeSynchronizationWord(&synchronizationState[arrivalWord], exitComplete);
// HEADER: while (!participantsHaveState(synchronizationState, exitComplete)) {
// HEADER: dfb_reconfiguration_detail::enter(synchronizationState);
// HEADER: dfb_reconfiguration_detail::exit(synchronizationState);
// CHECK: #include "api/compute/common.h"
// CHECK: #include "ttlang/Target/TTKernel/LLKs/experimental_dfb_reconfiguration.h"
// CHECK-NOT: experimental_dfb_reconfiguration.h
// CHECK-NOT: experimental_dfb_reset.h
// CHECK: get_arg_val<uint32_t>(get_compile_time_arg_val(0))
// CHECK: experimental::reconfigure_dfb_interfaces({{.*}});

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
