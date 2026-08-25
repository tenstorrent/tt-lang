// Verifies generated reset helper synchronization and interface-state updates.
// RUN: ttlang-opt --convert-ttkernel-to-emitc %s -o %t.emitc.mlir
// RUN: ttlang-translate --ttkernel-to-cpp %t.emitc.mlir | FileCheck %s

// CHECK: namespace dfb_reset_detail {
// CHECK: constexpr uint32_t stateWordCount = 4;
// CHECK: constexpr uint32_t participantCount = 3;
// CHECK: static_assert(releaseWord + 1 == stateWordCount);
// CHECK: FORCE_INLINE void completeInterfaceWork()
// CHECK: noc_async_full_barrier();
// CHECK: TTI_STALLWAIT(p_stall::STALL_TDMA, waitResources);
// CHECK-NEXT: tensix_sync();
// CHECK: FORCE_INLINE void enter(volatile uint32_t tt_l1_ptr *synchronizationState)
// CHECK: completeInterfaceWork();
// CHECK: while (!participantsHaveState(synchronizationState, entryComplete))
// CHECK: FORCE_INLINE void exit(volatile uint32_t tt_l1_ptr *synchronizationState)
// CHECK: while (!participantsHaveState(synchronizationState, exitComplete))
// CHECK: storeStateWord(&synchronizationState[releaseWord], 0);
// CHECK: FORCE_INLINE void applyMask(uint32_t activeMask, uint32_t firstDFBIndex)
// CHECK: interface.fifo_rd_ptr = base;
// CHECK: interface.fifo_wr_ptr = base;
// CHECK: *get_cb_tiles_received_ptr(dfbIndex) = 0;
// CHECK: *get_cb_tiles_acked_ptr(dfbIndex) = 0;
// CHECK: interface.fifo_wr_tile_ptr = 0;
// CHECK: FORCE_INLINE void complete_dfb_interface_work()
// CHECK: dfb_reset_detail::completeInterfaceWork();
// CHECK: FORCE_INLINE void reset_dfb_interfaces(uint32_t synchronizationAddress,
// CHECK: dfb_reset_detail::applyMask(lowMask, 0);
// CHECK: dfb_reset_detail::applyMask(highMask, 32);
// CHECK-LABEL: void kernel_main()
// CHECK: experimental::reset_dfb_interfaces({{.*}}, {{.*}}, {{.*}});

func.func @kernel_main() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %synchronization_address = arith.constant 4096 : i32
  %low_mask = arith.constant 1 : i32
  %high_mask = arith.constant 2 : i32
  ttkernel.opaque_call "experimental::reset_dfb_interfaces"(%synchronization_address, %low_mask, %high_mask) {header = "<cstdint>", unsigned_arg_indices = array<i32: 0, 1, 2>} : (i32, i32, i32) -> ()
  return
}
