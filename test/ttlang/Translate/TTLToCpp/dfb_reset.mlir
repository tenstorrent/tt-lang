// Verifies reset header emission and the hardware helper's synchronization.
// RUN: ttlang-opt --convert-ttkernel-to-emitc %s -o %t.emitc.mlir
// RUN: ttlang-translate --ttkernel-to-cpp %t.emitc.mlir | FileCheck %s --implicit-check-not="namespace dfb_reset_detail" --implicit-check-not="asm volatile"
// RUN: FileCheck %s --check-prefix=HEADER --input-file=%S/../../../../include/ttlang/Target/TTKernel/LLKs/experimental_dfb_reset.h

// HEADER: namespace dfb_reset_detail {
// HEADER: constexpr uint32_t stateWordCount = 4;
// HEADER: constexpr uint32_t participantCount = 3;
// HEADER: static_assert(releaseWord + 1 == stateWordCount);
// HEADER: FORCE_INLINE void completeInterfaceWork()
// HEADER: noc_async_full_barrier();
// HEADER: TTI_STALLWAIT(p_stall::STALL_TDMA, waitResources);
// HEADER-NEXT: tensix_sync();
// HEADER: FORCE_INLINE void enter(volatile uint32_t tt_l1_ptr *synchronizationState)
// HEADER: completeInterfaceWork();
// HEADER: while (!participantsHaveState(synchronizationState, entryComplete))
// HEADER: FORCE_INLINE void exit(volatile uint32_t tt_l1_ptr *synchronizationState)
// HEADER: while (!participantsHaveState(synchronizationState, exitComplete))
// HEADER: storeStateWord(&synchronizationState[releaseWord], 0);
// HEADER: FORCE_INLINE void applyMask(uint32_t activeMask, uint32_t firstDFBIndex)
// HEADER: interface.fifo_rd_ptr = base;
// HEADER: interface.fifo_wr_ptr = base;
// HEADER: *get_cb_tiles_received_ptr(dfbIndex) = 0;
// HEADER: *get_cb_tiles_acked_ptr(dfbIndex) = 0;
// HEADER: interface.fifo_wr_tile_ptr = 0;
// HEADER: FORCE_INLINE void complete_dfb_interface_work()
// HEADER: dfb_reset_detail::completeInterfaceWork();
// HEADER: FORCE_INLINE void reset_dfb_interfaces(uint32_t synchronizationAddress,
// HEADER: dfb_reset_detail::applyMask(lowMask, 0);
// HEADER: dfb_reset_detail::applyMask(highMask, 32);
// CHECK: #include "api/dataflow/dataflow_api.h"
// CHECK: #include "ttlang/Target/TTKernel/LLKs/experimental_dfb_reset.h"
// CHECK-NOT: experimental_dfb_reset.h
// CHECK-NOT: experimental_dfb_reconfiguration.h
// CHECK-LABEL: void kernel_main()
// CHECK: experimental::reset_dfb_interfaces({{.*}}, {{.*}}, {{.*}});
// CHECK: experimental::reset_dfb_interfaces({{.*}}, {{.*}}, {{.*}});

func.func @kernel_main() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %synchronization_address = arith.constant 4096 : i32
  %low_mask = arith.constant 1 : i32
  %high_mask = arith.constant 2 : i32
  ttkernel.opaque_call "experimental::reset_dfb_interfaces"(%synchronization_address, %low_mask, %high_mask) {header = "<cstdint>", unsigned_arg_indices = array<i32: 0, 1, 2>} : (i32, i32, i32) -> ()
  ttkernel.opaque_call "experimental::reset_dfb_interfaces"(%synchronization_address, %low_mask, %high_mask) {header = "<cstdint>", unsigned_arg_indices = array<i32: 0, 1, 2>} : (i32, i32, i32) -> ()
  return
}
