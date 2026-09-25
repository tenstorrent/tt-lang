// RUN: ttlang-opt --convert-ttl-to-ttkernel --ttkernel-insert-inits %s -o %t.ttkernel.mlir
// RUN: FileCheck %s --input-file=%t.ttkernel.mlir --check-prefix=TTKERNEL
// RUN: ttlang-opt --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP

// Summary: Tests TopK tile operations from TTL through generated kernel C++.

// The three TopK stages share one initialization and retain their operands.
// TTKERNEL-LABEL: func.func @topk
// TTKERNEL: ttkernel.tile_regs_acquire
// TTKERNEL-NEXT: ttkernel.topk_tile_init
// TTKERNEL-NEXT: ttkernel.topk_local_sort
// TTKERNEL-NEXT: ttkernel.topk_merge
// TTKERNEL-NEXT: ttkernel.topk_rebuild
// TTKERNEL-NEXT: ttkernel.tile_regs_release

// CPP: #include "api/compute/topk.h"
// CPP-LABEL: void kernel_main()
// CPP: tile_regs_acquire();
// CPP-NEXT: topk_tile_init();
// CPP-NEXT: topk_local_sort(
// CPP-NEXT: topk_merge(
// CPP-NEXT: topk_rebuild(
// CPP-NEXT: tile_regs_release();
func.func @topk() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %descending = arith.constant 0 : i32
  %end_phase = arith.constant 4 : i32
  %start_phase = arith.constant 0 : i32
  %merge_iteration = arith.constant 0 : i32
  %k = arith.constant 32 : i32
  %logk = arith.constant 5 : i32
  %skip_second = arith.constant 1 : i32
  ttl.tile_regs_acquire
  ttl.tile_topk_local_sort dst[%dst] direction = %descending
      end_phase = %end_phase start_phase = %start_phase
      : (index, i32, i32, i32) -> ()
  ttl.tile_topk_merge dst[%dst] iteration = %merge_iteration k = %k
      : (index, i32, i32) -> ()
  ttl.tile_topk_rebuild dst[%dst] direction = %descending
      iteration = %merge_iteration k = %k logk = %logk
      skip_second = %skip_second
      : (index, i32, i32, i32, i32, i32) -> ()
  ttl.tile_regs_release
  return
}
