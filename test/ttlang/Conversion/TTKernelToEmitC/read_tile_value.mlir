// Verify that the processor-distributed DFB scalar read lowers to the
// corresponding TT-Metal compute API.
// RUN: ttlang-opt --convert-ttkernel-to-emitc %s -o %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.emitc.mlir --check-prefix=EMITC
// RUN: ttlang-translate --ttkernel-to-cpp %t.emitc.mlir > %t.cpp
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP

// EMITC-LABEL: func.func @kernel_main
// EMITC: %[[VALUE:.*]] = emitc.call_opaque "read_tile_value"(%{{.*}}, %{{.*}}, %{{.*}}) : (i32, i32, i32) -> i32

// CPP-LABEL: void kernel_main()
// CPP: int32_t [[DFB_ID:v[0-9]+]] = 7;
// CPP-NEXT: int32_t [[PAGE_INDEX:v[0-9]+]] = 2;
// CPP-NEXT: int32_t [[WORD_OFFSET:v[0-9]+]] = 3;
// CPP-NEXT: int32_t [[VALUE:v[0-9]+]] = read_tile_value([[DFB_ID]], [[PAGE_INDEX]], [[WORD_OFFSET]]);
func.func @kernel_main() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %dfb_id = arith.constant 7 : i32
  %page_index = arith.constant 2 : i32
  %word_offset = arith.constant 3 : i32
  %value = ttkernel.read_tile_value(%dfb_id, %page_index, %word_offset) : (i32, i32, i32) -> i32
  return
}
