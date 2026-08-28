// Verify row packing lowers to the public compute API for bf16 and f32 DFBs.

// RUN: ttlang-opt --convert-ttkernel-to-emitc --split-input-file -o %t.emitc.mlir %s
// RUN: FileCheck %s --input-file=%t.emitc.mlir --check-prefix=EMITC
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP

// EMITC-LABEL: func.func @pack_rows_bf16
// EMITC: %[[ROWS:.*]] = emitc.literal "28U" : i32
// EMITC-NEXT: emitc.call_opaque "pack_rows_init"(%[[ROWS]])
// EMITC-NEXT: emitc.call_opaque "pack_rows"
// EMITC-NEXT: emitc.call_opaque "pack_rows_uninit"()
// CPP: #include "api/compute/pack.h"
// CPP-LABEL: void kernel_main() {
// CPP: pack_rows_init(28U);
// CPP-NEXT: pack_rows({{.*}});
// CPP-NEXT: pack_rows_uninit();
func.func @pack_rows_bf16() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %cb = ttkernel.get_compile_time_arg_val(0)
      : () -> !ttkernel.cb<14, !ttcore.tile<1x32, bf16>>
  ttkernel.pack_rows(%c0, %cb, %c0) {row_count = 28 : i64}
      : (index, !ttkernel.cb<14, !ttcore.tile<1x32, bf16>>, index) -> ()
  func.return
}

// -----

// EMITC-LABEL: func.func @pack_rows_f32
// EMITC: %[[ROWS:.*]] = emitc.literal "28U" : i32
// EMITC-NEXT: emitc.call_opaque "pack_rows_init"(%[[ROWS]])
// EMITC-NEXT: emitc.call_opaque "pack_rows"
// EMITC-NEXT: emitc.call_opaque "pack_rows_uninit"()
func.func @pack_rows_f32() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %cb = ttkernel.get_compile_time_arg_val(1)
      : () -> !ttkernel.cb<14, !ttcore.tile<1x32, f32>>
  ttkernel.pack_rows(%c0, %cb, %c0) {row_count = 28 : i64}
      : (index, !ttkernel.cb<14, !ttcore.tile<1x32, f32>>, index) -> ()
  func.return
}
