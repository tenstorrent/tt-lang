// Round-trip verification of ttkernel.get_dfb_id.

// RUN: ttlang-opt %s | FileCheck %s

// CHECK-LABEL: func.func @get_dfb_id_basic
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
// CHECK: %[[ID:.*]] = ttkernel.get_dfb_id %[[CB]] : <1, !ttcore.tile<32x32, bf16>>
func.func @get_dfb_id_basic() -> i32 {
  %cb = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  %id = ttkernel.get_dfb_id %cb : !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  func.return %id : i32
}
