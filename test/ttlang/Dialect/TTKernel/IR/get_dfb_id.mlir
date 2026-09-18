// Round-trip verification of ttkernel.get_dfb_id.
// Verifies that the op parses, prints, and round-trips correctly with the
// expected type abbreviation in the assembly format.

// RUN: ttlang-opt %s | FileCheck %s

// Verify basic round-trip of get_dfb_id on a compile-time CB arg.
// CHECK-LABEL: func.func @get_dfb_id_basic
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
// CHECK-NEXT: %[[ID:.*]] = ttkernel.get_dfb_id %[[CB]] : <1, !ttcore.tile<32x32, bf16>>
func.func @get_dfb_id_basic() -> i32 {
  %cb = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  %id = ttkernel.get_dfb_id %cb : !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  func.return %id : i32
}
