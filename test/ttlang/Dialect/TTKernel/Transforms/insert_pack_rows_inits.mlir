// Verify row-packer configuration is reused across a compatible loop and
// remains operation-local when another packer operation is present.

// RUN: ttlang-opt %s --ttkernel-insert-inits --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @loop_scoped
// CHECK: ttkernel.pack_rows_init {row_count = 28 : i64}
// CHECK-NEXT: scf.for
// CHECK-NOT: ttkernel.pack_rows_init
// CHECK: ttkernel.pack_rows({{.*}}) {row_count = 28 : i64} : {{.*}} -> (){{$}}
// CHECK-NEXT: }
// CHECK-NEXT: ttkernel.pack_rows_uninit
func.func @loop_scoped() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %cb = ttkernel.get_compile_time_arg_val(0)
      : () -> !ttkernel.cb<14, !ttcore.tile<1x32, bf16>>
  scf.for %iteration = %c0 to %c6 step %c1 {
    ttkernel.pack_rows(%c0, %cb, %c0) {row_count = 28 : i64}
        : (index, !ttkernel.cb<14, !ttcore.tile<1x32, bf16>>, index) -> ()
  }
  func.return
}

// -----

// CHECK-LABEL: func.func @operation_scoped
// CHECK: scf.for
// CHECK: ttkernel.pack_rows_init {row_count = 28 : i64}
// CHECK-NEXT: ttkernel.pack_rows({{.*}}) {row_count = 28 : i64} : {{.*}} -> (){{$}}
// CHECK-NEXT: ttkernel.pack_rows_uninit
// CHECK-NEXT: ttkernel.pack_tile(
// CHECK: }
func.func @operation_scoped() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %cb = ttkernel.get_compile_time_arg_val(0)
      : () -> !ttkernel.cb<14, !ttcore.tile<1x32, bf16>>
  scf.for %iteration = %c0 to %c6 step %c1 {
    ttkernel.pack_rows(%c0, %cb, %c0) {row_count = 28 : i64}
        : (index, !ttkernel.cb<14, !ttcore.tile<1x32, bf16>>, index) -> ()
    ttkernel.pack_tile(%c0, %cb, %c0, false)
        : (index, !ttkernel.cb<14, !ttcore.tile<1x32, bf16>>, index) -> ()
  }
  func.return
}
