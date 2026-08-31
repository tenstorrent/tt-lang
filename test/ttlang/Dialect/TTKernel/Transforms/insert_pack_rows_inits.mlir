// Verify row-packer configuration is reused across a compatible loop and
// remains operation-local when another operation may reconfigure the packer.

// RUN: ttlang-opt %s --ttkernel-insert-inits --split-input-file | FileCheck %s
// RUN: ttlang-opt %s --ttkernel-insert-inits --ttkernel-insert-l1-accumulation --split-input-file | FileCheck %s --check-prefix=L1

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

// -----

// CHECK-LABEL: func.func @opaque_call_prevents_hoisting
// CHECK: scf.for
// CHECK: ttkernel.opaque_call
// CHECK-NEXT: ttkernel.pack_rows_init {row_count = 28 : i64}
// CHECK-NEXT: ttkernel.pack_rows({{.*}}) {row_count = 28 : i64} : {{.*}} -> (){{$}}
// CHECK-NEXT: ttkernel.pack_rows_uninit
// CHECK: }
func.func @opaque_call_prevents_hoisting() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %cb = ttkernel.get_compile_time_arg_val(0)
      : () -> !ttkernel.cb<14, !ttcore.tile<1x32, bf16>>
  scf.for %iteration = %c0 to %c6 step %c1 {
    ttkernel.opaque_call "configure_external_packer"()
        {header = "configure_external_packer.hpp"} : () -> ()
    ttkernel.pack_rows(%c0, %cb, %c0) {row_count = 28 : i64}
        : (index, !ttkernel.cb<14, !ttcore.tile<1x32, bf16>>, index) -> ()
  }
  func.return
}

// -----

// CHECK-LABEL: func.func @function_call_prevents_hoisting
// CHECK: scf.for
// CHECK: func.call @configure_packer()
// CHECK-NEXT: ttkernel.pack_rows_init {row_count = 28 : i64}
// CHECK-NEXT: ttkernel.pack_rows({{.*}}) {row_count = 28 : i64} : {{.*}} -> (){{$}}
// CHECK-NEXT: ttkernel.pack_rows_uninit
// CHECK: }
func.func @function_call_prevents_hoisting() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %cb = ttkernel.get_compile_time_arg_val(0)
      : () -> !ttkernel.cb<14, !ttcore.tile<1x32, bf16>>
  scf.for %iteration = %c0 to %c6 step %c1 {
    func.call @configure_packer() : () -> ()
    ttkernel.pack_rows(%c0, %cb, %c0) {row_count = 28 : i64}
        : (index, !ttkernel.cb<14, !ttcore.tile<1x32, bf16>>, index) -> ()
  }
  func.return
}

func.func private @configure_packer()

// -----

// L1-LABEL: func.func @loop_scoped_l1_accumulate_existing
// L1: ttkernel.pack_rows_init {row_count = 28 : i64}
// L1-NEXT: %[[ENABLE:.*]] = arith.constant 1 : i32
// L1-NEXT: ttkernel.pack_reconfig_l1_acc(%[[ENABLE]]) : (i32) -> ()
// L1-NEXT: scf.for
// L1: ttkernel.pack_rows({{.*}}) {row_count = 28 : i64} : {{.*}} -> (){{$}}
// L1: }
// L1-NEXT: ttkernel.pack_rows_uninit
// L1-NEXT: ttkernel.cb_push_back
func.func @loop_scoped_l1_accumulate_existing()
    attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0)
      : () -> !ttkernel.cb<14, !ttcore.tile<1x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c14_i32 = arith.constant 14 : i32
  scf.for %iteration = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_rows(%c0, %cb, %c0) {row_count = 28 : i64}
        : (index, !ttkernel.cb<14, !ttcore.tile<1x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_initial = 1 : i32, ttl.l1_acc_loop,
     ttl.l1_acc_scope_id = 0 : i64}
  ttkernel.cb_push_back(%cb, %c14_i32)
      : (!ttkernel.cb<14, !ttcore.tile<1x32, bf16>>, i32) -> ()
  func.return
}

// -----

// L1-LABEL: func.func @operation_scoped_l1_accumulate_existing
// L1: ttkernel.pack_reconfig_l1_acc
// L1-NEXT: scf.for
// L1: ttkernel.copy_tile_init
// L1: ttkernel.pack_rows_init {row_count = 28 : i64}
// L1-NEXT: %[[REENABLE:.*]] = arith.constant 1 : i32
// L1-NEXT: ttkernel.pack_reconfig_l1_acc(%[[REENABLE]]) : (i32) -> ()
// L1-NEXT: ttkernel.pack_rows({{.*}}) {row_count = 28 : i64} : {{.*}} -> (){{$}}
// L1-NEXT: ttkernel.pack_rows_uninit
func.func @operation_scoped_l1_accumulate_existing()
    attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0)
      : () -> !ttkernel.cb<14, !ttcore.tile<1x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c14_i32 = arith.constant 14 : i32
  scf.for %iteration = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.copy_tile_init(%cb)
        : (!ttkernel.cb<14, !ttcore.tile<1x32, bf16>>) -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_rows(%c0, %cb, %c0) {row_count = 28 : i64}
        : (index, !ttkernel.cb<14, !ttcore.tile<1x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_initial = 1 : i32, ttl.l1_acc_loop,
     ttl.l1_acc_scope_id = 0 : i64}
  ttkernel.cb_push_back(%cb, %c14_i32)
      : (!ttkernel.cb<14, !ttcore.tile<1x32, bf16>>, i32) -> ()
  func.return
}
