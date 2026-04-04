// Summary: In a bcast-only TTKernel function, the init pass derives
// the output CB for unary_bcast_init from the pack_tile op in the
// sync region.

// RUN: ttlang-opt %s --ttkernel-insert-inits | FileCheck %s

// CHECK-LABEL: func.func @bcast_only
// CHECK-DAG: %[[IN_CB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[OUT_CB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.tile_regs_acquire
// unary_bcast_init derives output CB from the pack_tile in the sync region.
// CHECK: ttkernel.unary_bcast_init(%[[IN_CB]], %[[OUT_CB]], <col>)
// CHECK-NEXT: ttkernel.unary_bcast(%[[IN_CB]],
// CHECK: ttkernel.tile_regs_commit
// CHECK: ttkernel.pack_tile({{.*}}, %[[OUT_CB]],
func.func @bcast_only() {
  %in_cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %out_cb = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.unary_bcast(%in_cb, %c0, %c0, <col>) {ttl.bcast_output_cb_index = 1 : index} : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %out_cb, %c0, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}
