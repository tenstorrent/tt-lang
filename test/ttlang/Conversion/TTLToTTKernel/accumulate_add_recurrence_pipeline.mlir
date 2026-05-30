// Summary: Verifies loop-carried additive tensor recurrence lowering through
// the full TTL to TTKernel pipeline uses in-DST accumulation.
//
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline --split-input-file | FileCheck %s

// The source recurrence carries `acc = acc + delta` through an scf.for. The
// pipeline materializes it as a reduction compute, copies the initial tile into
// DST once, reuses that DST slot across the reduction loop, and packs once.
// CHECK-LABEL: func.func @carried_add_dst_compute_pipeline
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C3_I32:.*]] = arith.constant 3 : i32
// CHECK: %[[INIT_CB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[DELTA_CB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: %[[OUT_CB:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK: ttkernel.cb_wait_front(%[[DELTA_CB]], %[[C3_I32]])
// CHECK: ttkernel.tile_regs_acquire
// CHECK-NEXT: ttkernel.copy_tile_init(%[[INIT_CB]])
// CHECK-NEXT: ttkernel.copy_tile(%[[INIT_CB]], %[[C0]], %[[C0]])
// CHECK-NEXT: ttkernel.binary_dest_reuse_tiles_init(%[[DELTA_CB]], <add>, <dest_to_srca>)
// CHECK-NEXT: scf.for %[[RED:.*]] = %[[C0]]
// CHECK-NEXT: ttkernel.binary_dest_reuse_tiles(%[[DELTA_CB]], %[[RED]], %[[C0]], <add>, <dest_to_srca>)
// CHECK-NOT: ttkernel.copy_tile(%[[DELTA_CB]]
// CHECK: ttkernel.pack_tile(%[[C0]], %[[OUT_CB]], %[[C0]], true)
// CHECK: ttkernel.cb_pop_front(%[[DELTA_CB]], %[[C3_I32]])
func.func @carried_add_dst_compute_pipeline() {
  %cb_init = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_delta = ttl.bind_cb {cb_index = 1, block_count = 3} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
  %cb_out = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %init_wait = ttl.cb_wait %cb_init : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init = ttl.attach_cb %init_wait, %cb_init : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %iter = %c0 to %c3 step %c1 iter_args(%acc = %init) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %delta_wait = ttl.cb_wait %cb_delta : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %delta = ttl.attach_cb %delta_wait, %cb_delta : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sum = ttl.add %acc, %delta : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield %sum : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  %reserve = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %loop, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}
