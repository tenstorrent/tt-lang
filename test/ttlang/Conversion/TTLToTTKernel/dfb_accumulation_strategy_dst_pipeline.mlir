// Summary: Verifies the full TTL-to-TTKernel pipeline treats
// accumulation-strategy=dst as a tensor recurrence policy and still lowers
// user-written DFB accumulation to L1 packer reconfiguration.
//
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline='accumulation-strategy=dst' | FileCheck %s

// The source loop accumulates into a reserved output dataflow buffer slot.
// `accumulation-strategy=dst` must not reject this DFB scope; the DFB lowering
// emits L1 packer metadata consumed by ttkernel-insert-l1-accumulation.
// CHECK-LABEL: func.func @dfb_accumulate_ignores_tensor_dst_strategy
// CHECK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[C0_I32]])
// CHECK-NEXT: scf.for
// CHECK: ttkernel.pack_tile({{.*}}, {{.*}}, {{.*}}, true)
// CHECK: } {ttl.l1_acc_initial = 0 : i32, ttl.l1_acc_loop, ttl.l1_acc_scope_id = 0 : i64}
// CHECK: ttkernel.cb_push_back({{.*}}, %[[C1_I32]])
// CHECK-NEXT: ttkernel.pack_reconfig_l1_acc(%[[C0_I32]])
func.func @dfb_accumulate_ignores_tensor_dst_strategy() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cb_in = ttl.bind_cb {cb_index = 0, block_count = 4} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %cb_out = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %input_wait = ttl.cb_wait %cb_in : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %cb_in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.for %iter = %c0 to %c4 step %c1 {
    ttl.store %input, %reserve {accumulate} : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  func.return
}
