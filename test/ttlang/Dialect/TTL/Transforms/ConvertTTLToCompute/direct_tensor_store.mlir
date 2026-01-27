// RUN: ttlang-opt %s --convert-ttl-to-compute --split-input-file | FileCheck %s

// Test that direct tensor_store (not from compute result) gets lowered to loops.

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @direct_store
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[CB0:.*]] = ttl.bind_cb{cb_index = 0
// CHECK-DAG:     %[[CB1:.*]] = ttl.bind_cb{cb_index = 1
// CHECK:         %[[WAIT:.*]] = ttl.cb_wait %[[CB0]]
// CHECK-NEXT:    %[[ATTACHED:.*]] = ttl.attach_cb %[[WAIT]], %[[CB0]]
// CHECK-NEXT:    scf.for %[[IV0:.*]] = %[[C0]] to %[[C1]] step %[[C1]]
// CHECK-NEXT:      scf.for %[[IV1:.*]] = %[[C0]] to %[[C1]] step %[[C1]]
// CHECK-NEXT:        %[[TILE:.*]] = tensor.extract %[[ATTACHED]][%[[IV0]], %[[IV1]]]
// CHECK-NEXT:        %[[VIEW:.*]] = ttl.cb_reserve %[[CB1]]
// CHECK-NEXT:        ttl.store %[[TILE]], %[[VIEW]]
// CHECK-NOT:   ttl.tensor_store
func.func @direct_store() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>

  // Wait for input and attach to CB
  %input = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %attached = ttl.attach_cb %input, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  // Direct store to output CB (not from a compute result)
  ttl.tensor_store %attached, %cb1 : tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>

  func.return
}

// -----

// Test multi-tile case: loops should iterate over all tiles.

// CHECK-LABEL: func.func @direct_store_multi_tile
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[CB1:.*]] = ttl.bind_cb{cb_index = 1
// CHECK:         %[[ATTACHED:.*]] = ttl.attach_cb
// CHECK-NEXT:    scf.for %[[IV0:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-NEXT:      scf.for %[[IV1:.*]] = %[[C0]] to %[[C4]] step %[[C1]]
// CHECK-NEXT:        %[[TILE:.*]] = tensor.extract %[[ATTACHED]][%[[IV0]], %[[IV1]]]
// CHECK-NEXT:        %[[VIEW:.*]] = ttl.cb_reserve %[[CB1]]
// CHECK-NEXT:        ttl.store %[[TILE]], %[[VIEW]]
// CHECK-NOT:   ttl.tensor_store
func.func @direct_store_multi_tile() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 1>

  %input = ttl.cb_wait %cb0 : <[2, 4], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  %attached = ttl.attach_cb %input, %cb0 : (tensor<2x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 1>) -> tensor<2x4x!ttcore.tile<32x32, bf16>>

  ttl.tensor_store %attached, %cb1 : tensor<2x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 1>

  func.return
}
