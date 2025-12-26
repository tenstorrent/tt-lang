// RUN: ttlang-opt --convert-ttl-to-ttkernel --canonicalize -cse --split-input-file %s | FileCheck %s
// Summary: Edge case tests for DMA copy grouping.
// Tests verify: (1) CB bindings between copies prevent grouping,
// (2) multi-tile write fusion works correctly.

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Test: CB bind between copies prevents grouping.
// The second CB is bound between two copy operations. Grouping would use cb1
// before definition, so each copy must be lowered separately.
//
// CHECK-LABEL: func.func @cb_bind_between_copies
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK:       %[[ACC0:.*]] = ttkernel.TensorAccessor({{.*}})
// CHECK-NEXT:  %[[PTR0:.*]] = ttkernel.get_write_ptr({{.*}})
// CHECK:       scf.for %[[TY0:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT:    scf.for %[[TX0:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT:      %[[OFFY0:.*]] = arith.muli %[[TY0]], %[[C2]]
// CHECK-NEXT:      %[[OFF0:.*]] = arith.addi %[[OFFY0]], %[[TX0]]
// CHECK-NEXT:      %[[OFF0I32:.*]] = arith.index_cast %[[OFF0]]
// CHECK-NEXT:      ttkernel.noc_async_read_tile(%[[OFF0I32]], %[[ACC0]], %[[PTR0]])
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK:       %[[ACC1:.*]] = ttkernel.TensorAccessor({{.*}})
// CHECK-NEXT:  %[[PTR1:.*]] = ttkernel.get_write_ptr({{.*}})
// CHECK:       scf.for %[[TY1:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT:    scf.for %[[TX1:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT:      %[[OFFY1:.*]] = arith.muli %[[TY1]], %[[C2]]
// CHECK-NEXT:      %[[OFF1:.*]] = arith.addi %[[OFFY1]], %[[TX1]]
// CHECK-NEXT:      %[[OFF1I32:.*]] = arith.index_cast %[[OFF1]]
// CHECK-NEXT:      ttkernel.noc_async_read_tile(%[[OFF1I32]], %[[ACC1]], %[[PTR1]])
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK:       ttkernel.noc_async_read_barrier
// CHECK-NOT:   ttkernel.noc_async_read_barrier
// CHECK-NOT:   ttkernel.noc_async_write_barrier
module {
  func.func @cb_bind_between_copies(%t0: tensor<64x64xf32, #layout>, %t1: tensor<64x64xf32, #layout>)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf0 = ttl.copy %t0, %cb0 : (tensor<64x64xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    // CB bound between copies - prevents grouping.
    %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf1 = ttl.copy %t1, %cb1 : (tensor<64x64xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf0 : !ttl.transfer_handle<read>
    ttl.wait %xf1 : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Test: Batched multi-tile writes fuse into a single loop.
// Verifies write operations (CB->tensor) benefit from grouping.
//
// CHECK-LABEL: func.func @batched_multi_tile_writes
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK:       %[[ACC0:.*]] = ttkernel.TensorAccessor({{.*}})
// CHECK-NEXT:  %[[PTR0:.*]] = ttkernel.get_read_ptr({{.*}})
// CHECK:       %[[ACC1:.*]] = ttkernel.TensorAccessor({{.*}})
// CHECK-NEXT:  %[[PTR1:.*]] = ttkernel.get_read_ptr({{.*}})
// CHECK:       scf.for %[[TY:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT:    scf.for %[[TX:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT:      %[[OFFY:.*]] = arith.muli %[[TY]], %[[C2]]
// CHECK-NEXT:      %[[OFF:.*]] = arith.addi %[[OFFY]], %[[TX]]
// CHECK-NEXT:      %[[OFFI32:.*]] = arith.index_cast %[[OFF]]
// CHECK-NEXT:      ttkernel.noc_async_write_tile(%[[OFFI32]], %[[ACC0]], %[[PTR0]])
// CHECK-NEXT:      ttkernel.noc_async_write_tile(%[[OFFI32]], %[[ACC1]], %[[PTR1]])
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK:       ttkernel.noc_async_write_barrier
// CHECK-NOT:   ttkernel.noc_async_read_barrier
// CHECK-NOT:   ttkernel.noc_async_write_barrier
module {
  func.func @batched_multi_tile_writes(%t0: tensor<64x64xf32, #layout>, %t1: tensor<64x64xf32, #layout>)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf0 = ttl.copy %cb0, %t0 : (!ttl.cb<[1, 1], f32, 2>, tensor<64x64xf32, #layout>) -> !ttl.transfer_handle<write>
    %xf1 = ttl.copy %cb1, %t1 : (!ttl.cb<[1, 1], f32, 2>, tensor<64x64xf32, #layout>) -> !ttl.transfer_handle<write>
    ttl.wait %xf0 : !ttl.transfer_handle<write>
    ttl.wait %xf1 : !ttl.transfer_handle<write>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Test: All CBs bound before copies allows grouping.
// Both operands dominate the first copy, so grouping is safe.
//
// CHECK-LABEL: func.func @all_cbs_before_copies
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK:       %[[ACC0:.*]] = ttkernel.TensorAccessor({{.*}})
// CHECK-NEXT:  %[[PTR0:.*]] = ttkernel.get_write_ptr({{.*}})
// CHECK:       %[[ACC1:.*]] = ttkernel.TensorAccessor({{.*}})
// CHECK-NEXT:  %[[PTR1:.*]] = ttkernel.get_write_ptr({{.*}})
// CHECK:       scf.for %[[TY:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT:    scf.for %[[TX:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT:      %[[OFFY:.*]] = arith.muli %[[TY]], %[[C2]]
// CHECK-NEXT:      %[[OFF:.*]] = arith.addi %[[OFFY]], %[[TX]]
// CHECK-NEXT:      %[[OFFI32:.*]] = arith.index_cast %[[OFF]]
// CHECK-NEXT:      ttkernel.noc_async_read_tile(%[[OFFI32]], %[[ACC0]], %[[PTR0]])
// CHECK-NEXT:      ttkernel.noc_async_read_tile(%[[OFFI32]], %[[ACC1]], %[[PTR1]])
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK:       ttkernel.noc_async_read_barrier
// CHECK-NOT:   ttkernel.noc_async_read_barrier
// CHECK-NOT:   ttkernel.noc_async_write_barrier
module {
  func.func @all_cbs_before_copies(%t0: tensor<64x64xf32, #layout>, %t1: tensor<64x64xf32, #layout>)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf0 = ttl.copy %t0, %cb0 : (tensor<64x64xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf1 = ttl.copy %t1, %cb1 : (tensor<64x64xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf0 : !ttl.transfer_handle<read>
    ttl.wait %xf1 : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Test: Partial fusion with CB bind in the middle of 4 copies.
// Copies 1-2 can fuse, CB3 is bound, then copies 3-4 can fuse.
// Result: two fused loops instead of four separate loops.
//
// CHECK-LABEL: func.func @partial_fusion_four_copies
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// First fused loop (copies 0 and 1)
// CHECK:       %[[ACC0:.*]] = ttkernel.TensorAccessor({{.*}})
// CHECK-NEXT:  %[[PTR0:.*]] = ttkernel.get_write_ptr({{.*}})
// CHECK:       %[[ACC1:.*]] = ttkernel.TensorAccessor({{.*}})
// CHECK-NEXT:  %[[PTR1:.*]] = ttkernel.get_write_ptr({{.*}})
// CHECK:       scf.for %[[TY0:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT:    scf.for %[[TX0:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT:      %[[OFFY0:.*]] = arith.muli %[[TY0]], %[[C2]]
// CHECK-NEXT:      %[[OFF0:.*]] = arith.addi %[[OFFY0]], %[[TX0]]
// CHECK-NEXT:      %[[OFF0I32:.*]] = arith.index_cast %[[OFF0]]
// CHECK-NEXT:      ttkernel.noc_async_read_tile(%[[OFF0I32]], %[[ACC0]], %[[PTR0]])
// CHECK-NEXT:      ttkernel.noc_async_read_tile(%[[OFF0I32]], %[[ACC1]], %[[PTR1]])
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// Second fused loop (copies 2 and 3)
// CHECK:       %[[ACC2:.*]] = ttkernel.TensorAccessor({{.*}})
// CHECK-NEXT:  %[[PTR2:.*]] = ttkernel.get_write_ptr({{.*}})
// CHECK:       %[[ACC3:.*]] = ttkernel.TensorAccessor({{.*}})
// CHECK-NEXT:  %[[PTR3:.*]] = ttkernel.get_write_ptr({{.*}})
// CHECK:       scf.for %[[TY1:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT:    scf.for %[[TX1:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT:      %[[OFFY1:.*]] = arith.muli %[[TY1]], %[[C2]]
// CHECK-NEXT:      %[[OFF1:.*]] = arith.addi %[[OFFY1]], %[[TX1]]
// CHECK-NEXT:      %[[OFF1I32:.*]] = arith.index_cast %[[OFF1]]
// CHECK-NEXT:      ttkernel.noc_async_read_tile(%[[OFF1I32]], %[[ACC2]], %[[PTR2]])
// CHECK-NEXT:      ttkernel.noc_async_read_tile(%[[OFF1I32]], %[[ACC3]], %[[PTR3]])
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK:       ttkernel.noc_async_read_barrier
// CHECK-NOT:   ttkernel.noc_async_read_barrier
// CHECK-NOT:   ttkernel.noc_async_write_barrier
module {
  func.func @partial_fusion_four_copies(
      %t0: tensor<64x64xf32, #layout>, %t1: tensor<64x64xf32, #layout>,
      %t2: tensor<64x64xf32, #layout>, %t3: tensor<64x64xf32, #layout>)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    // Copies 0 and 1 can be fused (cb0, cb1 dominate).
    %xf0 = ttl.copy %t0, %cb0 : (tensor<64x64xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf1 = ttl.copy %t1, %cb1 : (tensor<64x64xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    // CB bound here breaks the chain.
    %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    // Copies 2 and 3 can be fused (cb2, cb3 dominate after the break).
    %xf2 = ttl.copy %t2, %cb2 : (tensor<64x64xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf3 = ttl.copy %t3, %cb3 : (tensor<64x64xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf0 : !ttl.transfer_handle<read>
    ttl.wait %xf1 : !ttl.transfer_handle<read>
    ttl.wait %xf2 : !ttl.transfer_handle<read>
    ttl.wait %xf3 : !ttl.transfer_handle<read>
    func.return
  }
}
