// RUN: ttlang-opt --split-input-file %s | FileCheck %s
// RUN: ttlang-opt --convert-ttl-to-ttkernel --canonicalize -cse --split-input-file %s | FileCheck %s --check-prefix=LOWERED
// Summary: MVP DMA lowering tests for tensor<->CB copies (no pipes).

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: func.func @dma_single(
// CHECK-SAME: %[[T0:.*]]: tensor<32x32xf32, #ttnn_layout>
// CHECK: %[[CB:.*]] = ttl.create_cb() {{.*}} : <[1, 1], f32, 2>
// CHECK: %[[XF:.*]] = ttl.copy %[[T0]], %[[CB]] : (tensor<32x32xf32, #ttnn_layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
// CHECK: ttl.wait %[[XF]] : !ttl.transfer_handle<read>

// LOWERED-LABEL: func.func @dma_single(
// LOWERED-SAME: %[[ARG0:.*]]: tensor<32x32xf32, {{.*}}>
// LOWERED-DAG: %[[C128:.*]] = arith.constant 128 : i32
// LOWERED-DAG: %[[C1:.*]] = arith.constant 1 : i32
// LOWERED-DAG: %[[C32:.*]] = arith.constant 32 : i32
// LOWERED-DAG: %[[C0:.*]] = arith.constant 0 : i32
// LOWERED-NEXT: %[[SRC_ARGS:.*]] = ttkernel.TensorAccessorArgs(%[[C32]], %[[C1]]) : (i32, i32) -> !ttkernel.TensorAccessorArgs
// LOWERED-NEXT: %[[SRC_ACC:.*]] = ttkernel.TensorAccessor(%[[SRC_ARGS]], %[[C0]], %[[C128]]) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// LOWERED-NEXT: ttkernel.noc_async_read_tile(%[[C0]], %[[SRC_ACC]], %[[C0]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// LOWERED-NEXT: ttkernel.noc_async_read_barrier() : () -> ()
// LOWERED-NOT: ttkernel.noc_async_write_barrier
// LOWERED-NEXT: return
module {
  func.func @dma_single(%arg0: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %arg0, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: func.func @cb_to_tensor(
// CHECK-SAME: %[[T0:.*]]: tensor<32x32xf32, #ttnn_layout>
// CHECK: %[[CB:.*]] = ttl.create_cb() {{.*}} : <[1, 1], f32, 2>
// CHECK: %[[XF:.*]] = ttl.copy %[[CB]], %[[T0]] : (!ttl.cb<[1, 1], f32, 2>, tensor<32x32xf32, #ttnn_layout>) -> !ttl.transfer_handle<write>
// CHECK: ttl.wait %[[XF]] : !ttl.transfer_handle<write>

// LOWERED-LABEL: func.func @cb_to_tensor(
// LOWERED-SAME: %[[ARG0:.*]]: tensor<32x32xf32, {{.*}}>
// LOWERED-DAG: %[[C128:.*]] = arith.constant 128 : i32
// LOWERED-DAG: %[[C1:.*]] = arith.constant 1 : i32
// LOWERED-DAG: %[[C32:.*]] = arith.constant 32 : i32
// LOWERED-DAG: %[[C0:.*]] = arith.constant 0 : i32
// LOWERED-NEXT: %[[DST_ARGS:.*]] = ttkernel.TensorAccessorArgs(%[[C32]], %[[C1]]) : (i32, i32) -> !ttkernel.TensorAccessorArgs
// LOWERED-NEXT: %[[DST_ACC:.*]] = ttkernel.TensorAccessor(%[[DST_ARGS]], %[[C0]], %[[C128]]) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// LOWERED-NEXT: ttkernel.noc_async_write_tile(%[[C0]], %[[DST_ACC]], %[[C0]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// LOWERED-NEXT: ttkernel.noc_async_write_barrier() : () -> ()
// LOWERED-NOT: ttkernel.noc_async_read_barrier
// LOWERED-NEXT: return
module {
  func.func @cb_to_tensor(%arg0: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %cb, %arg0 : (!ttl.cb<[1, 1], f32, 2>, tensor<32x32xf32, #layout>) -> !ttl.transfer_handle<write>
    ttl.wait %xf : !ttl.transfer_handle<write>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Batched transfer pattern: issue multiple transfers, then wait on all of them.
// Mirrors TT-Metal kernels that batch NOC async operations for throughput.
// CHECK-LABEL: func.func @dma_batched(
// CHECK-SAME: %[[T0:[^,]+]]: tensor<32x32xf32, #ttnn_layout>
// CHECK-SAME: %[[T1:[^)]+]]: tensor<32x32xf32, #ttnn_layout>
// CHECK: %[[CB0:.*]] = ttl.create_cb() {{.*}} : <[1, 1], f32, 2>
// CHECK: %[[CB1:.*]] = ttl.create_cb() {{.*}} : <[1, 1], f32, 2>
// CHECK: %[[XF0:.*]] = ttl.copy %[[T0]], %[[CB0]] : (tensor<32x32xf32, #ttnn_layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
// CHECK: %[[XF1:.*]] = ttl.copy %[[T1]], %[[CB1]] : (tensor<32x32xf32, #ttnn_layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
// CHECK: ttl.wait %[[XF0]] : !ttl.transfer_handle<read>
// CHECK: ttl.wait %[[XF1]] : !ttl.transfer_handle<read>
//
// LOWERED-LABEL: func.func @dma_batched
// LOWERED-DAG: %[[C128:.*]] = arith.constant 128 : i32
// LOWERED-DAG: %[[C1:.*]] = arith.constant 1 : i32
// LOWERED-DAG: %[[C32:.*]] = arith.constant 32 : i32
// LOWERED-DAG: %[[C0:.*]] = arith.constant 0 : i32
// LOWERED-NEXT: %[[SRC0_ARGS:.*]] = ttkernel.TensorAccessorArgs(%[[C32]], %[[C1]]) : (i32, i32) -> !ttkernel.TensorAccessorArgs
// LOWERED-NEXT: %[[SRC0_ACC:.*]] = ttkernel.TensorAccessor(%[[SRC0_ARGS]], %[[C0]], %[[C128]]) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// LOWERED-NEXT: ttkernel.noc_async_read_tile(%[[C0]], %[[SRC0_ACC]], %[[C0]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// LOWERED-NEXT: %[[SRC1_ARGS:.*]] = ttkernel.TensorAccessorArgs(%[[C32]], %[[C1]]) : (i32, i32) -> !ttkernel.TensorAccessorArgs
// LOWERED-NEXT: %[[SRC1_ACC:.*]] = ttkernel.TensorAccessor(%[[SRC1_ARGS]], %[[C0]], %[[C128]]) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// LOWERED-NEXT: ttkernel.noc_async_read_tile(%[[C0]], %[[SRC1_ACC]], %[[C0]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// LOWERED-NEXT: ttkernel.noc_async_read_barrier() : () -> ()
// LOWERED-NEXT: ttkernel.noc_async_read_barrier() : () -> ()
// LOWERED-NOT: ttkernel.noc_async_write_barrier
// LOWERED-NEXT: return
module {
  func.func @dma_batched(%t0: tensor<32x32xf32, #layout>, %t1: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb0 = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf0 = ttl.copy %t0, %cb0 : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf1 = ttl.copy %t1, %cb1 : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf0 : !ttl.transfer_handle<read>
    ttl.wait %xf1 : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Pipelined loop pattern: wait on the previous transfer while issuing the next.
// This approximates "copies in one loop, waits in another" by separating the wait
// from the copy in time while staying in SSA form.
// CHECK-LABEL: func.func @dma_pipelined_loop
// CHECK: scf.for
// CHECK: ttl.copy
// CHECK: ttl.wait
//
// LOWERED-LABEL: func.func @dma_pipelined_loop
// LOWERED:      ttkernel.noc_async_read_tile({{.*}}, {{.*}}, {{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// LOWERED:      scf.for {{.*}} {
// LOWERED-NEXT:   %[[LOOP_ARGS:.*]] = ttkernel.TensorAccessorArgs({{.*}}, {{.*}}) : (i32, i32) -> !ttkernel.TensorAccessorArgs
// LOWERED-NEXT:   %[[LOOP_ACC:.*]] = ttkernel.TensorAccessor(%[[LOOP_ARGS]], {{.*}}, {{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// LOWERED-NEXT:   ttkernel.noc_async_read_tile({{.*}}, %[[LOOP_ACC]], {{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// LOWERED-NEXT:   ttkernel.noc_async_read_barrier() : () -> ()
// LOWERED-NEXT: }
// LOWERED-NEXT: ttkernel.noc_async_read_barrier() : () -> ()
// LOWERED-NOT: ttkernel.noc_async_write_barrier
// LOWERED-NEXT: return
module {
  func.func @dma_pipelined_loop(%t: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index

    %xf_init = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %last = scf.for %i = %c0 to %c3 step %c1 iter_args(%prev = %xf_init) -> (!ttl.transfer_handle<read>) {
      %xf_next = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
      ttl.wait %prev : !ttl.transfer_handle<read>
      scf.yield %xf_next : !ttl.transfer_handle<read>
    }
    ttl.wait %last : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Two-phase pattern: issue all copies in one loop, then wait on all handles in
// a second loop. This mirrors TT-Metal kernels that batch NOC async ops and then
// block on a barrier after issuing the batch.
// CHECK-LABEL: func.func @dma_two_phase_loops
// CHECK: scf.for
// CHECK: ttl.copy
// CHECK: scf.for
// CHECK: tensor.extract
// CHECK: ttl.wait
//
// LOWERED-LABEL: func.func @dma_two_phase_loops
// LOWERED:      %[[HANDLES0:.*]] = tensor.empty() : tensor<4x!ttl.transfer_handle<read>>
// LOWERED-NEXT: %[[HANDLES:.*]] = scf.for {{.*}} iter_args(%[[H:.*]] = %[[HANDLES0]]) -> (tensor<4x!ttl.transfer_handle<read>>) {
// LOWERED-NEXT:   %[[ARGS:.*]] = ttkernel.TensorAccessorArgs({{.*}}, {{.*}}) : (i32, i32) -> !ttkernel.TensorAccessorArgs
// LOWERED-NEXT:   %[[ACC:.*]] = ttkernel.TensorAccessor(%[[ARGS]], {{.*}}, {{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// LOWERED-NEXT:   ttkernel.noc_async_read_tile({{.*}}, %[[ACC]], {{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// LOWERED-NEXT:   %[[XF:.*]] = builtin.unrealized_conversion_cast {{.*}} : i32 to !ttl.transfer_handle<read>
// LOWERED-NEXT:   %[[INS:.*]] = tensor.insert %[[XF]] into %[[H]]{{\[}}{{.*}}{{\]}} : tensor<4x!ttl.transfer_handle<read>>
// LOWERED-NEXT:   scf.yield %[[INS]] : tensor<4x!ttl.transfer_handle<read>>
// LOWERED-NEXT: }
// LOWERED-NEXT: scf.for {{.*}} {
// LOWERED-NEXT:   ttkernel.noc_async_read_barrier() : () -> ()
// LOWERED-NEXT: }
// LOWERED-NOT: ttkernel.noc_async_write_barrier
// LOWERED: return
module {
  func.func @dma_two_phase_loops(%t: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %handles0 = tensor.empty(%c4) : tensor<?x!ttl.transfer_handle<read>>

    %handles = scf.for %i = %c0 to %c4 step %c1 iter_args(%h = %handles0) -> tensor<?x!ttl.transfer_handle<read>> {
      %xf = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
      %h2 = tensor.insert %xf into %h[%i] : tensor<?x!ttl.transfer_handle<read>>
      scf.yield %h2 : tensor<?x!ttl.transfer_handle<read>>
    }

    scf.for %i = %c0 to %c4 step %c1 {
      %xf = tensor.extract %handles[%i] : tensor<?x!ttl.transfer_handle<read>>
      ttl.wait %xf : !ttl.transfer_handle<read>
    }
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Corner case: waiting twice on the same transfer handle is allowed.
// CHECK-LABEL: func.func @dma_double_wait
// CHECK: ttl.copy
// CHECK: ttl.wait
// CHECK: ttl.wait
//
// LOWERED-LABEL: func.func @dma_double_wait
// LOWERED:      ttkernel.noc_async_read_barrier() : () -> ()
// LOWERED-NEXT: ttkernel.noc_async_read_barrier() : () -> ()
// LOWERED-NOT: ttkernel.noc_async_write_barrier
module {
  func.func @dma_double_wait(%t: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Corner case: one-element handle batching via tensor.insert, then waiting
// outside of a loop.
// CHECK-LABEL: func.func @dma_single_element_container
// CHECK: tensor.insert
// CHECK: tensor.extract
// CHECK: ttl.wait
//
// LOWERED-LABEL: func.func @dma_single_element_container
// LOWERED: ttkernel.noc_async_read_tile({{.*}}, {{.*}}, {{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// LOWERED: ttkernel.noc_async_read_barrier() : () -> ()
// LOWERED-NOT: ttkernel.noc_async_write_barrier
// LOWERED: return
module {
  func.func @dma_single_element_container(%t: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %handles0 = tensor.empty(%c1) : tensor<?x!ttl.transfer_handle<read>>

    %handles = scf.for %i = %c0 to %c1 step %c1 iter_args(%h = %handles0) -> tensor<?x!ttl.transfer_handle<read>> {
      %xf = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
      %h2 = tensor.insert %xf into %h[%i] : tensor<?x!ttl.transfer_handle<read>>
      scf.yield %h2 : tensor<?x!ttl.transfer_handle<read>>
    }

    %xf0 = tensor.extract %handles[%c0] : tensor<?x!ttl.transfer_handle<read>>
    ttl.wait %xf0 : !ttl.transfer_handle<read>
    func.return
  }
}
