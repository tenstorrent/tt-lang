// RUN: ttlang-opt --split-input-file %s | FileCheck %s
// RUN: ttlang-opt --convert-ttl-to-ttkernel --split-input-file %s | FileCheck %s --check-prefix=LOWERED
// Summary: MVP DMA lowering tests for tensor<->CB copies (no pipes).

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: func.func @dma_single
// CHECK: ttl.create_cb
// CHECK: ttl.copy
// CHECK: ttl.wait

// LOWERED-LABEL: func.func @dma_single
// LOWERED: ttkernel.TensorAccessorArgs
// LOWERED: ttkernel.TensorAccessor
// LOWERED: ttkernel.noc_async_read_tile
// LOWERED: ttkernel.noc_async_read_barrier
// LOWERED: ttkernel.noc_async_write_tile
// LOWERED: ttkernel.noc_async_write_barrier
module {
  func.func @dma_single(%arg0: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %arg0, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.xf
    ttl.wait %xf
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: func.func @cb_to_tensor
// CHECK: ttl.create_cb
// CHECK: ttl.copy
// CHECK: ttl.wait

// LOWERED-LABEL: func.func @cb_to_tensor
// LOWERED: ttkernel.TensorAccessorArgs
// LOWERED: ttkernel.TensorAccessor
// LOWERED: ttkernel.noc_async_write_tile
// LOWERED: ttkernel.noc_async_write_barrier
module {
  func.func @cb_to_tensor(%arg0: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %cb, %arg0 : (!ttl.cb<[1, 1], f32, 2>, tensor<32x32xf32, #layout>) -> !ttl.xf
    ttl.wait %xf
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Batched transfer pattern: issue multiple transfers, then wait on all of them.
// Mirrors TT-Metal kernels that batch NOC async operations for throughput.
// CHECK-LABEL: func.func @dma_batched
// CHECK: ttl.copy
// CHECK: ttl.copy
// CHECK: ttl.wait
// CHECK: ttl.wait
//
// LOWERED-LABEL: func.func @dma_batched
// LOWERED: ttkernel.noc_async_read_tile
// LOWERED: ttkernel.noc_async_read_tile
// LOWERED: ttkernel.noc_async_read_barrier
module {
  func.func @dma_batched(%t0: tensor<32x32xf32, #layout>, %t1: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb0 = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf0 = ttl.copy %t0, %cb0 : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.xf
    %xf1 = ttl.copy %t1, %cb1 : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.xf
    ttl.wait %xf0
    ttl.wait %xf1
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
// LOWERED: ttkernel.noc_async_read_tile
// LOWERED: ttkernel.noc_async_read_barrier
module {
  func.func @dma_pipelined_loop(%t: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index

    %xf_init = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.xf
    %last = scf.for %i = %c0 to %c3 step %c1 iter_args(%prev = %xf_init) -> (!ttl.xf) {
      %xf_next = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.xf
      ttl.wait %prev
      scf.yield %xf_next : !ttl.xf
    }
    ttl.wait %last
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
// LOWERED: ttkernel.noc_async_read_tile
// LOWERED: ttkernel.noc_async_read_barrier
module {
  func.func @dma_two_phase_loops(%t: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %handles0 = tensor.empty(%c4) : tensor<?x!ttl.xf>

    %handles = scf.for %i = %c0 to %c4 step %c1 iter_args(%h = %handles0) -> tensor<?x!ttl.xf> {
      %xf = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.xf
      %h2 = tensor.insert %xf into %h[%i] : tensor<?x!ttl.xf>
      scf.yield %h2 : tensor<?x!ttl.xf>
    }

    scf.for %i = %c0 to %c4 step %c1 {
      %xf = tensor.extract %handles[%i] : tensor<?x!ttl.xf>
      ttl.wait %xf
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
module {
  func.func @dma_double_wait(%t: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.xf
    ttl.wait %xf
    ttl.wait %xf
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
module {
  func.func @dma_single_element_container(%t: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %handles0 = tensor.empty(%c1) : tensor<?x!ttl.xf>

    %handles = scf.for %i = %c0 to %c1 step %c1 iter_args(%h = %handles0) -> tensor<?x!ttl.xf> {
      %xf = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.xf
      %h2 = tensor.insert %xf into %h[%i] : tensor<?x!ttl.xf>
      scf.yield %h2 : tensor<?x!ttl.xf>
    }

    %xf0 = tensor.extract %handles[%c0] : tensor<?x!ttl.xf>
    ttl.wait %xf0
    func.return
  }
}
