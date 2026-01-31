// RUN: ttlang-opt --convert-ttl-to-ttkernel="use-trid-barriers=1" --canonicalize -cse --split-input-file %s | FileCheck %s --check-prefix=TTKERNEL
// Summary: Regression tests for TRID-aware ttl.copy/ttl.wait lowering.

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// TTKERNEL-LABEL: func.func @trid_single_copy_wait_read
// TTKERNEL-DAG: %[[TRID:.*]] = arith.constant 0 : i32
// TTKERNEL-DAG: %[[NOC:.*]] = arith.constant 0 : i8
// TTKERNEL: ttkernel.noc_async_read_set_trid(%[[TRID]], %[[NOC]]) : (i32, i8) -> ()
// TTKERNEL: ttkernel.noc_async_read_tile(
// TTKERNEL: ttkernel.noc_async_read_barrier_with_trid(%[[TRID]], %[[NOC]]) : (i32, i8) -> ()
// TTKERNEL-NOT: ttkernel.noc_async_read_barrier() : () -> ()
// TTKERNEL-NOT: builtin.unrealized_conversion_cast
module {
  func.func @trid_single_copy_wait_read(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %slice = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf = ttl.copy %slice, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// TTKERNEL-LABEL: func.func @trid_two_copies_two_waits_read
// TTKERNEL-DAG: %[[TRID0:.*]] = arith.constant 0 : i32
// TTKERNEL-DAG: %[[TRID1:.*]] = arith.constant 1 : i32
// TTKERNEL-DAG: %[[NOC:.*]] = arith.constant 0 : i8
// TTKERNEL: ttkernel.noc_async_read_set_trid(%[[TRID0]], %[[NOC]]) : (i32, i8) -> ()
// TTKERNEL: ttkernel.noc_async_read_tile(
// TTKERNEL: ttkernel.noc_async_read_set_trid(%[[TRID1]], %[[NOC]]) : (i32, i8) -> ()
// TTKERNEL: ttkernel.noc_async_read_tile(
// TTKERNEL: ttkernel.noc_async_read_barrier_with_trid(%[[TRID0]], %[[NOC]]) : (i32, i8) -> ()
// TTKERNEL: ttkernel.noc_async_read_barrier_with_trid(%[[TRID1]], %[[NOC]]) : (i32, i8) -> ()
// TTKERNEL-NOT: ttkernel.noc_async_read_barrier() : () -> ()
// TTKERNEL-NOT: builtin.unrealized_conversion_cast
module {
  func.func @trid_two_copies_two_waits_read(%t0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, %t1: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [0, 1], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %slice0 = ttl.tensor_slice %t0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %slice1 = ttl.tensor_slice %t1[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf0 = ttl.copy %slice0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf1 = ttl.copy %slice1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf0 : !ttl.transfer_handle<read>
    ttl.wait %xf1 : !ttl.transfer_handle<read>
    func.return
  }
}
