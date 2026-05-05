// RUN: ttlang-opt --convert-ttl-to-ttkernel="use-trid-barriers=1" --canonicalize -cse --split-input-file %s | FileCheck %s --check-prefix=TTKERNEL
// Summary: Regression tests for TRID-aware ttl.copy/ttl.wait lowering.

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// TTKERNEL-LABEL: func.func @trid_single_copy_wait_read
// TTKERNEL: ttkernel.noc_async_read_set_trid(%[[TRID:.*]], %[[NOC:.*]]) : (i32, i8) -> ()
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
// TTKERNEL: ttkernel.noc_async_read_set_trid(%[[TRID0:.*]], %[[NOC:.*]]) : (i32, i8) -> ()
// TTKERNEL: ttkernel.noc_async_read_tile(
// TTKERNEL: ttkernel.noc_async_read_set_trid(%[[TRID1:.*]], %[[NOC]]) : (i32, i8) -> ()
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

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Verify TRID overflow handling: 17 copies without intervening waits exhaust
// the 16-entry TRID space. The 17th copy (reusing TRID 0) must emit an
// auto-barrier for TRID 0 before reassigning it.
//
// TTKERNEL-LABEL: func.func @trid_overflow_auto_barrier
// The first 16 copies each get a unique TRID (0..15) with no auto-barrier.
// TTKERNEL-COUNT-16: ttkernel.noc_async_read_set_trid
// The 17th copy reuses TRID 0. Because TRID 0 is still outstanding, the pass
// emits an auto-barrier first.
// TTKERNEL: ttkernel.noc_async_read_barrier_with_trid
// TTKERNEL: ttkernel.noc_async_read_set_trid
// No global barriers should appear.
// TTKERNEL-NOT: ttkernel.noc_async_read_barrier() : () -> ()
// TTKERNEL-NOT: builtin.unrealized_conversion_cast
module {
  func.func @trid_overflow_auto_barrier(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %s0 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf0 = ttl.copy %s0, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %s1 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf1 = ttl.copy %s1, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %s2 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf2 = ttl.copy %s2, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %s3 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf3 = ttl.copy %s3, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %s4 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf4 = ttl.copy %s4, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %s5 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf5 = ttl.copy %s5, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %s6 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf6 = ttl.copy %s6, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %s7 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf7 = ttl.copy %s7, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %s8 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf8 = ttl.copy %s8, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %s9 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf9 = ttl.copy %s9, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %s10 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf10 = ttl.copy %s10, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %s11 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf11 = ttl.copy %s11, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %s12 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf12 = ttl.copy %s12, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %s13 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf13 = ttl.copy %s13, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %s14 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf14 = ttl.copy %s14, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %s15 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf15 = ttl.copy %s15, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    // 17th copy: TRID 0 wraps. Auto-barrier expected here.
    %s16 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf16 = ttl.copy %s16, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf0 : !ttl.transfer_handle<read>
    ttl.wait %xf1 : !ttl.transfer_handle<read>
    ttl.wait %xf2 : !ttl.transfer_handle<read>
    ttl.wait %xf3 : !ttl.transfer_handle<read>
    ttl.wait %xf4 : !ttl.transfer_handle<read>
    ttl.wait %xf5 : !ttl.transfer_handle<read>
    ttl.wait %xf6 : !ttl.transfer_handle<read>
    ttl.wait %xf7 : !ttl.transfer_handle<read>
    ttl.wait %xf8 : !ttl.transfer_handle<read>
    ttl.wait %xf9 : !ttl.transfer_handle<read>
    ttl.wait %xf10 : !ttl.transfer_handle<read>
    ttl.wait %xf11 : !ttl.transfer_handle<read>
    ttl.wait %xf12 : !ttl.transfer_handle<read>
    ttl.wait %xf13 : !ttl.transfer_handle<read>
    ttl.wait %xf14 : !ttl.transfer_handle<read>
    ttl.wait %xf15 : !ttl.transfer_handle<read>
    ttl.wait %xf16 : !ttl.transfer_handle<read>
    func.return
  }
}
