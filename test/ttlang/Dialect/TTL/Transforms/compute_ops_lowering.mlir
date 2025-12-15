// RUN: ttlang-opt --convert-ttl-to-ttkernel --split-input-file %s | FileCheck %s
// Summary: Test convert-ttl-to-ttkernel pass for compute operations.

// -----

// TileAddLowering: Verify LHS CB (arg 0) flows to first copy, RHS CB (arg 1) to second.
// Verify DST indices: lhs -> DST[1], rhs -> DST[0], add(DST[1], DST[0]) -> DST[0].
// CHECK-LABEL: func.func @tile_add_lowering()
// CHECK-SAME: attributes {ttkernel.thread = #ttkernel.thread<compute>}
//
// Capture the two CB values from arg conversion.
// CHECK: %[[LHS_CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
// CHECK: %[[RHS_CB:.*]] = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
//
// Wait on both CBs.
// CHECK: ttkernel.cb_wait_front(%[[LHS_CB]],
// CHECK: ttkernel.cb_wait_front(%[[RHS_CB]],
//
// DST register sequence starts.
// CHECK: ttkernel.tile_regs_acquire()
//
// First copy: LHS CB to DST[1] (cb_idx=0, dst_idx=1).
// CHECK: ttkernel.copy_tile_init(%[[LHS_CB]])
// CHECK: ttkernel.copy_tile(%[[LHS_CB]], %[[CB_IDX:c[0-9_]*]], %[[DST1:c[0-9_]*]])
//
// Second copy: RHS CB to DST[0] (cb_idx=0, dst_idx=0).
// CHECK: ttkernel.copy_tile_init(%[[RHS_CB]])
// CHECK: ttkernel.copy_tile(%[[RHS_CB]], %[[CB_IDX2:c[0-9_]*]], %[[DST0:c[0-9_]*]])
//
// Binary add: DST[1] + DST[0] -> DST[0].
// CHECK: ttkernel.add_binary_tile_init()
// CHECK: ttkernel.add_binary_tile(%[[DST1]], %[[DST0]], %[[DST0]])
//
// Commit and wait for pack.
// CHECK: ttkernel.tile_regs_commit()
// CHECK: ttkernel.tile_regs_wait()
module {
  func.func @tile_add_lowering(
      %lhs_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %rhs_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
      attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32

    %lhs_view = ttl.cb_wait %lhs_cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %rhs_view = ttl.cb_wait %rhs_cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

    %lhs = tensor.extract %lhs_view[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %rhs = tensor.extract %rhs_view[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>

    %result = ttl.tile_add %lhs, %rhs : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

// StoreLowering: Verify output CB (arg 2) flows to pack_tile.
// CHECK-LABEL: func.func @store_lowering()
// CHECK-SAME: attributes {ttkernel.thread = #ttkernel.thread<compute>}
//
// Capture output CB.
// CHECK: %[[OUT_CB:.*]] = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
//
// After add completes, reserve output CB.
// CHECK: ttkernel.tile_regs_commit()
// CHECK: ttkernel.tile_regs_wait()
// CHECK: ttkernel.cb_reserve_back(%[[OUT_CB]],
//
// Pack tile from DST[0] to output CB at index 0.
// CHECK: ttkernel.pack_tile(%[[PACK_SRC:.*]], %[[OUT_CB]], %[[PACK_DST:.*]], false)
//
// Release DST.
// CHECK: ttkernel.tile_regs_release()
module {
  func.func @store_lowering(
      %lhs_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %rhs_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %out_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
      attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32

    %lhs_view = ttl.cb_wait %lhs_cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %rhs_view = ttl.cb_wait %rhs_cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

    %lhs = tensor.extract %lhs_view[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %rhs = tensor.extract %rhs_view[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>

    %result = ttl.tile_add %lhs, %rhs : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>

    %out_view = ttl.cb_reserve %out_cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %result, %out_view : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// Complete kernel: verify full CB flow through compute and store.
// CHECK-LABEL: func.func @complete_add_kernel()
// CHECK-SAME: attributes {ttkernel.thread = #ttkernel.thread<compute>}
//
// All three CBs are retrieved.
// CHECK: %[[LHS_CB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[RHS_CB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: %[[OUT_CB:.*]] = ttkernel.get_compile_time_arg_val(2)
//
// Wait on input CBs.
// CHECK: ttkernel.cb_wait_front(%[[LHS_CB]],
// CHECK: ttkernel.cb_wait_front(%[[RHS_CB]],
//
// Copy from correct CBs.
// CHECK: ttkernel.tile_regs_acquire()
// CHECK: ttkernel.copy_tile_init(%[[LHS_CB]])
// CHECK: ttkernel.copy_tile(%[[LHS_CB]],
// CHECK: ttkernel.copy_tile_init(%[[RHS_CB]])
// CHECK: ttkernel.copy_tile(%[[RHS_CB]],
// CHECK: ttkernel.add_binary_tile
// CHECK: ttkernel.tile_regs_commit
// CHECK: ttkernel.tile_regs_wait
//
// Store to correct output CB.
// CHECK: ttkernel.cb_reserve_back(%[[OUT_CB]],
// CHECK: ttkernel.pack_tile({{.*}}, %[[OUT_CB]],
// CHECK: ttkernel.tile_regs_release
// CHECK: ttkernel.cb_push_back(%[[OUT_CB]],
//
// Pop correct input CBs.
// CHECK: ttkernel.cb_pop_front(%[[LHS_CB]],
// CHECK: ttkernel.cb_pop_front(%[[RHS_CB]],
module {
  func.func @complete_add_kernel(
      %lhs_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %rhs_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %out_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
      attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32

    %lhs_view = ttl.cb_wait %lhs_cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %rhs_view = ttl.cb_wait %rhs_cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

    %lhs = tensor.extract %lhs_view[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %rhs = tensor.extract %rhs_view[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>

    %result = ttl.tile_add %lhs, %rhs : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>

    %out_view = ttl.cb_reserve %out_cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %result, %out_view : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %out_cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>

    ttl.cb_pop %lhs_cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %rhs_cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>

    return
  }
}
