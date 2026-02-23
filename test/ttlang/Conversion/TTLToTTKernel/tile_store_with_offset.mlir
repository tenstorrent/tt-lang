// RUN: ttlang-opt %s --split-input-file --convert-ttl-to-ttkernel | FileCheck %s

// Tile store with tile_offset attribute: the CB tile index should include
// the offset (from batch-dst-sync unrolling) instead of being computed
// from (now absent) enclosing loops.

// CHECK-LABEL: func.func @store_with_tile_offset(
// CHECK: %[[C3:.*]] = arith.constant 3 : index
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: ttkernel.pack_tile(%[[C3]], %[[CB]], %[[C3]], {{.*}})
module {
  func.func @store_with_tile_offset(%tile: !ttcore.tile<32x32, bf16>) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
    %view = ttl.cb_reserve %cb : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.tile_store %tile, %view {ttl.tile_offset = 3 : i64} : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Without tile_offset (baseline): CB index is 0 when no loops.

// CHECK-LABEL: func.func @store_without_offset(
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: ttkernel.pack_tile(%[[C0]], %[[CB]], %[[C0]], {{.*}})
module {
  func.func @store_without_offset(%tile: !ttcore.tile<32x32, bf16>) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
    %view = ttl.cb_reserve %cb : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.tile_store %tile, %view : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    func.return
  }
}
