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

// Tile store with subblock_stride inside a loop with non-unit step.
// Exercises the normalized loop linearization path.
// CB index = (IV / step) * subblock_stride + tile_offset = (IV/2) * 4 + 1.

// CHECK-LABEL: func.func @store_with_subblock_stride(
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: scf.for %[[IV:.*]] =
// CHECK:   %[[NORM:.*]] = arith.divui %[[IV]], %[[C2]]
// CHECK:   %[[SCALED:.*]] = arith.muli %[[NORM]], %[[C4]]
// CHECK:   %[[IDX:.*]] = arith.addi %[[SCALED]], %[[C1]]
// CHECK:   ttkernel.pack_tile(%[[IDX]], %[[CB]], %[[IDX]], {{.*}})
module {
  func.func @store_with_subblock_stride(%tile: !ttcore.tile<32x32, bf16>) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
    %view = ttl.cb_reserve %cb : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    scf.for %i = %c0 to %c6 step %c2 {
      ttl.tile_store %tile, %view {ttl.subblock_stride = 4 : i64, ttl.tile_offset = 1 : i64} : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    }
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
