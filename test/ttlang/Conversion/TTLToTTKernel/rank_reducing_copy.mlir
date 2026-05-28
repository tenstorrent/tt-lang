// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttl-to-ttkernel --canonicalize -cse --split-input-file %s | FileCheck %s --check-prefix=TTKERNEL
// Summary: rank-reducing tensor<->CB copy lowering. A 4D tensor is read into a
// 2D DFB: the leading (tensorRank - cbRank) dims are squeezed by scalar start
// indices, so the tile loop nest covers only the trailing DFB dims.

#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                      buffer = dram, grid = [1, 1], memory = interleaved>

// Tensor: 4D tile grid (1x1x2x2), DFB: 2D [2,2]. rankDiff = 2, so dims 0 and 1
// are squeezed and only the trailing two dims become loops.
// TTKERNEL-LABEL: func.func @rank_reducing_read_4d_into_2d
// TTKERNEL-DAG: %[[TILE_LB:.*]] = arith.constant 0 : index
// TTKERNEL-DAG: %[[TILE_STEP:.*]] = arith.constant 1 : index
// TTKERNEL-DAG: %[[TILES_BOUND:.*]] = arith.constant 2 : index
// Exactly two nested loops, one per trailing DFB dim, and no loop for the
// squeezed leading dims.
// TTKERNEL: scf.for %{{.*}} = %[[TILE_LB]] to %[[TILES_BOUND]] step %[[TILE_STEP]]
// TTKERNEL:   scf.for %{{.*}} = %[[TILE_LB]] to %[[TILES_BOUND]] step %[[TILE_STEP]]
// TTKERNEL-NOT: scf.for
// TTKERNEL: ttkernel.noc_async_read_tile({{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// TTKERNEL: ttkernel.noc_async_read_barrier() : () -> ()
// TTKERNEL-NOT: ttkernel.noc_async_write_barrier
module {
  func.func @rank_reducing_read_4d_into_2d(%arg0: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
    %slice = ttl.tensor_slice %arg0[%c0, %c0, %c0, %c0]
        : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
          -> tensor<2x2x!ttcore.tile<32x32, f32>, #layout>
    %xf = ttl.copy %slice, %cb : (tensor<2x2x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}
