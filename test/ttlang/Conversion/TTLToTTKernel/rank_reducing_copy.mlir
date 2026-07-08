// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttl-to-ttkernel --canonicalize -cse --split-input-file %s | FileCheck %s --check-prefix=TTKERNEL
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttl-to-ttkernel --split-input-file %s | FileCheck %s --check-prefix=ADDR
// Summary: rank-reducing tensor<->CB copy lowering. A 4D tensor is read into a
// 2D DFB: the leading (tensorRank - cbRank) dims are squeezed by scalar start
// indices, so the tile loop nest covers only the trailing DFB dims, and the
// squeezed start index enters the full-grid tile-address linearization.

#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                      buffer = dram, grid = [1, 1], memory = interleaved>

// Tensor: 4D tile grid (2x2x2x2), DFB: 2D [2,2]. rankDiff = 2, so dims 0 and 1
// are squeezed (here selecting leading slot b=1) and only the trailing two dims
// become loops.
// TTKERNEL-LABEL: func.func @rank_reducing_read_4d_into_2d
// TTKERNEL-DAG: %[[TILE_LB:.*]] = arith.constant 0 : index
// TTKERNEL-DAG: %[[TILE_STEP:.*]] = arith.constant 1 : index
// TTKERNEL-DAG: %[[TILES_BOUND:.*]] = arith.constant 2 : index
// TTKERNEL-DAG: %[[NOC:.*]] = arith.constant 0 : i8
// Exactly two nested loops, one per trailing DFB dim, and no loop for the
// squeezed leading dims.
// TTKERNEL: scf.for %{{.*}} = %[[TILE_LB]] to %[[TILES_BOUND]] step %[[TILE_STEP]]
// TTKERNEL:   scf.for %{{.*}} = %[[TILE_LB]] to %[[TILES_BOUND]] step %[[TILE_STEP]]
// TTKERNEL-NOT: scf.for
// TTKERNEL: ttkernel.noc_async_read_tile({{.*}}, %[[NOC]]) : (i32, !ttkernel.TensorAccessor, i32, i8) -> ()
// TTKERNEL: ttkernel.noc_async_read_barrier(%[[NOC]]) : (i8) -> ()
// TTKERNEL-NOT: ttkernel.noc_async_write_barrier

// Addressing contract: the tile index linearizes against the full 4D grid
// (2,2,2,2), and that linearization is expanded into arith ops. The squeezed
// leading slot b=1 folds to a constant tile offset of 1*(2*2*2)=8 (n=0 adds
// nothing); the two trailing coords add the loop IVs, and the result is the
// tile id fed to noc_async_read_tile.
// ADDR-LABEL: func.func @rank_reducing_read_4d_into_2d
// ADDR-DAG: %[[B_OFF:.*]] = arith.constant 8 : index
// ADDR-DAG: %[[NOC:.*]] = arith.constant 0 : i8
// ADDR: scf.for %[[IV0:.*]] =
// ADDR:   scf.for %[[IV1:.*]] =
// ADDR:     %[[M:.*]] = arith.muli %[[IV0]], %{{.*}}
// ADDR:     %[[A0:.*]] = arith.addi %[[M]], %[[B_OFF]]
// ADDR:     %[[TILE:.*]] = arith.addi %[[A0]], %[[IV1]]
// ADDR:     %[[TILE_I32:.*]] = arith.index_cast %[[TILE]] : index to i32
// ADDR:     ttkernel.noc_async_read_tile(%[[TILE_I32]], %{{.*}}, %{{.*}}, %[[NOC]])
module {
  func.func @rank_reducing_read_4d_into_2d(%arg0: tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
    %slice = ttl.tensor_slice %arg0[%c1, %c0, %c0, %c0]
        : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
          -> tensor<2x2x!ttcore.tile<32x32, f32>, #layout>
    %xf = ttl.copy %slice, %cb : (tensor<2x2x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}
