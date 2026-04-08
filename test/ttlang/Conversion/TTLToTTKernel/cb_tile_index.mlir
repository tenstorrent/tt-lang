// Verify computeCBTileIndex produces correct linearized CB indices.
// Tests identity maps (elementwise), broadcast maps (non-identity),
// and reduction maps (output projection via cb_index_map attribute).

// Identity map test: full pipeline.
// RUN: ttlang-opt %s --split-input-file \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, canonicalize, cse)' \
// RUN:   | FileCheck %s

// 2x3 output with tile loops (not unrolled): pack_tile receives
// index from affine.linearize_index [%row, %col] by (2, 3).
//
// CHECK-LABEL: func.func @tile_index_2x3
// CHECK:       scf.for %[[ROW:.*]] = %{{.*}} to %{{.*}}
// CHECK:         scf.for %[[COL:.*]] = %{{.*}} to %{{.*}}
// CHECK:           %[[IDX:.*]] = affine.linearize_index [%[[ROW]], %[[COL]]] by (2, 3)
// CHECK:           ttkernel.pack_tile(%{{.*}}, %{{.*}}, %[[IDX]]

func.func @tile_index_2x3(
    %output: tensor<2x3x!ttcore.tile<32x32, bf16>>
) -> tensor<2x3x!ttcore.tile<32x32, bf16>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 1>
  %cb_out = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 1>

  %in0 = ttl.cb_wait %cb0 : <[2, 3], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  %in1 = ttl.cb_wait %cb1 : <[2, 3], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  %out_attached = ttl.attach_cb %output, %cb_out : (tensor<2x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 1>) -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  %view = ttl.cb_reserve %cb_out : <[2, 3], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x3x!ttcore.tile<32x32, bf16>>

  %result = ttl.compute
      ins(%in0, %in1 : tensor<2x3x!ttcore.tile<32x32, bf16>>,
                        tensor<2x3x!ttcore.tile<32x32, bf16>>)
      outs(%out_attached : tensor<2x3x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a: !ttcore.tile<32x32, bf16>, %b: !ttcore.tile<32x32, bf16>,
       %c: !ttcore.tile<32x32, bf16>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %sum = ttl.tile_add %a, %b : !ttcore.tile<32x32, bf16>
    ttl.tile_store %sum, %view[%i, %j] : !ttcore.tile<32x32, bf16>, tensor<2x3x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<2x3x!ttcore.tile<32x32, bf16>>

  ttl.cb_pop %cb0 : <[2, 3], !ttcore.tile<32x32, bf16>, 1>
  ttl.cb_pop %cb1 : <[2, 3], !ttcore.tile<32x32, bf16>, 1>
  ttl.cb_push %cb_out : <[2, 3], !ttcore.tile<32x32, bf16>, 1>
  func.return %result : tensor<2x3x!ttcore.tile<32x32, bf16>>
}

// -----

// Broadcast CB index verification with both col and row broadcasts.
// Col-broadcast (2x1 input): map (i,j)->(i), input index = %row only.
// Row-broadcast (1x3 input): map (i,j)->(j), input index = %col only.
// Output pack index: %row * 3 + %col (identity, full linearization).
//
// Hand-crafted post-lowering IR with tile_bcast inside tile loops to
// directly test computeCBTileIndex with non-identity indexing maps.
//
// CHECK-LABEL: func.func @bcast_index_2x3
// CHECK:       scf.for %[[ROW:.*]] = %{{.*}} to %{{.*}}
// CHECK:         scf.for %[[COL:.*]] = %{{.*}} to %{{.*}}
//                  Col-broadcast input: index = %row
// CHECK:           ttkernel.unary_bcast(%{{.*}}, %[[ROW]], %{{.*}}, <col>)
//                  Row-broadcast input: index = %col
// CHECK:           ttkernel.unary_bcast(%{{.*}}, %[[COL]], %{{.*}}, <row>)
//                  Output pack: index from affine.linearize_index
// CHECK:           %[[OUT_IDX:.*]] = affine.linearize_index [%[[ROW]], %[[COL]]] by (2, 3)
// CHECK:           ttkernel.pack_tile(%{{.*}}, %{{.*}}, %[[OUT_IDX]]

func.func @bcast_index_2x3()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb_col = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb_row = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 3], !ttcore.tile<32x32, bf16>, 1>
  %cb_out = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 1>

  %col_in = ttl.cb_wait %cb_col : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %col_cb = ttl.attach_cb %col_in, %cb_col : (tensor<2x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %row_in = ttl.cb_wait %cb_row : <[1, 3], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x3x!ttcore.tile<32x32, bf16>>
  %row_cb = ttl.attach_cb %row_in, %cb_row : (tensor<1x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 3], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x3x!ttcore.tile<32x32, bf16>>
  %view = ttl.cb_reserve %cb_out : <[2, 3], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  %view_cb = ttl.attach_cb %view, %cb_out : (tensor<2x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 1>) -> tensor<2x3x!ttcore.tile<32x32, bf16>>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  scf.for %row = %c0 to %c2 step %c1 {
    scf.for %col = %c0 to %c3 step %c1 {
      ttl.tile_regs_acquire
      // Col-broadcast: 2x1 input, extract at [%row, 0]
      %col_tile = tensor.extract %col_cb[%row, %c0] : tensor<2x1x!ttcore.tile<32x32, bf16>>
      %out_tile = tensor.extract %view_cb[%row, %col] : tensor<2x3x!ttcore.tile<32x32, bf16>>
      %col_bcast = ttl.tile_bcast %col_tile, %out_tile 1 : i32
          {dst_idx = 0 : i32, ttl.bcast_output_cb_index = 2 : index}
          : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
      // Row-broadcast: 1x3 input, extract at [0, %col]
      %row_tile = tensor.extract %row_cb[%c0, %col] : tensor<1x3x!ttcore.tile<32x32, bf16>>
      %row_bcast = ttl.tile_bcast %row_tile, %out_tile 2 : i32
          {dst_idx = 1 : i32, ttl.bcast_output_cb_index = 2 : index}
          : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
      // Store the row-broadcast result (arbitrary choice for the test)
      ttl.tile_store %row_bcast, %view[%row, %col] : !ttcore.tile<32x32, bf16>, tensor<2x3x!ttcore.tile<32x32, bf16>>
      ttl.tile_regs_commit
      ttl.tile_regs_wait
      ttl.tile_regs_release
    } {ttl.tile_loop_stride = 1 : index}
  } {ttl.tile_loop_stride = 3 : index}

  ttl.cb_pop %cb_col : <[2, 1], !ttcore.tile<32x32, bf16>, 1>
  ttl.cb_pop %cb_row : <[1, 3], !ttcore.tile<32x32, bf16>, 1>
  ttl.cb_push %cb_out : <[2, 3], !ttcore.tile<32x32, bf16>, 1>
  func.return
}
