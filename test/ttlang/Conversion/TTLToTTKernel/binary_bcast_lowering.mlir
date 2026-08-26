// Summary: ttl.tile_binary_bcast lowers to a single ttkernel.binary_bcast
// reading both operands from CBs, plus a binary_bcast_init carrying the
// elementwise op and the broadcast dimension. Neither unary_bcast nor
// copy_tile should appear: the broadcast is applied during unpack rather than
// materialized into DST.

// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(ttl-set-compute-kernel-config, func.func(ttl-assign-dst, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   --split-input-file | FileCheck %s

// Row broadcast folded into an add: the (1, N) operand is unpacked with the
// row broadcast while the (M, N) operand is unpacked normally.
// CHECK-LABEL: func.func @binary_bcast_row_add
// CHECK: %[[DATA_CB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[OUT_CB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: %[[BCAST_CB:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK: ttkernel.binary_op_init_common(%[[DATA_CB]], %[[BCAST_CB]], %[[OUT_CB]])
// CHECK: ttkernel.binary_bcast_init(%[[DATA_CB]], %[[BCAST_CB]], <add>, <row>)
// CHECK: ttkernel.binary_bcast(%[[DATA_CB]], %[[BCAST_CB]],
// CHECK-NOT: ttkernel.unary_bcast
// CHECK-NOT: ttkernel.copy_tile
// CHECK-NOT: ttkernel.add_binary_tile
#map = affine_map<(d0, d1) -> (d0, d1)>
#bcast_row = affine_map<(d0, d1) -> (0, d1)>
func.func @binary_bcast_row_add()
    attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [],
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %data_ready = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %data = ttl.attach_cb %data_ready, %cb0 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %bcast_ready = ttl.cb_wait %cb2 : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %bcast = ttl.attach_cb %bcast_ready, %cb2 : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %out_view = ttl.cb_reserve %cb1 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %empty = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>
  %out_cb = ttl.attach_cb %empty, %cb1 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute
      ins(%data, %bcast : tensor<2x2x!ttcore.tile<32x32, bf16>>,
                          tensor<1x2x!ttcore.tile<32x32, bf16>>)
      outs(%out_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #bcast_row, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%data_tile: !ttcore.tile<32x32, bf16>,
       %bcast_tile: !ttcore.tile<32x32, bf16>,
       %out_tile: !ttcore.tile<32x32, bf16>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %c0 = arith.constant 0 : index
    %sum = ttl.tile_binary_bcast %data_tile, %bcast_tile, %out_tile 0 : i32 2 : i32 into dst[%c0] : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
    ttl.tile_store %sum, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb1 : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb2 : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  func.return
}
