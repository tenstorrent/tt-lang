// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst, ttl-subblock-compute-for-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, canonicalize, cse, lower-affine)' \
// RUN:   -o %t.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// Purpose: end-to-end TTL -> TTKernel -> emitc -> C++ for fused chain.
// Verifies: add + mul + exp fused compute with unrolled tile processing.
// 2x2 tile grid (4 tiles), dstPerIteration=2 (add→dst[0], copy→dst[1]),
// all 4 tiles fit in DST capacity 8, so fully unrolled with no loops.

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: void kernel_main()
// CHECK-DAG:   int32_t [[TILES:.*]] = 4
// CHECK-DAG:   size_t [[C0:.*]] = 0
// CHECK-DAG:   size_t [[C1:.*]] = 1
// CHECK-DAG:   size_t [[C2:.*]] = 2
// CHECK-DAG:   size_t [[C3:.*]] = 3
// CHECK:   cb_wait_front(get_compile_time_arg_val(0), [[TILES]]);
// CHECK-NEXT:   cb_wait_front(get_compile_time_arg_val(1), [[TILES]]);
// CHECK-NEXT:   cb_reserve_back(get_compile_time_arg_val(2), [[TILES]]);
// CHECK-NEXT:   init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(2));
// CHECK-NEXT:   tile_regs_acquire();
//
// Tile 0: cb_idx=0, dst_base=0
// CHECK-NEXT:   add_tiles_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1));
// CHECK-NEXT:   add_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), [[C0]], [[C0]], [[C0]]);
// CHECK-NEXT:   copy_tile_init(get_compile_time_arg_val(1));
// CHECK-NEXT:   copy_tile(get_compile_time_arg_val(1), [[C0]], [[C1]]);
// CHECK-NEXT:   mul_binary_tile_init();
// CHECK-NEXT:   mul_binary_tile([[C0]], [[C1]], [[C0]]);
// CHECK-NEXT:   exp_tile_init();
// CHECK-NEXT:   exp_tile([[C0]]);
//
// Tile 1: cb_idx=1, dst_base=2
// CHECK-NEXT:   add_tiles_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1));
// CHECK-NEXT:   add_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), [[C1]], [[C1]], [[C2]]);
// CHECK-NEXT:   copy_tile_init(get_compile_time_arg_val(1));
// CHECK-NEXT:   copy_tile(get_compile_time_arg_val(1), [[C1]], [[C3]]);
// CHECK-NEXT:   mul_binary_tile_init();
// CHECK-NEXT:   mul_binary_tile([[C2]], [[C3]], [[C2]]);
// CHECK-NEXT:   exp_tile_init();
// CHECK-NEXT:   exp_tile([[C2]]);
//
// Tile 2: cb_idx=2, dst_base=4
// CHECK:   add_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), [[C2]], [[C2]],
// CHECK:   exp_tile(
//
// Tile 3: cb_idx=3, dst_base=6
// CHECK:   add_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), [[C3]], [[C3]],
// CHECK:   exp_tile(
//
// CHECK:   tile_regs_commit();
// CHECK-NEXT:   tile_regs_wait();
// CHECK-NEXT:   pack_tile<true>([[C0]], get_compile_time_arg_val(2), [[C0]]);
// CHECK:   pack_tile<true>({{.*}}, get_compile_time_arg_val(2), [[C1]]);
// CHECK-NEXT:   pack_tile<true>({{.*}}, get_compile_time_arg_val(2), [[C2]]);
// CHECK-NEXT:   pack_tile<true>({{.*}}, get_compile_time_arg_val(2), [[C3]]);
// CHECK-NEXT:   tile_regs_release();
// CHECK-NEXT:   cb_push_back(get_compile_time_arg_val(2), [[TILES]]);
// CHECK-NEXT:   return;
// CHECK-NOT:   tensor.extract
// CHECK-NOT:   tensor.insert
func.func @fused_chain_lowering(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %output = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>

  // Wait for input CBs (entire blocks) before compute.
  %a_ready = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_ready = ttl.cb_wait %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %output_cb = ttl.attach_cb %output, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_ready, %b_ready : tensor<2x2x!ttcore.tile<32x32, f32>>,
                               tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%output_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %mul = ttl.tile_mul %sum, %b_tile : !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %mul : !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %result_view : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  ttl.cb_push %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
