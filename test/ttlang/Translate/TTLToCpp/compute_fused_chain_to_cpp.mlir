// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst, ttl-subblock-compute-for-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-consolidate-inits, canonicalize, cse, lower-affine)' \
// RUN:   -o %t.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// Purpose: end-to-end TTL -> TTKernel -> emitc -> C++ for fused chain.
// Verifies: add + mul + exp fused compute with unrolled tile processing.
// 2x2 tile grid (4 tiles), dstPerIteration=2 (add->dst[0], copy->dst[1]),
// all 4 tiles fit in DST capacity 8, so fully unrolled with no loops.
//
// With operation scheduling and init consolidation:
// - Ops are grouped by kind: all add_tiles, then all copy_tiles, then all
//   mul_binary_tiles, then all exp_tiles
// - One init op per group instead of per-tile

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
// With scheduling + consolidation: ops are grouped by dependency level,
// then by category within each level. Dependency levels:
//   Level 0: copy_tiles (CB->DST) + add_tiles (FPU CB->DST) - no tile op deps
//   Level 1: mul_binary_tiles (SFPU binary) - depends on add+copy results
//   Level 2: exp_tiles (SFPU unary) - depends on mul results
//
// All copy_tiles grouped (one init, depth 0, category 0):
// CHECK:        copy_tile_init(get_compile_time_arg_val(1));
// CHECK:        copy_tile(get_compile_time_arg_val(1),
//
// All add_tiles grouped (one init, depth 0, category 2):
// CHECK:        add_tiles_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1));
// CHECK:        add_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1),
//
// All mul_binary_tiles grouped (one init, depth 1):
// CHECK:        mul_binary_tile_init();
// CHECK:        mul_binary_tile(
//
// All exp_tiles grouped (one init, depth 2):
// CHECK:        exp_tile_init();
// CHECK:        exp_tile(
//
// CHECK:   tile_regs_commit();
// CHECK-NEXT:   tile_regs_wait();
// CHECK:   pack_tile<true>([[C0]], get_compile_time_arg_val(2), [[C0]]);
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
