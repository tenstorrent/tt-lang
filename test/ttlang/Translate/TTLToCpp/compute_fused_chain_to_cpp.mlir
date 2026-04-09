// FPU path (default): add uses add_tiles (reads from CB), mul uses SFPU.
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse, lower-affine)' \
// RUN:   -o %t.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=FPU

// SFPU path: all binary ops use copy_tile + SFPU binary ops.
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst{enable-fpu-binary-ops=0}, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse, lower-affine)' \
// RUN:   -o %t.sfpu.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.sfpu.ttkernel.mlir -o %t.sfpu.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.sfpu.cpp %t.sfpu.emitc.mlir
// RUN: FileCheck %s --input-file=%t.sfpu.cpp --check-prefix=SFPU

// Purpose: end-to-end TTL -> TTKernel -> emitc -> C++ for fused chain.
// Verifies: add + mul + exp fused compute with CB-based data flow.

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// FPU path: binary_op_init_common, add_tiles, copy_tile (for mul rhs), mul_binary_tile, exp
// =============================================================================
// FPU-LABEL: void kernel_main()

// FPU-DAG:   int32_t [[TILES:v[0-9]+]] = 4
// FPU-DAG:   size_t [[BOUND:v[0-9]+]] = 2
// FPU-DAG:   size_t [[STEP:v[0-9]+]] = 1
// FPU-DAG:   size_t [[ZERO:v[0-9]+]] = 0

// FPU:       cb_reserve_back(get_compile_time_arg_val(2), [[TILES]]);
// FPU:       binary_op_init_common(get_compile_time_arg_val(0), get_compile_time_arg_val(1), get_compile_time_arg_val(2));

// FPU:       for (size_t [[I:.*]] = [[ZERO]]; [[I]] < [[BOUND]]; [[I]] += [[STEP]]) {
// FPU-NEXT:    for (size_t [[J:.*]] = [[ZERO]]; [[J]] < [[BOUND]]; [[J]] += [[STEP]]) {

// FPU:           tile_regs_acquire();

// FPU:           add_tiles_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1));
// FPU-NEXT:      add_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1),

// mul rhs from CB needs copy_tile
// FPU:           copy_tile_init(get_compile_time_arg_val(1));
// FPU-NEXT:      copy_tile(get_compile_time_arg_val(1),

// FPU:           mul_binary_tile_init();
// FPU-NEXT:      mul_binary_tile(

// FPU:           exp_tile_init();
// FPU-NEXT:      exp_tile(

// FPU:           tile_regs_commit();
// FPU-NEXT:      tile_regs_wait();
// FPU:           pack_tile<true>([[ZERO]], get_compile_time_arg_val(2),
// FPU:           tile_regs_release();

// FPU-NOT:   init_sfpu
// FPU-NOT:   add_binary_tile

// =============================================================================
// SFPU path: init_sfpu, copy_tile + add_binary_tile, mul_binary_tile, exp
// =============================================================================
// SFPU-LABEL: void kernel_main()

// SFPU-DAG:   int32_t [[TILES:v[0-9]+]] = 4
// SFPU-DAG:   size_t [[BOUND:v[0-9]+]] = 2
// SFPU-DAG:   size_t [[STEP:v[0-9]+]] = 1
// SFPU-DAG:   size_t [[ZERO:v[0-9]+]] = 0

// SFPU:       cb_reserve_back(get_compile_time_arg_val(2), [[TILES]]);
// SFPU:       init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(2));

// SFPU:       for (size_t [[I:.*]] = [[ZERO]]; [[I]] < [[BOUND]]; [[I]] += [[STEP]]) {
// SFPU-NEXT:    for (size_t [[J:.*]] = [[ZERO]]; [[J]] < [[BOUND]]; [[J]] += [[STEP]]) {

// SFPU:           tile_regs_acquire();

// SFPU:           copy_tile_init(get_compile_time_arg_val(0));
// SFPU-NEXT:      copy_tile(get_compile_time_arg_val(0), {{.*}}, [[ZERO]]);
// SFPU-NEXT:      copy_tile_init(get_compile_time_arg_val(1));
// SFPU-NEXT:      copy_tile(get_compile_time_arg_val(1), {{.*}}, [[STEP]]);

// SFPU-NEXT:      add_binary_tile_init();
// SFPU-NEXT:      add_binary_tile([[ZERO]], [[STEP]], [[ZERO]]);

// SFPU-NEXT:      mul_binary_tile_init();
// SFPU-NEXT:      mul_binary_tile([[ZERO]], [[STEP]], [[ZERO]]);

// SFPU-NEXT:      exp_tile_init();
// SFPU-NEXT:      exp_tile([[ZERO]]);

// SFPU-NEXT:      tile_regs_commit();
// SFPU-NEXT:      tile_regs_wait();

// SFPU:           pack_tile<true>([[ZERO]], get_compile_time_arg_val(2),
// SFPU:           tile_regs_release();

// SFPU-NOT:   binary_op_init_common
// SFPU-NOT:   add_tiles

func.func @fused_chain_lowering(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %output = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>

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
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %c0 = arith.constant 0 : index
    %sum = ttl.tile_add %a_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %mul = ttl.tile_mul %sum, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %mul into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %result_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
