// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst, ttl-lower-to-loops, ttl-insert-tile-regs-sync, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, canonicalize, cse, lower-affine)' \
// RUN:   -o %t.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// Purpose: end-to-end TTL -> TTKernel -> emitc -> C++ for two sequential compute ops.
// Verifies: Each compute gets its own init_sfpu and DST sync ops in the generated C++.
// Pattern: compute1(a + b) -> r0, then compute2(r0 * r0) -> result

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: void kernel_main()
// CHECK-DAG:   int32_t [[TILES:v[0-9]+]] = 4
// CHECK-DAG:   size_t [[BOUND:v[0-9]+]] = 2
// CHECK-DAG:   size_t [[STEP:v[0-9]+]] = 1
// CHECK-DAG:   size_t [[ZERO:v[0-9]+]] = 0
// CHECK:       cb_wait_front(get_compile_time_arg_val(0), [[TILES]]);
// CHECK-NEXT:  cb_wait_front(get_compile_time_arg_val(1), [[TILES]]);
// CHECK-NEXT:  init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(2));
// CHECK-NEXT:  for (size_t [[I1:.*]] = [[ZERO]]; [[I1]] < [[BOUND]]; [[I1]] += [[STEP]]) {
// CHECK-NEXT:    for (size_t [[J1:.*]] = [[ZERO]]; [[J1]] < [[BOUND]]; [[J1]] += [[STEP]]) {
// CHECK-NEXT:      tile_regs_acquire();
// CHECK:           size_t [[COL_SIZE1:.*]] = 2;
// CHECK-NEXT:      size_t [[IOFF1:.*]] = [[I1]] * [[COL_SIZE1]];
// CHECK-NEXT:      size_t [[LINIDX1:.*]] = [[IOFF1]] + [[J1]];
// CHECK-NEXT:      copy_tile_init(get_compile_time_arg_val(0));
// CHECK-NEXT:      copy_tile(get_compile_time_arg_val(0), [[LINIDX1]], [[ZERO]]);
// CHECK-NEXT:      copy_tile_init(get_compile_time_arg_val(1));
// CHECK-NEXT:      copy_tile(get_compile_time_arg_val(1), [[LINIDX1]], [[STEP]]);
// CHECK-NEXT:      add_binary_tile_init();
// CHECK-NEXT:      add_binary_tile([[ZERO]], [[STEP]], [[ZERO]]);
// CHECK-NEXT:      tile_regs_commit();
// CHECK-NEXT:      tile_regs_wait();
// CHECK-NEXT:      cb_reserve_back(get_compile_time_arg_val(2), [[TILES]]);
// CHECK:           size_t [[CB_OFF1:.*]] = [[I1]] * {{.*}};
// CHECK-NEXT:      size_t [[CB_IDX1:.*]] = [[CB_OFF1]] + [[J1]];
// CHECK-NEXT:      pack_tile<false>([[ZERO]], get_compile_time_arg_val(2), [[CB_IDX1]]);
// CHECK-NEXT:      cb_push_back(get_compile_time_arg_val(2), [[TILES]]);
// CHECK-NEXT:      tile_regs_release();
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  init_sfpu(get_compile_time_arg_val(3), get_compile_time_arg_val(4));
// CHECK-NEXT:  for (size_t [[I2:.*]] = [[ZERO]]; [[I2]] < [[BOUND]]; [[I2]] += [[STEP]]) {
// CHECK-NEXT:    for (size_t [[J2:.*]] = [[ZERO]]; [[J2]] < [[BOUND]]; [[J2]] += [[STEP]]) {
// CHECK-NEXT:      tile_regs_acquire();
// CHECK:           size_t [[COL_SIZE2:.*]] = 2;
// CHECK-NEXT:      size_t [[IOFF2:.*]] = [[I2]] * [[COL_SIZE2]];
// CHECK-NEXT:      size_t [[LINIDX2:.*]] = [[IOFF2]] + [[J2]];
// CHECK-NEXT:      copy_tile_init(get_compile_time_arg_val(3));
// CHECK-NEXT:      copy_tile(get_compile_time_arg_val(3), [[LINIDX2]], [[ZERO]]);
// CHECK-NEXT:      copy_tile_init(get_compile_time_arg_val(3));
// CHECK-NEXT:      copy_tile(get_compile_time_arg_val(3), [[LINIDX2]], [[STEP]]);
// CHECK-NEXT:      mul_binary_tile_init();
// CHECK-NEXT:      mul_binary_tile([[ZERO]], [[STEP]], [[ZERO]]);
// CHECK-NEXT:      tile_regs_commit();
// CHECK-NEXT:      tile_regs_wait();
// CHECK-NEXT:      cb_reserve_back(get_compile_time_arg_val(4), [[TILES]]);
// CHECK:           size_t [[CB_OFF2:.*]] = [[I2]] * {{.*}};
// CHECK-NEXT:      size_t [[CB_IDX2:.*]] = [[CB_OFF2]] + [[J2]];
// CHECK-NEXT:      pack_tile<false>([[ZERO]], get_compile_time_arg_val(4), [[CB_IDX2]]);
// CHECK-NEXT:      cb_push_back(get_compile_time_arg_val(4), [[TILES]]);
// CHECK-NEXT:      tile_regs_release();
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return;
// CHECK-NOT:   tensor.extract
// CHECK-NOT:   tensor.insert
// CHECK-NOT:   tensor.empty

func.func @two_computes(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                        %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>

  // Wait for input CBs before first compute.
  %a_ready = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_ready = ttl.cb_wait %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // First compute: a + b -> r0
  %r0 = ttl.compute
      ins(%a_ready, %b_ready : tensor<2x2x!ttcore.tile<32x32, f32>>,
                               tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %view0 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %sum, %view0 : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Attach r0 to a new CB for second compute input.
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %r0_cb = ttl.attach_cb %r0, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // New init for second compute output.
  %init2 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %cb4 = ttl.bind_cb {cb_index = 4, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %init2_cb = ttl.attach_cb %init2, %cb4 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Second compute: r0 * r0 -> result
  %result = ttl.compute
      ins(%r0_cb, %r0_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                           tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init2_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%r0_tile1: !ttcore.tile<32x32, f32>,
       %r0_tile2: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %product = ttl.tile_mul %r0_tile1, %r0_tile2 : !ttcore.tile<32x32, f32>
    %view1 = ttl.cb_reserve %cb4 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %product, %view1 : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb4 : <[2, 2], !ttcore.tile<32x32, f32>, 1>
    ttl.yield %product : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
