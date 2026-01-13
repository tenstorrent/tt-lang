// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-tile-and-assign-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, canonicalize, cse, lower-affine)' \
// RUN:   -o %t.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// Purpose: end-to-end TTL -> TTKernel -> emitc -> C++ for fused chain.
// Verifies: add + exp fused compute with CB-based data flow.

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: void kernel_main()

// --- Constants ---
// CHECK-DAG:   int32_t [[TILES:v[0-9]+]] = 4
// CHECK-DAG:   size_t [[BOUND:v[0-9]+]] = 2
// CHECK-DAG:   size_t [[STEP:v[0-9]+]] = 1
// CHECK-DAG:   size_t [[ZERO:v[0-9]+]] = 0

// --- DST register lifecycle (acquire before loops) ---
// CHECK:       tile_regs_acquire();

// --- Nested loops over 2x2 tile grid ---
// CHECK-NEXT:  for (size_t [[I:.*]] = [[ZERO]]; [[I]] < [[BOUND]]; [[I]] += [[STEP]]) {
// CHECK-NEXT:    for (size_t [[J:.*]] = [[ZERO]]; [[J]] < [[BOUND]]; [[J]] += [[STEP]]) {

// --- Compute linear tile index: i * cols + j ---
// CHECK:           size_t [[COL_SIZE:.*]] = 2;
// CHECK-NEXT:      size_t [[IOFF:.*]] = [[I]] * [[COL_SIZE]];
// CHECK-NEXT:      size_t [[LINIDX:.*]] = [[IOFF]] + [[J]];

// --- Load tile from CB0 (input A) into DST[dst_idx_a] ---
// CHECK-NEXT:      copy_tile_init(get_compile_time_arg_val(0));
// Dynamic DST index: base + (i * footprint * cols + j * footprint) where footprint=2
// CHECK:           size_t [[DST_A:.*]] = {{.*}} + {{.*}};
// CHECK-NEXT:      copy_tile(get_compile_time_arg_val(0), [[LINIDX]], [[DST_A]]);

// --- Load tile from CB1 (input B) into DST[dst_idx_b] ---
// CHECK-NEXT:      copy_tile_init(get_compile_time_arg_val(1));
// Dynamic DST index: (i * footprint * cols + j * footprint) + 1 for second operand
// CHECK:           size_t [[DST_B_BASE:.*]] = {{.*}} + {{.*}};
// CHECK:           size_t [[DST_B:.*]] = [[DST_B_BASE]] + {{.*}};
// CHECK-NEXT:      copy_tile(get_compile_time_arg_val(1), [[LINIDX]], [[DST_B]]);

// --- Add: DST[dst_idx_a] + DST[dst_idx_b] -> DST[dst_idx_a] ---
// CHECK-NEXT:      add_binary_tile_init();
// CHECK-NEXT:      add_binary_tile([[DST_A]], [[DST_B]], [[DST_A]]);

// --- Mul: DST[dst_idx_a] * DST[dst_idx_b] -> DST[dst_idx_a] ---
// CHECK-NEXT:      mul_binary_tile_init();
// CHECK-NEXT:      mul_binary_tile([[DST_A]], [[DST_B]], [[DST_A]]);

// --- Exp: exp(DST[dst_idx_a]) -> DST[dst_idx_a] ---
// CHECK-NEXT:      exp_tile_init();
// CHECK-NEXT:      exp_tile([[DST_A]]);

// --- End compute loops ---
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// --- DST register synchronization (OUTSIDE loops after fix) ---
// CHECK-NEXT:  tile_regs_commit();
// CHECK-NEXT:  tile_regs_wait();

// --- Pack loop (separate from compute loop after multitile fix) ---
// CHECK-NEXT:  for (size_t [[PACK_I:i[0-9]+]] = [[ZERO]]; [[PACK_I]] < {{.*}}; [[PACK_I]] += [[STEP]]) {
// CHECK-NEXT:  for (size_t [[PACK_J:j[0-9]+]] = [[ZERO]]; [[PACK_J]] < {{.*}}; [[PACK_J]] += [[STEP]]) {

// --- Reserve output CB2 for packing ---
// CHECK-NEXT:      cb_reserve_back(get_compile_time_arg_val(2), [[TILES]]);

// --- Compute CB tile index: i * 2 + j (linearized row-major index) ---
// CHECK:      size_t [[CB_OFF_I:v[0-9]+]] = [[PACK_I]] * {{.*}};
// CHECK-NEXT:      size_t [[CB_IDX:v[0-9]+]] = [[CB_OFF_I]] + [[PACK_J]];

// --- Dynamic DST index for pack: cbTileIndex * footprint ---
// CHECK-NEXT:      size_t [[PACK_DST:v[0-9]+]] = [[CB_IDX]] * {{.*}};

// --- Pack DST[pack_dst] to output CB2 ---
// CHECK-NEXT:      pack_tile<false>([[PACK_DST]], get_compile_time_arg_val(2), [[CB_IDX]]);

// --- Push to signal data ready ---
// CHECK-NEXT:      cb_push_back(get_compile_time_arg_val(2), [[TILES]]);

// --- End of pack loops ---
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// --- DST register lifecycle (release after loops) ---
// CHECK-NEXT:  tile_regs_release();
// CHECK-NEXT:  return;

// --- Verify no tensor operations remain ---
// CHECK-NOT:   tensor.extract
// CHECK-NOT:   tensor.insert
// CHECK-NOT:   tensor.empty
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

  %result = ttl.compute
      ins(%a_ready, %b_ready : tensor<2x2x!ttcore.tile<32x32, f32>>,
                               tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%output_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"],
       "ttl.dst_footprint" = 3 : i32} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %mul = ttl.tile_mul %sum, %b_tile : !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %mul : !ttcore.tile<32x32, f32>
    %result_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %exp, %result_view : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1>
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
