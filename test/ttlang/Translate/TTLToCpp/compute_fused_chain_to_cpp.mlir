// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-tile-and-assign-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, canonicalize, cse)' \
// RUN:   -o %t.ttkernel.mlir
// RUN: ttmlir-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttmlir-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
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
// CHECK-NEXT:      size_t [[IOFF:.*]] = [[I]] * [[BOUND]];
// CHECK-NEXT:      size_t [[LINIDX:.*]] = [[IOFF]] + [[J]];

// --- Load tile from CB0 (input A) into DST[0] ---
// CHECK-NEXT:      copy_tile_init(get_compile_time_arg_val(0));
// CHECK-NEXT:      copy_tile(get_compile_time_arg_val(0), [[LINIDX]], [[ZERO]]);

// --- Load tile from CB1 (input B) into DST[1] ---
// CHECK-NEXT:      copy_tile_init(get_compile_time_arg_val(1));
// CHECK-NEXT:      copy_tile(get_compile_time_arg_val(1), [[LINIDX]], [[STEP]]);

// --- Add: DST[0] + DST[1] -> DST[0] ---
// CHECK-NEXT:      add_binary_tile_init();
// CHECK-NEXT:      add_binary_tile([[ZERO]], [[STEP]], [[ZERO]]);

// --- Mul: DST[0] * DST[1] -> DST[0] ---
// CHECK-NEXT:      mul_binary_tile_init();
// CHECK-NEXT:      mul_binary_tile([[ZERO]], [[STEP]], [[ZERO]]);

// --- Exp: exp(DST[0]) -> DST[0] ---
// CHECK-NEXT:      exp_tile_init();
// CHECK-NEXT:      exp_tile([[ZERO]]);

// --- DST register synchronization ---
// CHECK-NEXT:      tile_regs_commit();
// CHECK-NEXT:      tile_regs_wait();

// --- Reserve output CB2 for packing ---
// CHECK-NEXT:      cb_reserve_back(get_compile_time_arg_val(2), [[TILES]]);

// --- Pack DST[0] to output CB2 ---
// CHECK-NEXT:      pack_tile<false>([[ZERO]], get_compile_time_arg_val(2), [[ZERO]]);

// --- Push to signal data ready ---
// CHECK-NEXT:      cb_push_back(get_compile_time_arg_val(2), [[TILES]]);

// --- End of inner and outer loops ---
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
       iterator_types = ["parallel", "parallel"]} {
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
