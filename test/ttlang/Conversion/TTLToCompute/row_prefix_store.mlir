// RUN: ttlang-opt %s --split-input-file \
// RUN:   --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute),canonicalize)' \
// RUN:   | FileCheck %s
// RUN: ttlang-opt %s --split-input-file \
// RUN:   --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute),ttl-set-compute-kernel-config{enable-fpu-binary-ops=0 matmul-full-fp32=0 reduce-full-fp32=0},func.func(ttl-assign-dst,ttl-lower-to-loops,ttl-annotate-cb-associations),convert-ttl-to-ttkernel,ttkernel-insert-inits,canonicalize,cse)' \
// RUN:   | FileCheck %s --check-prefix=TTKERNEL

// Summary: Verifies row-prefix outputs retain complete DFB attachments while
// compute uses compact formal output views.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, 0)>

// A direct elementwise compute publishes a compact bf16 output.
// CHECK-LABEL: func.func @direct_bf16
// CHECK:       %[[EMPTY:.*]] = tensor.empty() : tensor<1x14x!ttcore.tile<1x32, bf16>>
// CHECK:       %[[ATTACHED:.*]] = ttl.attach_cb %[[EMPTY]], %{{.*}}
// CHECK:       %[[OUTPUT:.*]] = tensor.extract_slice %[[ATTACHED]][0, 0] [1, 1] [1, 1]
// CHECK:       ttl.compute
// CHECK-SAME:  ins(%{{.*}}, %{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>)
// CHECK-SAME:  outs(%[[OUTPUT]] : tensor<1x1x!ttcore.tile<1x32, bf16>>)
// CHECK-SAME:  indexing_maps = [#map, #map, #map1]
// CHECK:       ^bb0({{.*}}!ttcore.tile<32x32, bf16>{{.*}}!ttcore.tile<32x32, bf16>{{.*}}!ttcore.tile<1x32, bf16>):
// CHECK:         %[[SUM:.*]] = ttl.tile_add
// CHECK:         ttl.tile_store %[[SUM]], %{{.*}}[%{{.*}}, %{{.*}}] from dst[%{{.*}}] {row_prefix{{.*}}}
// CHECK:       } -> tensor<1x1x!ttcore.tile<1x32, bf16>>
// TTKERNEL-LABEL: func.func @direct_bf16
// TTKERNEL:       ttkernel.pack_rows({{.*}}) {row_count = 28 : i64}
func.func @direct_bf16(
    %lhs: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %rhs: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 16, block_count = 1}
      : !ttl.cb<[1, 14], !ttcore.tile<1x32, bf16>, 1>
  %attached_lhs = ttl.attach_cb %lhs, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %attached_rhs = ttl.attach_cb %rhs, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 14], !ttcore.tile<1x32, bf16>, 1>
        -> tensor<1x14x!ttcore.tile<1x32, bf16>>
  %sum = ttl.add %attached_lhs, %attached_rhs
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %output {row_prefix}
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x14x!ttcore.tile<1x32, bf16>>
  return
}

// -----

// A fused expression publishes a compact f32 output.
// CHECK-LABEL: func.func @fused_f32
// CHECK:       %[[EMPTY:.*]] = tensor.empty() : tensor<1x7x!ttcore.tile<2x32, f32>>
// CHECK:       %[[ATTACHED:.*]] = ttl.attach_cb %[[EMPTY]], %{{.*}}
// CHECK:       %[[OUTPUT:.*]] = tensor.extract_slice %[[ATTACHED]][0, 0] [1, 1] [1, 1]
// CHECK:       ttl.compute
// CHECK-SAME:  ins(%{{.*}}, %{{.*}} : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>)
// CHECK-SAME:  outs(%[[OUTPUT]] : tensor<1x1x!ttcore.tile<2x32, f32>>)
// CHECK-SAME:  indexing_maps = [#map, #map, #map1]
// CHECK:         %[[EXP:.*]] = ttl.tile_exp
// CHECK:         %[[SUM:.*]] = ttl.tile_add %[[EXP]],
// CHECK:         ttl.tile_store %[[SUM]], %{{.*}}[%{{.*}}, %{{.*}}] from dst[%{{.*}}] {row_prefix{{.*}}}
// CHECK:       } -> tensor<1x1x!ttcore.tile<2x32, f32>>
// TTKERNEL-LABEL: func.func @fused_f32
// TTKERNEL:       ttkernel.pack_rows({{.*}}) {row_count = 28 : i64}
func.func @fused_f32(
    %lhs: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %rhs: tensor<1x1x!ttcore.tile<32x32, f32>>) {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 16, block_count = 1}
      : !ttl.cb<[1, 7], !ttcore.tile<2x32, f32>, 1>
  %attached_lhs = ttl.attach_cb %lhs, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %attached_rhs = ttl.attach_cb %rhs, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 7], !ttcore.tile<2x32, f32>, 1>
        -> tensor<1x7x!ttcore.tile<2x32, f32>>
  %exp = ttl.exp %attached_lhs
      : tensor<1x1x!ttcore.tile<32x32, f32>>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %sum = ttl.add %exp, %attached_rhs
      : tensor<1x1x!ttcore.tile<32x32, f32>>,
        tensor<1x1x!ttcore.tile<32x32, f32>>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.store %sum, %output {row_prefix}
      : tensor<1x1x!ttcore.tile<32x32, f32>>,
        tensor<1x7x!ttcore.tile<2x32, f32>>
  return
}

// -----

// A DFB-backed input can publish directly through a compact output.
// CHECK-LABEL: func.func @passthrough_bf16
// CHECK:       %[[EMPTY:.*]] = tensor.empty() : tensor<1x4x!ttcore.tile<4x32, bf16>>
// CHECK:       %[[ATTACHED:.*]] = ttl.attach_cb %[[EMPTY]], %{{.*}}
// CHECK:       %[[OUTPUT:.*]] = tensor.extract_slice %[[ATTACHED]][0, 0] [1, 1] [1, 1]
// CHECK:       ttl.compute
// CHECK-SAME:  ins(%{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>)
// CHECK-SAME:  outs(%[[OUTPUT]] : tensor<1x1x!ttcore.tile<4x32, bf16>>)
// CHECK-SAME:  indexing_maps = [#map, #map1]
// CHECK:       ^bb0(%[[INPUT:.*]]: !ttcore.tile<32x32, bf16>, %{{.*}}: !ttcore.tile<4x32, bf16>):
// CHECK:         ttl.tile_store %[[INPUT]], %{{.*}}[%{{.*}}, %{{.*}}] from dst[%{{.*}}] {row_prefix{{.*}}}
// CHECK:       } -> tensor<1x1x!ttcore.tile<4x32, bf16>>
// TTKERNEL-LABEL: func.func @passthrough_bf16
// TTKERNEL:       ttkernel.pack_rows({{.*}}) {row_count = 32 : i64}
func.func @passthrough_bf16(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 16, block_count = 1}
      : !ttl.cb<[1, 4], !ttcore.tile<4x32, bf16>, 1>
  %attached_input = ttl.attach_cb %input, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 4], !ttcore.tile<4x32, bf16>, 1>
        -> tensor<1x4x!ttcore.tile<4x32, bf16>>
  ttl.store %attached_input, %output {row_prefix}
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x4x!ttcore.tile<4x32, bf16>>
  return
}

// -----

// A zero-input compute publishes the same tile to multiple compact outputs.
// CHECK-LABEL: func.func @multiple_compact_outputs
// CHECK-DAG:   tensor.empty() : tensor<1x8x!ttcore.tile<2x32, bf16>>
// CHECK-DAG:   tensor.empty() : tensor<1x8x!ttcore.tile<2x32, bf16>>
// CHECK-DAG:   tensor.extract_slice {{.*}}[0, 0] [1, 1] [1, 1]
// CHECK-DAG:   tensor.extract_slice {{.*}}[0, 0] [1, 1] [1, 1]
// CHECK:       ttl.compute ins() outs(%{{.*}}, %{{.*}} : tensor<1x1x!ttcore.tile<2x32, bf16>>, tensor<1x1x!ttcore.tile<2x32, bf16>>)
// CHECK-SAME:  indexing_maps = [#map, #map]
// CHECK:         %[[FILL:.*]] = ttl.tile_fill
// CHECK-COUNT-2: ttl.tile_store %[[FILL]], {{.*}} {row_prefix{{.*}}}
// CHECK:       } -> (tensor<1x1x!ttcore.tile<2x32, bf16>>, tensor<1x1x!ttcore.tile<2x32, bf16>>)
// TTKERNEL-LABEL: func.func @multiple_compact_outputs
// TTKERNEL-COUNT-2: ttkernel.pack_rows({{.*}}) {row_count = 32 : i64}
func.func @multiple_compact_outputs() {
  %first_output_dfb = ttl.bind_cb {cb_index = 16, block_count = 1}
      : !ttl.cb<[1, 8], !ttcore.tile<2x32, bf16>, 1>
  %second_output_dfb = ttl.bind_cb {cb_index = 17, block_count = 1}
      : !ttl.cb<[1, 8], !ttcore.tile<2x32, bf16>, 1>
  %first_output = ttl.cb_reserve %first_output_dfb
      : <[1, 8], !ttcore.tile<2x32, bf16>, 1>
        -> tensor<1x8x!ttcore.tile<2x32, bf16>>
  %second_output = ttl.cb_reserve %second_output_dfb
      : <[1, 8], !ttcore.tile<2x32, bf16>, 1>
        -> tensor<1x8x!ttcore.tile<2x32, bf16>>
  %filled = ttl.fill 0.000000e+00
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %filled, %first_output {row_prefix}
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x8x!ttcore.tile<2x32, bf16>>
  ttl.store %filled, %second_output {row_prefix}
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x8x!ttcore.tile<2x32, bf16>>
  return
}
