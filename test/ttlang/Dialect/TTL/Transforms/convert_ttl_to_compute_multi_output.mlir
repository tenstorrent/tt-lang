// Tests that convert-ttl-to-compute correctly generates multiple formal outputs
// when a single elementwise result is stored to multiple output CBs (#396).
// Also verifies DST assignment and subblocking work with multi-output computes.
// Covers: binary, unary, fused chains, 3 outputs, and larger shapes.

// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute))' | FileCheck %s --check-prefix=COMPUTE
// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-set-compute-kernel-config,ttl-assign-dst,ttl-subblock-compute-for-dst,ttl-lower-to-loops))' | FileCheck %s --check-prefix=DST

// ---- Test 1: Binary add, 1x1 shape, 2 outputs ----

// COMPUTE: #[[$ID:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// COMPUTE-LABEL: func.func @binary_two_outputs
// COMPUTE:      %[[CB2:.*]] = ttl.bind_cb{cb_index = 2
// COMPUTE:      %[[CB3:.*]] = ttl.bind_cb{cb_index = 3
// COMPUTE:      %[[R2:.*]] = ttl.cb_reserve %[[CB2]]
// COMPUTE:      %[[R3:.*]] = ttl.cb_reserve %[[CB3]]
// COMPUTE:      %[[INIT_ATT3:.*]] = ttl.attach_cb %{{[^,]+}}, %[[CB3]]
// COMPUTE:      %[[INIT_ATT2:.*]] = ttl.attach_cb %{{[^,]+}}, %[[CB2]]
// COMPUTE:      %[[C:.*]]:2 = ttl.compute
// COMPUTE-SAME:   outs(%[[INIT_ATT3]], %[[INIT_ATT2]] :
// COMPUTE-SAME:   indexing_maps = [#[[$ID]], #[[$ID]], #[[$ID]], #[[$ID]]]
// COMPUTE:      ^bb0(%[[IN0:.*]]: !ttcore.tile<32x32, bf16>, %[[IN1:.*]]: !ttcore.tile<32x32, bf16>, %[[OUT0:.*]]: !ttcore.tile<32x32, bf16>, %[[OUT1:.*]]: !ttcore.tile<32x32, bf16>):
// COMPUTE:        ttl.iter_index
// COMPUTE:        ttl.iter_index
// COMPUTE:        %[[SUM:.*]] = ttl.tile_add %[[IN0]], %[[IN1]] : !ttcore.tile<32x32, bf16>
// COMPUTE-NEXT:   ttl.tile_store %[[SUM]], %[[R3]]
// COMPUTE-NEXT:   ttl.tile_store %[[SUM]], %[[R2]]
// COMPUTE-NEXT:   ttl.yield
// COMPUTE:      -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>)

// DST-LABEL: func.func @binary_two_outputs
// DST: ttl.dst_section {
// DST:   ttl.tile_add {{.*}} {dst_idx = 0 : i32, ttl.fpu_binary}
// DST:   ttl.tile_store
// DST-NEXT: ttl.tile_store
// DST: }
module {
  func.func @binary_two_outputs() attributes {ttl.base_cta_index = 4 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb0 = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb1 = ttl.bind_cb{cb_index = 1, buffer_factor = 2} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb2 = ttl.bind_cb{cb_index = 2, buffer_factor = 2} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb3 = ttl.bind_cb{cb_index = 3, buffer_factor = 2} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %a = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %a_att = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b_att = ttl.attach_cb %b, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %r2 = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %r3 = ttl.cb_reserve %cb3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sum = ttl.add %a_att, %b_att : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %sum, %r2 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %sum, %r3 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    %att2 = ttl.attach_cb %sum, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_push %cb3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// ---- Test 2: Unary exp, 2x2 shape, 2 outputs ----
// Exercises the buildUnaryCompute path with multi-output.

// COMPUTE-LABEL: func.func @unary_two_outputs
// COMPUTE:      %[[C:.*]]:2 = ttl.compute
// COMPUTE-SAME:   ins(%{{[^:]+}} :
// COMPUTE-SAME:   outs({{[^)]+}}, {{[^)]+}})
// 1 input + 2 output maps = 3 total.
// COMPUTE-SAME:   indexing_maps = [#{{[^,]+}}, #{{[^,]+}}, #{{[^]]+}}]
// 3 block args: 1 input + 2 outputs.
// COMPUTE:      ^bb0(%{{[^:]+}}: !ttcore.tile<32x32, bf16>, %{{[^:]+}}: !ttcore.tile<32x32, bf16>, %{{[^:]+}}: !ttcore.tile<32x32, bf16>):
// COMPUTE:        ttl.tile_exp
// COMPUTE:        ttl.tile_store
// COMPUTE-NEXT:   ttl.tile_store
// COMPUTE:      -> (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>)

// DST-LABEL: func.func @unary_two_outputs
// DST: ttl.dst_section {
// DST:   ttl.tile_exp {{.*}} {dst_idx = 0 : i32
// DST:   ttl.tile_store
// DST-NEXT: ttl.tile_store
module {
  func.func @unary_two_outputs() attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb0 = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
    %cb1 = ttl.bind_cb{cb_index = 1, buffer_factor = 2} : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
    %cb2 = ttl.bind_cb{cb_index = 2, buffer_factor = 2} : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
    %a = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %a_att = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %r1 = ttl.cb_reserve %cb1 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %r2 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %e = ttl.exp %a_att : tensor<2x2x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.store %e, %r1 : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.store %e, %r2 : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    %att1 = ttl.attach_cb %e, %cb1 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb1 : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_push %cb2 : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb0 : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// ---- Test 3: Fused chain (exp + add), 2x2 shape, 2 outputs ----
// Exercises the buildFusedCompute path with multi-output: the exp is an
// intermediate (not CB-attached), so fusion traces through it.

// COMPUTE-LABEL: func.func @fused_two_outputs
// COMPUTE:      %[[C:.*]]:2 = ttl.compute
// COMPUTE-SAME:   outs({{[^)]+}}, {{[^)]+}})
// 2 input + 2 output maps = 4 total.
// COMPUTE-SAME:   indexing_maps = [#{{[^,]+}}, #{{[^,]+}}, #{{[^,]+}}, #{{[^]]+}}]
// COMPUTE:      ^bb0({{.*}}, {{.*}}, {{.*}}, {{.*}}):
// COMPUTE:        ttl.tile_exp
// COMPUTE:        ttl.tile_add
// COMPUTE:        ttl.tile_store
// COMPUTE-NEXT:   ttl.tile_store
// COMPUTE:      -> (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>)

// DST-LABEL: func.func @fused_two_outputs
// DST: ttl.dst_section {
// DST:   ttl.tile_exp
// DST:   ttl.tile_add
// DST:   ttl.tile_store
// DST-NEXT: ttl.tile_store
module {
  func.func @fused_two_outputs() attributes {ttl.base_cta_index = 4 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb0 = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
    %cb1 = ttl.bind_cb{cb_index = 1, buffer_factor = 2} : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
    %cb2 = ttl.bind_cb{cb_index = 2, buffer_factor = 2} : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
    %cb3 = ttl.bind_cb{cb_index = 3, buffer_factor = 2} : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
    %a = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %a_att = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %b = ttl.cb_wait %cb1 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %b_att = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %r2 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %r3 = ttl.cb_reserve %cb3 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %e = ttl.exp %a_att : tensor<2x2x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %sum = ttl.add %e, %b_att : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.store %sum, %r2 : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.store %sum, %r3 : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    %att = ttl.attach_cb %sum, %cb2 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb2 : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_push %cb3 : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb0 : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb1 : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// ---- Test 4: Binary add, 1x1, 3 outputs ----
// Exercises N > 2 outputs: 5 indexing maps, 5 block args, 3 tile_stores.

// COMPUTE-LABEL: func.func @three_outputs
// COMPUTE:      %[[C:.*]]:3 = ttl.compute
// COMPUTE-SAME:   outs({{[^)]+}}, {{[^)]+}}, {{[^)]+}})
// 2 inputs + 3 outputs = 5 maps.
// COMPUTE-SAME:   indexing_maps = [#{{[^,]+}}, #{{[^,]+}}, #{{[^,]+}}, #{{[^,]+}}, #{{[^]]+}}]
// COMPUTE:      ^bb0({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}):
// COMPUTE:        ttl.tile_add
// COMPUTE:        ttl.tile_store
// COMPUTE-NEXT:   ttl.tile_store
// COMPUTE-NEXT:   ttl.tile_store
// COMPUTE:      -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>)

// DST-LABEL: func.func @three_outputs
// DST: ttl.dst_section {
// DST:   ttl.tile_add {{.*}} {dst_idx = 0 : i32, ttl.fpu_binary}
// DST:   ttl.tile_store
// DST-NEXT: ttl.tile_store
// DST-NEXT: ttl.tile_store
// DST: }
module {
  func.func @three_outputs() attributes {ttl.base_cta_index = 5 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb0 = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb1 = ttl.bind_cb{cb_index = 1, buffer_factor = 2} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb2 = ttl.bind_cb{cb_index = 2, buffer_factor = 2} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb3 = ttl.bind_cb{cb_index = 3, buffer_factor = 2} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb4 = ttl.bind_cb{cb_index = 4, buffer_factor = 2} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %a = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %a_att = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b_att = ttl.attach_cb %b, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %r2 = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %r3 = ttl.cb_reserve %cb3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %r4 = ttl.cb_reserve %cb4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sum = ttl.add %a_att, %b_att : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %sum, %r2 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %sum, %r3 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %sum, %r4 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    %att = ttl.attach_cb %sum, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_push %cb3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_push %cb4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// ---- Test 5: Binary add, 4x4, 2 outputs with DST subblocking ----
// 16 tiles, DST capacity 8 -> subblock outer loop with step 2.
// Each subblock: 8 tile_add with dst_idx 0..7, each followed by 2 tile_stores.

// COMPUTE-LABEL: func.func @multi_output_4x4
// COMPUTE:      %[[C2:.*]]:2 = ttl.compute
// COMPUTE:      ^bb0({{.*}}, {{.*}}, {{.*}}, {{.*}}):
// COMPUTE:        ttl.tile_add
// COMPUTE:        ttl.tile_store
// COMPUTE-NEXT:   ttl.tile_store
// COMPUTE:      -> (tensor<4x4x!ttcore.tile<32x32, bf16>>, tensor<4x4x!ttcore.tile<32x32, bf16>>)

// DST-LABEL: func.func @multi_output_4x4
// DST:       scf.for
// DST:         ttl.dst_section {
// DST:           ttl.tile_add {{.*}} {dst_idx = 0 : i32
// DST:           ttl.tile_add {{.*}} {dst_idx = 7 : i32
// DST:           ttl.tile_store
// DST:           ttl.tile_store
// DST:         }
module {
  func.func @multi_output_4x4() attributes {ttl.base_cta_index = 4 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb0 = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
    %cb1 = ttl.bind_cb{cb_index = 1, buffer_factor = 2} : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
    %cb2 = ttl.bind_cb{cb_index = 2, buffer_factor = 2} : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
    %cb3 = ttl.bind_cb{cb_index = 3, buffer_factor = 2} : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
    %a = ttl.cb_wait %cb0 : <[4, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %a_att = ttl.attach_cb %a, %cb0 : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %b = ttl.cb_wait %cb1 : <[4, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %b_att = ttl.attach_cb %b, %cb1 : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %r2 = ttl.cb_reserve %cb2 : <[4, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %r3 = ttl.cb_reserve %cb3 : <[4, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %sum = ttl.add %a_att, %b_att : tensor<4x4x!ttcore.tile<32x32, bf16>>, tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    ttl.store %sum, %r2 : tensor<4x4x!ttcore.tile<32x32, bf16>>, tensor<4x4x!ttcore.tile<32x32, bf16>>
    ttl.store %sum, %r3 : tensor<4x4x!ttcore.tile<32x32, bf16>>, tensor<4x4x!ttcore.tile<32x32, bf16>>
    %att2 = ttl.attach_cb %sum, %cb2 : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb2 : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_push %cb3 : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb0 : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb1 : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
