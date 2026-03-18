// RUN: ttlang-opt %s --split-input-file \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst{enable-fpu-binary-ops=0}, ttl-subblock-compute-for-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s --check-prefix=CHECK-WAR

// Purpose: Regression test for WAR (Write-After-Read) hazard in DST scheduling.
//
// The compute chain `tanh(a) + b + c` requires 3 CB inputs per tile, but only
// 2 DST register slots (since b's slot is freed after the first add, c reuses
// it). With 2 tiles (2x1 bf16), the lowered body looks like:
//
//   copy(a0)->dst0, tanh, copy(b0)->dst1, add, copy(c0)->dst1, add  [tile 0]
//   copy(a1)->dst2, tanh, copy(b1)->dst3, add, copy(c1)->dst3, add  [tile 1]
//
// BUG (fixed): Without WAR tracking, the scheduler grouped ALL copy_tile ops
// at depth 0, moving copy(c)->dst1 before the add that reads dst1 (from b).
// The fix tracks anti-dependencies: copy(c)->dst1 must wait for the add that
// reads dst1 to complete.
//
// Expected schedule (WAR-correct):
//   depth 0: copy(a0,a1), copy(b0,b1)  -- initial copies
//   depth 1: tanh(0,2)
//   depth 2: add(0+1, 2+3)             -- first add reads dst1/dst3
//   depth 3: copy(c0->dst1, c1->dst3)  -- AFTER first add (WAR)
//   depth 4: add(0+1, 2+3)             -- second add

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-WAR-LABEL: func.func @war_hazard_three_input
// CHECK-WAR:       ttkernel.tile_regs_acquire
//
// Depth 0: initial copies for a (cb0) and b (cb1)
// CHECK-WAR:       ttkernel.copy_tile_init(
// CHECK-WAR:       ttkernel.copy_tile(
// CHECK-WAR:       ttkernel.copy_tile(
// CHECK-WAR:       ttkernel.copy_tile_init(
// CHECK-WAR:       ttkernel.copy_tile(
// CHECK-WAR:       ttkernel.copy_tile(
//
// Depth 1: tanh group
// CHECK-WAR:       ttkernel.tanh_tile_init
// CHECK-WAR:       ttkernel.tanh_tile(
// CHECK-WAR:       ttkernel.tanh_tile(
//
// Depth 2: first add (consumes dst1/dst3 from b copies)
// CHECK-WAR:       ttkernel.add_binary_tile_init
// CHECK-WAR:       ttkernel.add_binary_tile(
// CHECK-WAR:       ttkernel.add_binary_tile(
//
// Depth 3: c copies -- MUST be after first add (WAR on dst1/dst3)
// CHECK-WAR:       ttkernel.copy_tile_init(
// CHECK-WAR:       ttkernel.copy_tile(
// CHECK-WAR:       ttkernel.copy_tile(
//
// Depth 4: second add
// CHECK-WAR:       ttkernel.add_binary_tile_init
// CHECK-WAR:       ttkernel.add_binary_tile(
// CHECK-WAR:       ttkernel.add_binary_tile(
//
// CHECK-WAR:       ttkernel.tile_regs_commit
func.func @war_hazard_three_input(
    %a: tensor<2x1x!ttcore.tile<32x32, bf16>>,
    %b: tensor<2x1x!ttcore.tile<32x32, bf16>>,
    %c: tensor<2x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<2x1x!ttcore.tile<32x32, bf16>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %output = tensor.empty() : tensor<2x1x!ttcore.tile<32x32, bf16>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>

  %a_ready = ttl.cb_wait %cb0 : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %b_ready = ttl.cb_wait %cb1 : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %c_ready = ttl.cb_wait %cb2 : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %output_cb = ttl.attach_cb %output, %cb3 : (tensor<2x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>

  %result_view = ttl.cb_reserve %cb3 : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>

  %result = ttl.compute
      ins(%a_ready, %b_ready, %c_ready : tensor<2x1x!ttcore.tile<32x32, bf16>>,
                                          tensor<2x1x!ttcore.tile<32x32, bf16>>,
                                          tensor<2x1x!ttcore.tile<32x32, bf16>>)
      outs(%output_cb : tensor<2x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, bf16>,
       %b_tile: !ttcore.tile<32x32, bf16>,
       %c_tile: !ttcore.tile<32x32, bf16>,
       %out_tile: !ttcore.tile<32x32, bf16>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %t = ttl.tile_tanh %a_tile : !ttcore.tile<32x32, bf16>
    %sum1 = ttl.tile_add %t, %b_tile : !ttcore.tile<32x32, bf16>
    %sum2 = ttl.tile_add %sum1, %c_tile : !ttcore.tile<32x32, bf16>
    ttl.tile_store %sum2, %result_view[%i, %j] : !ttcore.tile<32x32, bf16>, tensor<2x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<2x1x!ttcore.tile<32x32, bf16>>

  ttl.cb_push %cb3 : <[2, 1], !ttcore.tile<32x32, bf16>, 1>

  func.return %result : tensor<2x1x!ttcore.tile<32x32, bf16>>
}
