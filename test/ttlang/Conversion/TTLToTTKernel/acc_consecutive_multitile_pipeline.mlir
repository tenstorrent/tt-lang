// Summary: Two consecutive acc=true computes targeting a 2x2 output view through
// the full pipeline. The first compute packs without L1 accumulation (overwrite),
// the second wraps pack_tile with pack_reconfig_l1_acc(1)/pack_reconfig_l1_acc(0).
// No DST accumulation (fill_tile, add_binary_tile) is emitted.
//
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-form-accumulation-groups{maximize-dst=0}, ttl-set-compute-kernel-config, ttl-assign-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @acc_consecutive_multitile
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG:   %[[C1_I32:.*]] = arith.constant 1 : i32
//
// No DST accumulation ops anywhere.
// CHECK-NOT:   ttkernel.fill_tile
// CHECK-NOT:   ttkernel.add_binary_tile
//
// First compute: copy + pack, no pack_reconfig_l1_acc.
// CHECK:       ttkernel.copy_tile_init
// CHECK:       ttkernel.copy_tile
// CHECK:       ttkernel.tile_regs_commit
// CHECK-NEXT:  ttkernel.tile_regs_wait
// CHECK-NOT:   ttkernel.pack_reconfig_l1_acc
// CHECK:       ttkernel.pack_tile(%[[C0]],
// CHECK-NOT:   ttkernel.pack_reconfig_l1_acc
// CHECK:       ttkernel.tile_regs_release
//
// Second compute: copy + pack wrapped by pack_reconfig_l1_acc(1) / (0).
// CHECK:       ttkernel.copy_tile_init
// CHECK:       ttkernel.copy_tile
// CHECK:       ttkernel.tile_regs_commit
// CHECK-NEXT:  ttkernel.tile_regs_wait
// CHECK-NEXT:  ttkernel.pack_reconfig_l1_acc(%[[C1_I32]])
// CHECK-NEXT:  ttkernel.pack_tile(%[[C0]],
// CHECK-NEXT:  ttkernel.pack_reconfig_l1_acc(%[[C0_I32]])
// CHECK-NEXT:  ttkernel.tile_regs_release

func.func @acc_consecutive_multitile(%a: tensor<2x2x!ttcore.tile<32x32, bf16>>,
                                      %b: tensor<2x2x!ttcore.tile<32x32, bf16>>)
    -> tensor<2x2x!ttcore.tile<32x32, bf16>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %out_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %r0 = ttl.compute
      ins(%a_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, bf16>,
       %out_tile: !ttcore.tile<32x32, bf16>):
    %i0 = ttl.iter_index 0 : index
    %j0 = ttl.iter_index 1 : index
    ttl.tile_store %a_tile, %out_view[%i0, %j0] {acc = true} : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %r1 = ttl.compute
      ins(%b_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%b_tile: !ttcore.tile<32x32, bf16>,
       %out_tile: !ttcore.tile<32x32, bf16>):
    %i1 = ttl.iter_index 0 : index
    %j1 = ttl.iter_index 1 : index
    ttl.tile_store %b_tile, %out_view[%i1, %j1] {acc = true} : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  func.return %r1 : tensor<2x2x!ttcore.tile<32x32, bf16>>
}
