// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-set-compute-kernel-config{enable-fpu-binary-ops=0 matmul-full-fp32=0 reduce-full-fp32=0}, func.func(ttl-assign-dst, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' | FileCheck %s
// Summary: Verifies DST-resident additive recurrence lowering to TTKernel.

// The accumulator is copied into DST once before the reduction loop.
// Each reduction iteration uses binary_dest_reuse_tiles to add the
// contribution directly from the DFB into the accumulator DST slot. The result
// is packed once after the reduction loop.

#map_acc_init = affine_map<(d0, d1) -> (d0)>
#map_acc_contrib = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @accumulate_add_reduction
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[INIT_CB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[CONTRIB_CB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: %[[OUT_CB:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK: ttkernel.binary_op_init_common(%[[CONTRIB_CB]], %[[CONTRIB_CB]], %[[OUT_CB]])
// CHECK: scf.for %[[I:.*]] =
// CHECK-NEXT: ttkernel.tile_regs_acquire
// CHECK-NEXT: ttkernel.copy_tile_init(%[[INIT_CB]])
// CHECK-NEXT: ttkernel.copy_tile(%[[INIT_CB]], %[[I]], %[[C0]])
// CHECK-NEXT: ttkernel.binary_dest_reuse_tiles_init(%[[CONTRIB_CB]], <add>, <dest_to_srca>)
// CHECK-NEXT: scf.for %[[J:.*]] =
// CHECK-NOT: ttkernel.copy_tile(%[[CONTRIB_CB]]
// CHECK: %[[CONTRIB_IDX:.*]] = affine.linearize_index [%[[I]], %[[J]]] by (2, 3) : index
// CHECK-NEXT: ttkernel.binary_dest_reuse_tiles(%[[CONTRIB_CB]], %[[CONTRIB_IDX]], %[[C0]], <add>, <dest_to_srca>)
// CHECK-NOT: ttkernel.copy_tile(%[[CONTRIB_CB]]
// CHECK: ttkernel.tile_regs_commit
// CHECK-NEXT: ttkernel.tile_regs_wait
// CHECK-NEXT: ttkernel.pack_tile(%[[C0]], %[[OUT_CB]], %[[I]], true)
// CHECK-NEXT: ttkernel.tile_regs_release
func.func @accumulate_add_reduction(
    %init_arg: tensor<2x!ttcore.tile<32x32, f32>>,
    %contrib_arg: tensor<2x3x!ttcore.tile<32x32, f32>>) -> tensor<2x!ttcore.tile<32x32, f32>> {
  %out_init = tensor.empty() : tensor<2x!ttcore.tile<32x32, f32>>
  %cbinit = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>
  %cbcontrib = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>
  %init_att = ttl.attach_cb %init_arg, %cbinit : (tensor<2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x!ttcore.tile<32x32, f32>>
  %contrib_att = ttl.attach_cb %contrib_arg, %cbcontrib : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %out_init_att = ttl.attach_cb %out_init, %cbout : (tensor<2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x!ttcore.tile<32x32, f32>>
  %out_view = ttl.cb_reserve %cbout : <[1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute ins(%init_att, %contrib_att : tensor<2x!ttcore.tile<32x32, f32>>, tensor<2x3x!ttcore.tile<32x32, f32>>) outs(%out_init_att : tensor<2x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map_acc_init, #map_acc_contrib, #map_acc_init], iterator_types = ["parallel", "reduction"]} {
  ^bb0(%init_tile: !ttcore.tile<32x32, f32>, %contrib_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %c0 = arith.constant 0 : index
    %acc_token, %acc = ttl.copy_tile %init_tile[%i] into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttl.dst, !ttcore.tile<32x32, f32>
    %next = ttl.tile_accumulate %acc, %contrib_tile add into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %next, %out_view[%i] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x!ttcore.tile<32x32, f32>>
}
