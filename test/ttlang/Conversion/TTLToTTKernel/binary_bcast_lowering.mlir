// Verify ttl.tile_binary_bcast lowers to ttkernel.binary_bcast (the FPU
// broadcast-binary op that reads BOTH operands from CBs and writes DST) and
// that ttkernel-insert-inits emits the matching, hoisted binary_bcast_init.
//
// RUN: ttlang-opt %s --split-input-file \
// RUN:   -pass-pipeline='builtin.module(convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s

// mul + column broadcast (the OPT8 rewrite): C[i,j] = data[i,j] * bcast_col(alpha[i,0]).
// in0 = data (full tile, index linearized [row, col]); in1 = alpha (broadcast
// source, index = %row). The init is hoisted above the loops.
//
// CHECK-LABEL: func.func @binary_bcast_mul_col
// CHECK: ttkernel.binary_bcast_init(%{{.*}}, %{{.*}}, %{{.*}}, <mul>, <col>)
// CHECK: scf.for %[[ROW:.*]] = %{{.*}} to %{{.*}}
// CHECK:   scf.for %[[COL:.*]] = %{{.*}} to %{{.*}}
// CHECK:     ttkernel.binary_bcast(%{{.*}}, %{{.*}}, %{{.*}}, %[[ROW]], %{{.*}}, <mul>, <col>)
func.func @binary_bcast_mul_col()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb_data = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb_alpha = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 1>
  %cb_out = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>

  %data_in = ttl.cb_wait %cb_data : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %data_cb = ttl.attach_cb %data_in, %cb_data : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %alpha_in = ttl.cb_wait %cb_alpha : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %alpha_cb = ttl.attach_cb %alpha_in, %cb_alpha : (tensor<2x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %view = ttl.cb_reserve %cb_out : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %view_cb = ttl.attach_cb %view, %cb_out : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  scf.for %row = %c0 to %c2 step %c1 {
    scf.for %col = %c0 to %c2 step %c1 {
      ttl.tile_regs_acquire
      %data_tile = tensor.extract %data_cb[%row, %col] : tensor<2x2x!ttcore.tile<32x32, f32>>
      %alpha_tile = tensor.extract %alpha_cb[%row, %c0] : tensor<2x1x!ttcore.tile<32x32, f32>>
      %out_tile = tensor.extract %view_cb[%row, %col] : tensor<2x2x!ttcore.tile<32x32, f32>>
      %res = ttl.tile_binary_bcast %data_tile, %alpha_tile, %out_tile <mul> 1 : i32 into dst[%c0] {ttl.bcast_output_cb_index = 2 : index}
          : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>)
          -> !ttcore.tile<32x32, f32>
      ttl.tile_store %res, %view[%row, %col] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
      ttl.tile_regs_commit
      ttl.tile_regs_wait
      ttl.tile_regs_release
    } {ttl.tile_loop_stride = 1 : index}
  } {ttl.tile_loop_stride = 2 : index}

  ttl.cb_pop %cb_data : <[2, 2], !ttcore.tile<32x32, f32>, 1>
  ttl.cb_pop %cb_alpha : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  ttl.cb_push %cb_out : <[2, 2], !ttcore.tile<32x32, f32>, 1>
  func.return
}
