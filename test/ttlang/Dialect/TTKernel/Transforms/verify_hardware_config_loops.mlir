// RUN: ttlang-opt %s --split-input-file --ttkernel-verify-hardware-config | FileCheck %s
// Summary: Static trip counts of 0 and 1 are not treated as looping fixed
// points. Count 0 does not check the body. Count 1 does not join the backedge.

// CHECK-LABEL: func.func @trip_count_one
func.func @trip_count_one() {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ttkernel.exp_tile_init() : () -> ()
  scf.for %i = %c0 to %c1 step %c1 {
    ttkernel.exp_tile(%c0) : (index) -> ()
    ttkernel.add_tiles_init(%cb, %cb) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>) -> ()
    ttkernel.add_tiles(%cb, %cb, %c0, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
  }
  func.return
}

// -----

// CHECK-LABEL: func.func @trip_count_zero
func.func @trip_count_zero() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c0 step %c1 {
    ttkernel.exp_tile(%c0) : (index) -> ()
  }
  ttkernel.exp_tile_init() : () -> ()
  ttkernel.exp_tile(%c0) : (index) -> ()
  func.return
}
