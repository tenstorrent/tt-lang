// RUN: ttlang-opt --convert-ttl-to-ttkernel --ttkernel-insert-inits %s | FileCheck %s
// Summary: ttl.tile_exp hardware flags are forwarded to ttkernel.exp_tile, and
// the matching ttkernel.exp_tile_init (carrying approx / scale / input_clamping)
// is emitted by the init-insertion pass from the flags read off exp_tile.
//
// TTKernel exp ops do not define typed flag properties, so the lowering carries
// the flags as ordinary MLIR attributes for init insertion and EmitC lowering.

// A single flagged exp: one exp_tile_init then the exp_tile.
// CHECK-LABEL: func.func @tile_exp_flags
// CHECK: ttkernel.tile_regs_acquire
// CHECK: ttkernel.exp_tile_init() {{[{].*}}approx = true{{.*}}input_clamping = #ttkernel.input_clamping<none>{{.*}}scale = 1073741824 : i32{{.*[}]}}
// CHECK: ttkernel.exp_tile({{.*}}) {{[{].*}}approx = true{{.*}}input_clamping = #ttkernel.input_clamping<none>{{.*}}iterations = 4 : i32{{.*}}scale = 1073741824 : i32{{.*[}]}}
// CHECK: ttkernel.tile_regs_release
func.func @tile_exp_flags(%a: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  %c0 = arith.constant 0 : index
  ttkernel.tile_regs_acquire() : () -> ()
  %exp = ttl.tile_exp %a into dst[%c0] {approx = true, scale = 2.000000e+00 : f32,
                                        input_clamping = 0 : i32,
                                        iterations = 4 : i32}
      : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  ttkernel.tile_regs_release() : () -> ()
  func.return %exp : !ttcore.tile<32x32, f32>
}

// -----

// Two exps with identical flags share a single init; a third with different
// flags forces a fresh init.
// CHECK-LABEL: func.func @tile_exp_flag_change
// CHECK: ttkernel.exp_tile_init
// CHECK: ttkernel.exp_tile
// CHECK-NOT: ttkernel.exp_tile_init
// CHECK: ttkernel.exp_tile
// CHECK: ttkernel.exp_tile_init
// CHECK: ttkernel.exp_tile
func.func @tile_exp_flag_change(%a: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  ttkernel.tile_regs_acquire() : () -> ()
  %e0 = ttl.tile_exp %a into dst[%c0] {approx = true}
      : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  %e1 = ttl.tile_exp %a into dst[%c1] {approx = true}
      : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  %e2 = ttl.tile_exp %a into dst[%c2] {approx = false}
      : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  ttkernel.tile_regs_release() : () -> ()
  func.return %e2 : !ttcore.tile<32x32, f32>
}
