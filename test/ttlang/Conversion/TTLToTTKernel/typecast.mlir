// RUN: ttlang-opt --convert-ttl-to-ttkernel --ttkernel-insert-inits %s | FileCheck %s
// Summary: Tests for ttl.tile_typecast lowering to TTKernel typecast_tile and
// init insertion of typecast_tile_init carrying both in/out dtype attributes.

// CHECK-LABEL: func.func @tile_typecast_bf16_to_f32
// CHECK: ttkernel.tile_regs_acquire
// CHECK: ttkernel.typecast_tile_init(<bf16>, <f32>)
// CHECK: ttkernel.typecast_tile(%{{.*}}, <bf16>, <f32>)
// CHECK: ttkernel.tile_regs_release
func.func @tile_typecast_bf16_to_f32(%a: !ttcore.tile<32x32, bf16>) {
  %c0 = arith.constant 0 : index
  ttkernel.tile_regs_acquire() : () -> ()
  %0 = ttl.tile_typecast %a into dst[%c0]
       : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, f32>
  ttkernel.tile_regs_release() : () -> ()
  return
}

// -----

// Two consecutive typecasts with the SAME (in_dtype, out_dtype) pair share a
// single init op. A third typecast with a DIFFERENT dtype pair gets its own
// init.

// CHECK-LABEL: func.func @tile_typecast_init_dedup
// CHECK: ttkernel.tile_regs_acquire
// First and second typecast share dtype pair (bf16, f32) -> single init.
// CHECK: ttkernel.typecast_tile_init(<bf16>, <f32>)
// CHECK-NEXT: ttkernel.typecast_tile
// CHECK-NOT: ttkernel.typecast_tile_init(<bf16>, <f32>)
// CHECK: ttkernel.typecast_tile
// Third typecast has different dtype pair (f32, bf16) -> new init.
// CHECK: ttkernel.typecast_tile_init(<f32>, <bf16>)
// CHECK-NEXT: ttkernel.typecast_tile
// CHECK: ttkernel.tile_regs_release
func.func @tile_typecast_init_dedup(%a: !ttcore.tile<32x32, bf16>,
                                     %b: !ttcore.tile<32x32, bf16>,
                                     %c: !ttcore.tile<32x32, f32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  ttkernel.tile_regs_acquire() : () -> ()
  %0 = ttl.tile_typecast %a into dst[%c0]
       : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, f32>
  %1 = ttl.tile_typecast %b into dst[%c1]
       : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, f32>
  %2 = ttl.tile_typecast %c into dst[%c2]
       : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, bf16>
  ttkernel.tile_regs_release() : () -> ()
  return
}

// -----

// The typecast result can feed a chained SFPU op. Both ops use the same DST
// slot because typecast is in-place.
// CHECK-LABEL: func.func @tile_typecast_result_consumed_by_sfpu
// CHECK: ttkernel.tile_regs_acquire
// CHECK: ttkernel.typecast_tile_init(<bf16>, <f32>)
// CHECK-NEXT: ttkernel.typecast_tile
// CHECK: ttkernel.exp_tile_init
// CHECK-NEXT: ttkernel.exp_tile
// CHECK: ttkernel.tile_regs_release
func.func @tile_typecast_result_consumed_by_sfpu(
    %a: !ttcore.tile<32x32, bf16>) {
  %c0 = arith.constant 0 : index
  ttkernel.tile_regs_acquire() : () -> ()
  %0 = ttl.tile_typecast %a into dst[%c0]
       : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, f32>
  %1 = ttl.tile_exp %0 into dst[%c0]
       : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  ttkernel.tile_regs_release() : () -> ()
  return
}
