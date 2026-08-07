// Conversion tests for integer-only ttl.read_index lowering from f32 and bf16
// IEEE-754 storage to an index value.
// RUN: ttlang-opt --convert-ttl-to-ttkernel --canonicalize -cse --split-input-file %s | FileCheck %s

// Convert an f32 element by extracting its exponent and significand from i32.
// CHECK-LABEL: func.func @read_index_f32
// CHECK: %[[BITS:.*]] = ttkernel.load_from_l1({{.*}}) : (!ttkernel.l1_addr_ptr, i32) -> i32
// CHECK-NEXT: %[[EXP_BITS:.*]] = arith.shrui %[[BITS]],
// CHECK: %[[EXP:.*]] = arith.andi %[[EXP_BITS]],
// CHECK: %[[SIGNIFICAND:.*]] = arith.andi %[[BITS]],
// CHECK: %[[MAGNITUDE:.*]] = arith.select
// CHECK: %[[INTEGER:.*]] = arith.select {{.*}}, {{.*}}, %[[MAGNITUDE]] : i32
// CHECK-NEXT: %[[INDEX:.*]] = arith.index_cast %[[INTEGER]] : i32 to index
// CHECK-NEXT: return %[[INDEX]] : index
// CHECK-NOT: arith.fptosi
module {
  func.func @read_index_f32() -> index
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %row = arith.constant 0 : index
    %column = arith.constant 5 : index
    %index = ttl.read_index %block[%row, %column] : tensor<1x1x!ttcore.tile<32x32, f32>> -> index
    func.return %index : index
  }
}

// -----

// Zero-extend bf16 storage before applying the same integer conversion.
// CHECK-LABEL: func.func @read_index_bf16
// CHECK: %[[BITS16:.*]] = ttkernel.load_from_l1({{.*}}) : (!ttkernel.l1_addr_ptr<16>, i32) -> i16
// CHECK-NEXT: %[[BITS32:.*]] = arith.extui %[[BITS16]] : i16 to i32
// CHECK-NEXT: %[[EXP_BITS:.*]] = arith.shrui %[[BITS32]],
// CHECK: %[[SIGNIFICAND:.*]] = arith.andi %[[BITS32]],
// CHECK: %[[MAGNITUDE:.*]] = arith.select
// CHECK: %[[INTEGER:.*]] = arith.select {{.*}}, {{.*}}, %[[MAGNITUDE]] : i32
// CHECK-NEXT: %[[INDEX:.*]] = arith.index_cast %[[INTEGER]] : i32 to index
// CHECK-NEXT: return %[[INDEX]] : index
// CHECK-NOT: arith.fptosi
module {
  func.func @read_index_bf16() -> index
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %row = arith.constant 0 : index
    %column = arith.constant 1 : index
    %index = ttl.read_index %block[%row, %column] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> index
    func.return %index : index
  }
}
