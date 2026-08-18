// Conversion tests for ttl.read_index in data-movement and compute threads,
// including floating-point decoding and unsigned subword extraction.
// RUN: ttlang-opt --convert-ttl-to-ttkernel --split-input-file %s | FileCheck %s

// Convert an f32 element by extracting its exponent and significand from i32.
// CHECK-LABEL: func.func @read_index_f32
// CHECK-DAG: %[[F32_MANTISSA_WIDTH:.*]] = arith.constant 23 : i32
// CHECK-DAG: %[[F32_EXPONENT_MASK:.*]] = arith.constant 255 : i32
// CHECK-DAG: %[[F32_EXPONENT_BIAS:.*]] = arith.constant 127 : i32
// CHECK-DAG: %[[F32_MANTISSA_MASK:.*]] = arith.constant 8388607 : i32
// CHECK-DAG: %[[F32_HIDDEN_BIT:.*]] = arith.constant 8388608 : i32
// CHECK: %[[BITS:.*]] = ttkernel.load_from_l1({{.*}}) : (!ttkernel.l1_addr_ptr, i32) -> i32
// CHECK-NEXT: %[[EXP_BITS:.*]] = arith.shrui %[[BITS]], %[[F32_MANTISSA_WIDTH]] : i32
// CHECK-NEXT: %[[EXP_FIELD:.*]] = arith.andi %[[EXP_BITS]], %[[F32_EXPONENT_MASK]] : i32
// CHECK-NEXT: %[[EXP:.*]] = arith.subi %[[EXP_FIELD]], %[[F32_EXPONENT_BIAS]] : i32
// CHECK-NEXT: %[[MANTISSA:.*]] = arith.andi %[[BITS]], %[[F32_MANTISSA_MASK]] : i32
// CHECK-NEXT: %[[SIGNIFICAND:.*]] = arith.ori %[[MANTISSA]], %[[F32_HIDDEN_BIT]] : i32
// CHECK: arith.cmpi sge, %[[EXP]], %[[F32_MANTISSA_WIDTH]] : i32
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
// CHECK-DAG: %[[BF16_MANTISSA_WIDTH:.*]] = arith.constant 7 : i32
// CHECK-DAG: %[[BF16_EXPONENT_MASK:.*]] = arith.constant 255 : i32
// CHECK-DAG: %[[BF16_MASK_AND_BIAS:.*]] = arith.constant 127 : i32
// CHECK-DAG: %[[BF16_HIDDEN_BIT:.*]] = arith.constant 128 : i32
// CHECK: %[[BITS16:.*]] = ttkernel.load_from_l1({{.*}}) : (!ttkernel.l1_addr_ptr<16>, i32) -> i16
// CHECK-NEXT: %[[BITS32:.*]] = arith.extui %[[BITS16]] : i16 to i32
// CHECK-NEXT: %[[BF16_EXP_BITS:.*]] = arith.shrui %[[BITS32]], %[[BF16_MANTISSA_WIDTH]] : i32
// CHECK-NEXT: %[[BF16_EXP_FIELD:.*]] = arith.andi %[[BF16_EXP_BITS]], %[[BF16_EXPONENT_MASK]] : i32
// CHECK-NEXT: %[[BF16_EXP:.*]] = arith.subi %[[BF16_EXP_FIELD]], %[[BF16_MASK_AND_BIAS]] : i32
// CHECK-NEXT: %[[BF16_MANTISSA:.*]] = arith.andi %[[BITS32]], %[[BF16_MASK_AND_BIAS]] : i32
// CHECK-NEXT: %[[BF16_SIGNIFICAND:.*]] = arith.ori %[[BF16_MANTISSA]], %[[BF16_HIDDEN_BIT]] : i32
// CHECK: arith.cmpi sge, %[[BF16_EXP]], %[[BF16_MANTISSA_WIDTH]] : i32
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

// -----

// Compute reads distribute one packed DFB word to all three compute
// processors before decoding the scalar index.
// CHECK-LABEL: func.func @read_index_compute_f32
// CHECK: %[[F32_CB:.*]] = ttkernel.get_compile_time_arg_val(10) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, f32>>
// CHECK: %[[F32_DFB_ID:.*]] = ttkernel.get_dfb_id %[[F32_CB]] : <2, !ttcore.tile<32x32, f32>>
// CHECK: %[[F32_PACKED:.*]] = ttkernel.read_tile_value(%[[F32_DFB_ID]], %{{.*}}, %{{.*}}) : (i32, i32, i32) -> i32
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: arith.index_cast %{{.*}} : i32 to index
module {
  func.func @read_index_compute_f32() -> index
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 10, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %row = arith.constant 0 : index
    %column = arith.constant 5 : index
    %index = ttl.read_index %block[%row, %column] : tensor<1x1x!ttcore.tile<32x32, f32>> -> index
    func.return %index : index
  }
}

// -----

// CHECK-LABEL: func.func @read_index_compute_bf16
// CHECK: %[[BF16_CB:.*]] = ttkernel.get_compile_time_arg_val(11) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
// CHECK: %[[BF16_DFB_ID:.*]] = ttkernel.get_dfb_id %[[BF16_CB]] : <2, !ttcore.tile<32x32, bf16>>
// CHECK: %[[BF16_PACKED:.*]] = ttkernel.read_tile_value(%[[BF16_DFB_ID]], %{{.*}}, %{{.*}}) : (i32, i32, i32) -> i32
// CHECK: %[[BF16_SHIFTED:.*]] = arith.shrui %[[BF16_PACKED]], %{{.*}} : i32
// CHECK: %[[BF16_MASKED:.*]] = arith.andi %[[BF16_SHIFTED]], %{{.*}} : i32
// CHECK: %[[BF16_BITS:.*]] = arith.trunci %[[BF16_MASKED]] : i32 to i16
// CHECK: %[[BF16_EXTENDED:.*]] = arith.extui %[[BF16_BITS]] : i16 to i32
// CHECK: arith.index_cast %{{.*}} : i32 to index
module {
  func.func @read_index_compute_bf16() -> index
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 11, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %row = arith.constant 0 : index
    %column = arith.constant 3 : index
    %index = ttl.read_index %block[%row, %column] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> index
    func.return %index : index
  }
}

// -----

// CHECK-LABEL: func.func @read_index_compute_ui8
// CHECK: %[[UI8_CB:.*]] = ttkernel.get_compile_time_arg_val(12) : () -> !ttkernel.cb<2, !ttcore.tile<1x32, u8>>
// CHECK: %[[UI8_DFB_ID:.*]] = ttkernel.get_dfb_id %[[UI8_CB]] : <2, !ttcore.tile<1x32, u8>>
// CHECK: %[[UI8_PACKED:.*]] = ttkernel.read_tile_value(%[[UI8_DFB_ID]], %{{.*}}, %{{.*}}) : (i32, i32, i32) -> i32
// CHECK: %[[UI8_SHIFTED:.*]] = arith.shrui %[[UI8_PACKED]], %{{.*}} : i32
// CHECK: %[[UI8_MASKED:.*]] = arith.andi %[[UI8_SHIFTED]], %{{.*}} : i32
// CHECK: %[[UI8_BITS:.*]] = arith.trunci %[[UI8_MASKED]] : i32 to i8
// CHECK: %[[UI8_EXTENDED:.*]] = arith.extui %[[UI8_BITS]] : i8 to i32
// CHECK-NEXT: %{{.*}} = arith.index_cast %[[UI8_EXTENDED]] : i32 to index
module {
  func.func @read_index_compute_ui8() -> index
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 12, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<1x32, u8>, 2>
    %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<1x32, u8>, 2> -> tensor<1x1x!ttcore.tile<1x32, u8>>
    %row = arith.constant 0 : index
    %column = arith.constant 7 : index
    %index = ttl.read_index %block[%row, %column] : tensor<1x1x!ttcore.tile<1x32, u8>> -> index
    func.return %index : index
  }
}

// -----

// CHECK-LABEL: func.func @read_index_compute_ui16
// CHECK: %[[UI16_CB:.*]] = ttkernel.get_compile_time_arg_val(13) : () -> !ttkernel.cb<2, !ttcore.tile<1x32, u16>>
// CHECK: %[[UI16_DFB_ID:.*]] = ttkernel.get_dfb_id %[[UI16_CB]] : <2, !ttcore.tile<1x32, u16>>
// CHECK: %[[UI16_PACKED:.*]] = ttkernel.read_tile_value(%[[UI16_DFB_ID]], %{{.*}}, %{{.*}}) : (i32, i32, i32) -> i32
// CHECK: %[[UI16_SHIFTED:.*]] = arith.shrui %[[UI16_PACKED]], %{{.*}} : i32
// CHECK: %[[UI16_MASKED:.*]] = arith.andi %[[UI16_SHIFTED]], %{{.*}} : i32
// CHECK: %[[UI16_BITS:.*]] = arith.trunci %[[UI16_MASKED]] : i32 to i16
// CHECK: %[[UI16_EXTENDED:.*]] = arith.extui %[[UI16_BITS]] : i16 to i32
// CHECK-NEXT: %{{.*}} = arith.index_cast %[[UI16_EXTENDED]] : i32 to index
module {
  func.func @read_index_compute_ui16() -> index
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 13, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<1x32, u16>, 2>
    %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<1x32, u16>, 2> -> tensor<1x1x!ttcore.tile<1x32, u16>>
    %row = arith.constant 0 : index
    %column = arith.constant 7 : index
    %index = ttl.read_index %block[%row, %column] : tensor<1x1x!ttcore.tile<1x32, u16>> -> index
    func.return %index : index
  }
}

// -----

// CHECK-LABEL: func.func @read_index_compute_ui32
// CHECK: %[[UI32_CB:.*]] = ttkernel.get_compile_time_arg_val(14) : () -> !ttkernel.cb<2, !ttcore.tile<1x32, u32>>
// CHECK: %[[UI32_DFB_ID:.*]] = ttkernel.get_dfb_id %[[UI32_CB]] : <2, !ttcore.tile<1x32, u32>>
// CHECK: %[[UI32_PACKED:.*]] = ttkernel.read_tile_value(%[[UI32_DFB_ID]], %{{.*}}, %{{.*}}) : (i32, i32, i32) -> i32
// CHECK-NOT: ttkernel.load_from_l1
// CHECK-NEXT: %{{.*}} = arith.index_cast %[[UI32_PACKED]] : i32 to index
module {
  func.func @read_index_compute_ui32() -> index
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 14, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<1x32, u32>, 2>
    %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<1x32, u32>, 2> -> tensor<1x1x!ttcore.tile<1x32, u32>>
    %row = arith.constant 0 : index
    %column = arith.constant 9 : index
    %index = ttl.read_index %block[%row, %column] : tensor<1x1x!ttcore.tile<1x32, u32>> -> index
    func.return %index : index
  }
}
