// Conversion tests for ttl.raw_element_read and ttl.raw_element_write lowering
// to ttkernel.load_from_l1 / ttkernel.store_to_l1 with pointer arithmetic.
// Covers tiled (face-order) and row-major layouts, f32 and bf16 types,
// static and dynamic coordinates, and read-write chaining.
// RUN: ttlang-opt --convert-ttl-to-ttkernel --canonicalize -cse --split-input-file %s | FileCheck %s

// -----

// Read f32 from tiled block at (0,5) via cb_wait -> get_read_ptr.
// Offset = face0, row0, col5 = 5.
// CHECK-LABEL: func.func @read_tiled_f32_origin
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : i32
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: ttkernel.cb_wait_front(%[[CB]],
// CHECK: %[[PTR:.*]] = ttkernel.get_read_ptr(%[[CB]])
// CHECK: %[[L1:.*]] = ttkernel.reinterpret_cast<tt_l1_ptr uint32_t*>(%[[PTR]]) : (i32) -> !ttkernel.l1_addr_ptr
// CHECK: ttkernel.load_from_l1(%[[L1]], %[[C5]]) : (!ttkernel.l1_addr_ptr, i32) -> i32
module {
  func.func @read_tiled_f32_origin()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %val = ttl.raw_element_read %block[%c0, %c5] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
    func.return
  }
}

// -----

// Read bf16 from tiled block at (0,1) via cb_wait.
// Offset = 1. Pointer element width = 16.
// CHECK-LABEL: func.func @read_tiled_bf16
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[L1:.*]] = ttkernel.reinterpret_cast<tt_l1_ptr uint32_t*>({{.*}}) : (i32) -> !ttkernel.l1_addr_ptr<16>
// CHECK: ttkernel.load_from_l1(%[[L1]], %[[C1]]) : (!ttkernel.l1_addr_ptr<16>, i32) -> i16
module {
  func.func @read_tiled_bf16()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_wait %cb : <[2, 3], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x3x!ttcore.tile<32x32, bf16>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %val = ttl.raw_element_read %block[%c0, %c1] : tensor<2x3x!ttcore.tile<32x32, bf16>> -> bf16
    func.return
  }
}

// -----

// Read f32 at face1 origin (0, 16) -> offset 256.
// CHECK-LABEL: func.func @read_tiled_face1
// CHECK-DAG: %[[C256:.*]] = arith.constant 256 : i32
// CHECK: %[[L1:.*]] = ttkernel.reinterpret_cast<tt_l1_ptr uint32_t*>({{.*}}) : (i32) -> !ttkernel.l1_addr_ptr
// CHECK: ttkernel.load_from_l1(%[[L1]], %[[C256]])
module {
  func.func @read_tiled_face1()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %val = ttl.raw_element_read %block[%c0, %c16] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
    func.return
  }
}

// -----

// Read f32 at face2 origin (16, 0) -> offset 512.
// CHECK-LABEL: func.func @read_tiled_face2
// CHECK-DAG: %[[C512:.*]] = arith.constant 512 : i32
// CHECK: ttkernel.load_from_l1({{.*}}, %[[C512]])
module {
  func.func @read_tiled_face2()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %val = ttl.raw_element_read %block[%c16, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
    func.return
  }
}

// -----

// Read f32 at face3 origin (16, 16) -> offset 768.
// CHECK-LABEL: func.func @read_tiled_face3
// CHECK-DAG: %[[C768:.*]] = arith.constant 768 : i32
// CHECK: ttkernel.load_from_l1({{.*}}, %[[C768]])
module {
  func.func @read_tiled_face3()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %c16 = arith.constant 16 : index
    %val = ttl.raw_element_read %block[%c16, %c16] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
    func.return
  }
}

// -----

// Read f32 at tile corner (31, 31) -> face3, localRow=15, localCol=15.
// offset = 3*256 + 15*16 + 15 = 768 + 240 + 15 = 1023.
// CHECK-LABEL: func.func @read_tiled_last_elem
// CHECK-DAG: %[[C1023:.*]] = arith.constant 1023 : i32
// CHECK: ttkernel.load_from_l1({{.*}}, %[[C1023]])
module {
  func.func @read_tiled_last_elem()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %c31 = arith.constant 31 : index
    %val = ttl.raw_element_read %block[%c31, %c31] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
    func.return
  }
}

// -----

// Read bf16 at a face boundary (15, 15) -> face0 last elem, offset 255.
// CHECK-LABEL: func.func @read_tiled_bf16_face_boundary
// CHECK-DAG: %[[C255:.*]] = arith.constant 255 : i32
// CHECK: %[[L1:.*]] = ttkernel.reinterpret_cast<tt_l1_ptr uint32_t*>({{.*}}) : (i32) -> !ttkernel.l1_addr_ptr<16>
// CHECK: ttkernel.load_from_l1(%[[L1]], %[[C255]]) : (!ttkernel.l1_addr_ptr<16>, i32) -> i16
module {
  func.func @read_tiled_bf16_face_boundary()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %c15 = arith.constant 15 : index
    %val = ttl.raw_element_read %block[%c15, %c15] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> bf16
    func.return
  }
}

// -----

// Read f32 from row-major block at (1, 3) -> offset = 1*8 + 3 = 11.
// CHECK-LABEL: func.func @read_row_major_f32
// CHECK-DAG: %[[C11:.*]] = arith.constant 11 : i32
// CHECK: %[[L1:.*]] = ttkernel.reinterpret_cast<tt_l1_ptr uint32_t*>({{.*}}) : (i32) -> !ttkernel.l1_addr_ptr
// CHECK: ttkernel.load_from_l1(%[[L1]], %[[C11]]) : (!ttkernel.l1_addr_ptr, i32) -> i32
module {
  func.func @read_row_major_f32()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 8], f32, 2>
    %block = ttl.cb_wait %cb : <[4, 8], f32, 2> -> tensor<4x8xf32>
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %val = ttl.raw_element_read %block[%c1, %c3] : tensor<4x8xf32> -> f32
    func.return
  }
}

// -----

// Read bf16 from row-major block at (2, 7) -> offset = 2*16 + 7 = 39.
// CHECK-LABEL: func.func @read_row_major_bf16
// CHECK-DAG: %[[C39:.*]] = arith.constant 39 : i32
// CHECK: %[[L1:.*]] = ttkernel.reinterpret_cast<tt_l1_ptr uint32_t*>({{.*}}) : (i32) -> !ttkernel.l1_addr_ptr<16>
// CHECK: ttkernel.load_from_l1(%[[L1]], %[[C39]]) : (!ttkernel.l1_addr_ptr<16>, i32) -> i16
module {
  func.func @read_row_major_bf16()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[8, 16], bf16, 2>
    %block = ttl.cb_wait %cb : <[8, 16], bf16, 2> -> tensor<8x16xbf16>
    %c2 = arith.constant 2 : index
    %c7 = arith.constant 7 : index
    %val = ttl.raw_element_read %block[%c2, %c7] : tensor<8x16xbf16> -> bf16
    func.return
  }
}

// -----

// Write f32 constant (1.0) to tiled block via cb_reserve -> get_write_ptr.
// 1.0f = 0x3F800000 = 1065353216.
// CHECK-LABEL: func.func @write_tiled_f32_constant
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[BITS:.*]] = arith.constant 1065353216 : i32
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: ttkernel.cb_reserve_back(%[[CB]],
// CHECK: %[[PTR:.*]] = ttkernel.get_write_ptr(%[[CB]])
// CHECK: %[[L1:.*]] = ttkernel.reinterpret_cast<tt_l1_ptr uint32_t*>(%[[PTR]]) : (i32) -> !ttkernel.l1_addr_ptr
// CHECK: ttkernel.store_to_l1(%[[BITS]], %[[L1]], %[[C0]]) : (i32, !ttkernel.l1_addr_ptr, i32) -> ()
module {
  func.func @write_tiled_f32_constant()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %block = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : f32
    ttl.raw_element_write %block[%c0, %c0], %cst : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
    func.return
  }
}

// -----

// Write bf16 constant (1.0) to tiled block.
// 1.0 bf16 = 0x3F80 = 16256.
// CHECK-LABEL: func.func @write_tiled_bf16_constant
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[BITS:.*]] = arith.constant 16256 : i16
// CHECK: %[[L1:.*]] = ttkernel.reinterpret_cast<tt_l1_ptr uint32_t*>({{.*}}) : (i32) -> !ttkernel.l1_addr_ptr<16>
// CHECK: ttkernel.store_to_l1(%[[BITS]], %[[L1]], %[[C0]]) : (i16, !ttkernel.l1_addr_ptr<16>, i32) -> ()
module {
  func.func @write_tiled_bf16_constant()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : bf16
    ttl.raw_element_write %block[%c0, %c0], %cst : tensor<1x1x!ttcore.tile<32x32, bf16>>, bf16
    func.return
  }
}

// -----

// Write f32 to row-major block at (2, 5) -> offset = 2*8 + 5 = 21.
// CHECK-LABEL: func.func @write_row_major_f32
// CHECK-DAG: %[[C21:.*]] = arith.constant 21 : i32
// CHECK-DAG: %[[BITS:.*]] = arith.constant 1065353216 : i32
// CHECK: %[[PTR:.*]] = ttkernel.get_write_ptr({{.*}})
// CHECK: %[[L1:.*]] = ttkernel.reinterpret_cast<tt_l1_ptr uint32_t*>(%[[PTR]]) : (i32) -> !ttkernel.l1_addr_ptr
// CHECK: ttkernel.store_to_l1(%[[BITS]], %[[L1]], %[[C21]])
module {
  func.func @write_row_major_f32()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 8], f32, 2>
    %block = ttl.cb_reserve %cb : <[4, 8], f32, 2> -> tensor<4x8xf32>
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %cst = arith.constant 1.000000e+00 : f32
    ttl.raw_element_write %block[%c2, %c5], %cst : tensor<4x8xf32>, f32
    func.return
  }
}

// -----

// Read-then-write chain: load_from_l1 result feeds store_to_l1 directly.
// The unrealized_conversion_cast between read and write is folded away.
// CHECK-LABEL: func.func @read_write_chain
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[RPTR:.*]] = ttkernel.get_read_ptr(%[[CB]])
// CHECK: %[[RL1:.*]] = ttkernel.reinterpret_cast<tt_l1_ptr uint32_t*>(%[[RPTR]]) : (i32) -> !ttkernel.l1_addr_ptr
// CHECK: %[[VAL:.*]] = ttkernel.load_from_l1(%[[RL1]], {{.*}}) : (!ttkernel.l1_addr_ptr, i32) -> i32
// CHECK: %[[WPTR:.*]] = ttkernel.get_write_ptr(%[[CB]])
// CHECK: %[[WL1:.*]] = ttkernel.reinterpret_cast<tt_l1_ptr uint32_t*>(%[[WPTR]]) : (i32) -> !ttkernel.l1_addr_ptr
// CHECK: ttkernel.store_to_l1(%[[VAL]], %[[WL1]], {{.*}}) : (i32, !ttkernel.l1_addr_ptr, i32) -> ()
module {
  func.func @read_write_chain()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %rblock = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %val = ttl.raw_element_read %rblock[%c0, %c5] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    %wblock = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.raw_element_write %wblock[%c0, %c5], %val : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    func.return
  }
}

// -----

// Read from rank-1 tiled block at flat index 2 -> tileIdx=0, intraFlat=2.
// intraRow=0, intraCol=2 -> face0, offset=2.
// CHECK-LABEL: func.func @read_rank1_tiled
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : i32
// CHECK: ttkernel.load_from_l1({{.*}}, %[[C2]]) : (!ttkernel.l1_addr_ptr, i32) -> i32
module {
  func.func @read_rank1_tiled()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
    %block = ttl.cb_wait %cb : <[4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x!ttcore.tile<32x32, f32>>
    %c2 = arith.constant 2 : index
    %val = ttl.raw_element_read %block[%c2] : tensor<4x!ttcore.tile<32x32, f32>> -> f32
    func.return
  }
}

// -----

// Read from rank-1 row-major block at index 42 -> offset=42.
// CHECK-LABEL: func.func @read_rank1_row_major
// CHECK-DAG: %[[C42:.*]] = arith.constant 42 : i32
// CHECK: ttkernel.load_from_l1({{.*}}, %[[C42]]) : (!ttkernel.l1_addr_ptr, i32) -> i32
module {
  func.func @read_rank1_row_major()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[128], f32, 2>
    %block = ttl.cb_wait %cb : <[128], f32, 2> -> tensor<128xf32>
    %c42 = arith.constant 42 : index
    %val = ttl.raw_element_read %block[%c42] : tensor<128xf32> -> f32
    func.return
  }
}

// -----

// Dynamic coordinates: arith ops for face decomposition are present.
// CHECK-LABEL: func.func @read_tiled_dynamic
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[PTR:.*]] = ttkernel.get_read_ptr(%[[CB]])
// CHECK: %[[L1:.*]] = ttkernel.reinterpret_cast<tt_l1_ptr uint32_t*>(%[[PTR]]) : (i32) -> !ttkernel.l1_addr_ptr
// CHECK: arith.divui
// CHECK: arith.remui
// CHECK: ttkernel.load_from_l1(%[[L1]], {{.*}}) : (!ttkernel.l1_addr_ptr, i32) -> i32
module {
  func.func @read_tiled_dynamic(%row: index, %col: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %val = ttl.raw_element_read %block[%row, %col] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
    func.return
  }
}

// -----

// Read from 3D tiled block at (1, 0, 0).
// tileIdx = 1 * gridShape[1] * gridShape[2] + 0 + 0 = 1 * 3 * 4 = 12.
// intraRow=0, intraCol=0 -> offset = 12 * 1024 = 12288.
// CHECK-LABEL: func.func @read_tiled_3d
// CHECK-DAG: %[[OFF:.*]] = arith.constant 12288 : i32
// CHECK: ttkernel.load_from_l1({{.*}}, %[[OFF]])
module {
  func.func @read_tiled_3d()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 3, 4], !ttcore.tile<32x32, f32>, 2>
    %block = ttl.cb_wait %cb : <[2, 3, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<2x3x4x!ttcore.tile<32x32, f32>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %val = ttl.raw_element_read %block[%c1, %c0, %c0] : tensor<2x3x4x!ttcore.tile<32x32, f32>> -> f32
    func.return
  }
}

// -----

// Ensure no arith.bitcast appears in the lowered output.
// CHECK-LABEL: func.func @no_arith_bitcast
// CHECK-NOT: arith.bitcast
module {
  func.func @no_arith_bitcast()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %c0 = arith.constant 0 : index
    %rblock = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %val = ttl.raw_element_read %rblock[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    %wblock = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.raw_element_write %wblock[%c0, %c0], %val : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    func.return
  }
}

// -----

// Multiple reads from the same block share the L1 pointer.
// CHECK-LABEL: func.func @multiple_reads_same_block
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: ttkernel.cb_wait_front
// CHECK-COUNT-2: ttkernel.load_from_l1
module {
  func.func @multiple_reads_same_block()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %v0 = ttl.raw_element_read %block[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
    %v1 = ttl.raw_element_read %block[%c0, %c1] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
    func.return
  }
}

// -----

// Write f32 value truncated to bf16: materializeIntBits handles arith.truncf
// by extracting the upper 16 bits of the f32 encoding via shift+trunc.
// CHECK-LABEL: func.func @write_tiled_bf16_truncf
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[SHIFTED:.*]] = arith.shrui %arg0, {{.*}} : i32
// CHECK-DAG: %[[TRUNC:.*]] = arith.trunci %[[SHIFTED]] : i32 to i16
// CHECK: %[[L1:.*]] = ttkernel.reinterpret_cast<tt_l1_ptr uint32_t*>({{.*}}) : (i32) -> !ttkernel.l1_addr_ptr<16>
// CHECK: ttkernel.store_to_l1(%[[TRUNC]], %[[L1]], %[[C0]]) : (i16, !ttkernel.l1_addr_ptr<16>, i32) -> ()
module {
  func.func @write_tiled_bf16_truncf(%a_int: i32)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %c0 = arith.constant 0 : index
    %f32val = builtin.unrealized_conversion_cast %a_int : i32 to f32
    %bf16val = arith.truncf %f32val : f32 to bf16
    ttl.raw_element_write %block[%c0, %c0], %bf16val : tensor<1x1x!ttcore.tile<32x32, bf16>>, bf16
    func.return
  }
}

// -----

// Write bf16 constant (2.5) to row-major block at (1, 3) -> offset = 1*8 + 3 = 11.
// 2.5 bf16 = 0x4020 = 16416.
// CHECK-LABEL: func.func @write_row_major_bf16
// CHECK-DAG: %[[C11:.*]] = arith.constant 11 : i32
// CHECK-DAG: %[[BITS:.*]] = arith.constant 16416 : i16
// CHECK: %[[L1:.*]] = ttkernel.reinterpret_cast<tt_l1_ptr uint32_t*>({{.*}}) : (i32) -> !ttkernel.l1_addr_ptr<16>
// CHECK: ttkernel.store_to_l1(%[[BITS]], %[[L1]], %[[C11]]) : (i16, !ttkernel.l1_addr_ptr<16>, i32) -> ()
module {
  func.func @write_row_major_bf16()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 8], bf16, 2>
    %block = ttl.cb_reserve %cb : <[4, 8], bf16, 2> -> tensor<4x8xbf16>
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %cst = arith.constant 2.500000e+00 : bf16
    ttl.raw_element_write %block[%c1, %c3], %cst : tensor<4x8xbf16>, bf16
    func.return
  }
}

// -----

// Read from multi-tile [1,3] grid at coordinate (0, 48).
// col 48 spans into tile [0,1] (tileCol = 48/32 = 1), intraCol = 48 - 32 = 16.
// tileIdx = 0*3 + 1 = 1. IntraRow=0, intraCol=16 -> face1 origin, faceOff = 256.
// offset = 1*1024 + 256 = 1280.
// CHECK-LABEL: func.func @read_multitile_offset
// CHECK-DAG: %[[C1280:.*]] = arith.constant 1280 : i32
// CHECK: ttkernel.load_from_l1({{.*}}, %[[C1280]]) : (!ttkernel.l1_addr_ptr, i32) -> i32
module {
  func.func @read_multitile_offset()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 3], !ttcore.tile<32x32, f32>, 2>
    %block = ttl.cb_wait %cb : <[1, 3], !ttcore.tile<32x32, f32>, 2> -> tensor<1x3x!ttcore.tile<32x32, f32>>
    %c0 = arith.constant 0 : index
    %c48 = arith.constant 48 : index
    %val = ttl.raw_element_read %block[%c0, %c48] : tensor<1x3x!ttcore.tile<32x32, f32>> -> f32
    func.return
  }
}

// -----

// Raw element access inside a loop with dynamic iteration variable.
// CHECK-LABEL: func.func @read_in_loop
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: scf.for {{.*}} {
// CHECK:   ttkernel.load_from_l1
// CHECK: }
module {
  func.func @read_in_loop()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 8], f32, 2>
    %block = ttl.cb_wait %cb : <[4, 8], f32, 2> -> tensor<4x8xf32>
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c8 step %c1 {
      %val = ttl.raw_element_read %block[%c0, %i] : tensor<4x8xf32> -> f32
    }
    func.return
  }
}

// -----

// Read f32 from an attach_cb-wrapped cb_wait block (the form the Python
// frontend emits). Must trace through attach_cb to get_read_ptr.
// CHECK-LABEL: func.func @read_attach_cb_wait
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[PTR:.*]] = ttkernel.get_read_ptr(%[[CB]])
// CHECK: %[[L1:.*]] = ttkernel.reinterpret_cast<tt_l1_ptr uint32_t*>(%[[PTR]]) : (i32) -> !ttkernel.l1_addr_ptr
// CHECK: ttkernel.load_from_l1(%[[L1]], {{.*}}) : (!ttkernel.l1_addr_ptr, i32) -> i32
module {
  func.func @read_attach_cb_wait()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %wait = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %block = ttl.attach_cb %wait, %cb
        : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %val = ttl.raw_element_read %block[%c0, %c5] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
    func.return
  }
}

// -----

// Write f32 to an attach_cb-wrapped cb_reserve block. Must trace through
// attach_cb to get_write_ptr.
// 1.0f = 0x3F800000 = 1065353216.
// CHECK-LABEL: func.func @write_attach_cb_reserve
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[BITS:.*]] = arith.constant 1065353216 : i32
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[PTR:.*]] = ttkernel.get_write_ptr(%[[CB]])
// CHECK: %[[L1:.*]] = ttkernel.reinterpret_cast<tt_l1_ptr uint32_t*>(%[[PTR]]) : (i32) -> !ttkernel.l1_addr_ptr
// CHECK: ttkernel.store_to_l1(%[[BITS]], %[[L1]], %[[C0]]) : (i32, !ttkernel.l1_addr_ptr, i32) -> ()
module {
  func.func @write_attach_cb_reserve()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %res = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %block = ttl.attach_cb %res, %cb
        : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : f32
    ttl.raw_element_write %block[%c0, %c0], %cst : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
    func.return
  }
}
