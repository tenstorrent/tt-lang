// Integration tests: raw element read/write combined with scalar float
// comparisons through the full --convert-ttl-to-ttkernel + --ttl-lower-scalar-cmpf
// pipeline. Covers ogt/olt for f32 and bf16, comparisons against constants,
// and comparisons within loops.
// RUN: ttlang-opt --convert-ttl-to-ttkernel --ttl-lower-scalar-cmpf --canonicalize -cse --split-input-file %s | FileCheck %s

// -----

// f32 ogt: read two elements, compare with >, result feeds scf.if for
// conditional swap (pairwise sort pattern).
// CHECK-LABEL: func.func @sort_pair_ogt_f32
// CHECK-NOT: arith.cmpf
// CHECK-NOT: ttl.raw_element_read
// CHECK: %[[A:.*]] = ttkernel.load_from_l1({{.*}}) : (!ttkernel.l1_addr_ptr, i32) -> i32
// CHECK: %[[B:.*]] = ttkernel.load_from_l1({{.*}}) : (!ttkernel.l1_addr_ptr, i32) -> i32
// CHECK: ttkernel.store_to_l1(%[[A]], {{.*}}) : (i32, !ttkernel.l1_addr_ptr, i32) -> ()
// CHECK: ttkernel.store_to_l1(%[[B]], {{.*}}) : (i32, !ttkernel.l1_addr_ptr, i32) -> ()
// CHECK: %[[CMP:.*]] = ttkernel.float32_greater(%[[A]], %[[B]]) : (i32, i32) -> i1
// CHECK: scf.if %[[CMP]]
// CHECK:   ttkernel.store_to_l1(%[[B]], {{.*}}) : (i32, !ttkernel.l1_addr_ptr, i32) -> ()
// CHECK:   ttkernel.store_to_l1(%[[A]], {{.*}}) : (i32, !ttkernel.l1_addr_ptr, i32) -> ()
module {
  func.func @sort_pair_ogt_f32()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %inp_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %out_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %rblk = ttl.cb_wait %inp_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %wblk = ttl.cb_reserve %out_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a = ttl.raw_element_read %rblk[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
    %b = ttl.raw_element_read %rblk[%c0, %c1] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
    ttl.raw_element_write %wblk[%c0, %c0], %a : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
    ttl.raw_element_write %wblk[%c0, %c1], %b : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
    %cmp = arith.cmpf ogt, %a, %b : f32
    scf.if %cmp {
      ttl.raw_element_write %wblk[%c0, %c0], %b : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
      ttl.raw_element_write %wblk[%c0, %c1], %a : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
    }
    func.return
  }
}

// -----

// f32 olt: read two elements, compare with <, conditional swap.
// CHECK-LABEL: func.func @sort_pair_olt_f32
// CHECK-NOT: arith.cmpf
// CHECK: %[[A:.*]] = ttkernel.load_from_l1({{.*}}) : (!ttkernel.l1_addr_ptr, i32) -> i32
// CHECK: %[[B:.*]] = ttkernel.load_from_l1({{.*}}) : (!ttkernel.l1_addr_ptr, i32) -> i32
// CHECK: %[[CMP:.*]] = ttkernel.float32_greater(%[[B]], %[[A]]) : (i32, i32) -> i1
// CHECK: scf.if %[[CMP]]
module {
  func.func @sort_pair_olt_f32()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %inp_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %out_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %rblk = ttl.cb_wait %inp_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %wblk = ttl.cb_reserve %out_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a = ttl.raw_element_read %rblk[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
    %b = ttl.raw_element_read %rblk[%c0, %c1] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
    ttl.raw_element_write %wblk[%c0, %c0], %a : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
    ttl.raw_element_write %wblk[%c0, %c1], %b : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
    %cmp = arith.cmpf olt, %a, %b : f32
    scf.if %cmp {
      ttl.raw_element_write %wblk[%c0, %c0], %b : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
      ttl.raw_element_write %wblk[%c0, %c1], %a : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
    }
    func.return
  }
}

// -----

// f32 ogt against constant threshold (1.0f = 0x3F800000 = 1065353216).
// CHECK-LABEL: func.func @threshold_ogt_f32_const
// CHECK-NOT: arith.cmpf
// CHECK-DAG: %[[THRESH:.*]] = arith.constant 1065353216 : i32
// CHECK: %[[V:.*]] = ttkernel.load_from_l1({{.*}}) : (!ttkernel.l1_addr_ptr, i32) -> i32
// CHECK: %[[CMP:.*]] = ttkernel.float32_greater(%[[V]], %[[THRESH]]) : (i32, i32) -> i1
// CHECK: scf.if %[[CMP]]
// CHECK:   ttkernel.store_to_l1
module {
  func.func @threshold_ogt_f32_const()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %inp_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %out_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %rblk = ttl.cb_wait %inp_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %wblk = ttl.cb_reserve %out_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %c0 = arith.constant 0 : index
    %thresh = arith.constant 1.000000e+00 : f32
    %v = ttl.raw_element_read %rblk[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
    %cmp = arith.cmpf ogt, %v, %thresh : f32
    scf.if %cmp {
      ttl.raw_element_write %wblk[%c0, %c0], %v : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
    }
    func.return
  }
}

// -----

// bf16 ogt: read two bf16 elements, compare, conditional swap.
// CHECK-LABEL: func.func @sort_pair_ogt_bf16
// CHECK-NOT: arith.cmpf
// CHECK: %[[A:.*]] = ttkernel.load_from_l1({{.*}}) : (!ttkernel.l1_addr_ptr<16>, i32) -> i16
// CHECK: %[[B:.*]] = ttkernel.load_from_l1({{.*}}) : (!ttkernel.l1_addr_ptr<16>, i32) -> i16
// CHECK: %[[CMP:.*]] = ttkernel.bfloat16_greater(%[[A]], %[[B]]) : (i16, i16) -> i1
// CHECK: scf.if %[[CMP]]
module {
  func.func @sort_pair_ogt_bf16()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %inp_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %out_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %rblk = ttl.cb_wait %inp_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %wblk = ttl.cb_reserve %out_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a = ttl.raw_element_read %rblk[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> bf16
    %b = ttl.raw_element_read %rblk[%c0, %c1] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> bf16
    ttl.raw_element_write %wblk[%c0, %c0], %a : tensor<1x1x!ttcore.tile<32x32, bf16>>, bf16
    ttl.raw_element_write %wblk[%c0, %c1], %b : tensor<1x1x!ttcore.tile<32x32, bf16>>, bf16
    %cmp = arith.cmpf ogt, %a, %b : bf16
    scf.if %cmp {
      ttl.raw_element_write %wblk[%c0, %c0], %b : tensor<1x1x!ttcore.tile<32x32, bf16>>, bf16
      ttl.raw_element_write %wblk[%c0, %c1], %a : tensor<1x1x!ttcore.tile<32x32, bf16>>, bf16
    }
    func.return
  }
}

// -----

// bf16 olt: read two bf16 elements, compare with <, conditional swap.
// olt is lowered as bfloat16_greater with swapped operands.
// CHECK-LABEL: func.func @sort_pair_olt_bf16
// CHECK-NOT: arith.cmpf
// CHECK: %[[A:.*]] = ttkernel.load_from_l1({{.*}}) : (!ttkernel.l1_addr_ptr<16>, i32) -> i16
// CHECK: %[[B:.*]] = ttkernel.load_from_l1({{.*}}) : (!ttkernel.l1_addr_ptr<16>, i32) -> i16
// CHECK: %[[CMP:.*]] = ttkernel.bfloat16_greater(%[[B]], %[[A]]) : (i16, i16) -> i1
// CHECK: scf.if %[[CMP]]
module {
  func.func @sort_pair_olt_bf16()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %inp_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %out_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %rblk = ttl.cb_wait %inp_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %wblk = ttl.cb_reserve %out_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a = ttl.raw_element_read %rblk[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> bf16
    %b = ttl.raw_element_read %rblk[%c0, %c1] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> bf16
    ttl.raw_element_write %wblk[%c0, %c0], %a : tensor<1x1x!ttcore.tile<32x32, bf16>>, bf16
    ttl.raw_element_write %wblk[%c0, %c1], %b : tensor<1x1x!ttcore.tile<32x32, bf16>>, bf16
    %cmp = arith.cmpf olt, %a, %b : bf16
    scf.if %cmp {
      ttl.raw_element_write %wblk[%c0, %c0], %b : tensor<1x1x!ttcore.tile<32x32, bf16>>, bf16
      ttl.raw_element_write %wblk[%c0, %c1], %a : tensor<1x1x!ttcore.tile<32x32, bf16>>, bf16
    }
    func.return
  }
}

// -----

// f32 comparison inside a loop: read elements in a loop, compare each against
// a threshold, conditionally write.
// CHECK-LABEL: func.func @cmpf_in_loop_f32
// CHECK-NOT: arith.cmpf
// CHECK-DAG: %[[THRESH:.*]] = arith.constant 0 : i32
// CHECK: scf.for
// CHECK:   ttkernel.load_from_l1
// CHECK:   ttkernel.float32_greater
// CHECK:   scf.if
// CHECK:     ttkernel.store_to_l1
module {
  func.func @cmpf_in_loop_f32()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %inp_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 8], f32, 2>
    %out_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 8], f32, 2>
    %rblk = ttl.cb_wait %inp_cb : <[4, 8], f32, 2> -> tensor<4x8xf32>
    %wblk = ttl.cb_reserve %out_cb : <[4, 8], f32, 2> -> tensor<4x8xf32>
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.000000e+00 : f32
    scf.for %i = %c0 to %c8 step %c1 {
      %v = ttl.raw_element_read %rblk[%c0, %i] : tensor<4x8xf32> -> f32
      %cmp = arith.cmpf ogt, %v, %zero : f32
      scf.if %cmp {
        ttl.raw_element_write %wblk[%c0, %i], %v : tensor<4x8xf32>, f32
      }
    }
    func.return
  }
}
