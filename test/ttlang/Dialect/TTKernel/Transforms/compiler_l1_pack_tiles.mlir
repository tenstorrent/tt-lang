// Compiler-owned packing retains explicit tile indices; Metal packing combines them.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttkernel-combine-pack-tiles))' --split-input-file | FileCheck %s

module attributes {ttl.memory_model = "compiler-l1"} {
  // The block operation cannot preserve explicit output tile indices.
  // CHECK-LABEL: func.func @compiler_owned
  // CHECK: %[[STORAGE:.*]] = ttkernel.get_compile_time_arg_val
  // CHECK: %[[ZERO:.*]] = arith.constant 0
  // CHECK: %[[ONE:.*]] = arith.constant 1
  // CHECK-NEXT: ttkernel.pack_tile(%[[ZERO]], %[[STORAGE]], %[[ZERO]], true)
  // CHECK-NEXT: ttkernel.pack_tile(%[[ONE]], %[[STORAGE]], %[[ONE]], true)
  // CHECK-NEXT: return
  func.func @compiler_owned() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    ttkernel.pack_tile(%zero, %storage, %zero, true) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.pack_tile(%one, %storage, %one, true) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
    return
  }
}

// -----

module attributes {ttl.memory_model = "metal-cb"} {
  // The normal backend retains its descriptor-based block-packing optimization.
  // CHECK-LABEL: func.func @metal_owned
  // CHECK: ttkernel.pack_tile_block(
  // CHECK-NEXT: return
  func.func @metal_owned() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    ttkernel.pack_tile(%zero, %storage, %zero, true) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.pack_tile(%one, %storage, %one, true) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
    return
  }
}
