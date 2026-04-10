// Tests for ttl-subblock-compute-for-dst with multiple output CBs.
// Verifies that 3 independent compute chains writing to 3 separate output
// CBs are each correctly subblocked for DST.
// Derived from test_comprehensive_multinode (20 fused ops, 3 outputs).
// Shape: 4x4 bf16 (capacity=8). Multi-dim tiling: tileSizes=[2,4], product=8.
// Loop on dim 0 (0 to 4 step 2). Stride 4 for dim 0 offset.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-set-compute-kernel-config,ttl-assign-dst,ttl-subblock-compute-for-dst,canonicalize,cse))' | FileCheck %s --check-prefix=SUBBLOCK

// SUBBLOCK-LABEL: func.func @fused_compute
// Verify three separate scf.for loops with inner ttl.compute ops (one per
// output chain). iter_index ops produce local subblock coordinates directly
// (no arith.addi offset). tile_store views reference extract_slice of the
// attach_cb result.
//
// SUBBLOCK-DAG:    %[[C0:.*]] = arith.constant 0 : index
// SUBBLOCK-DAG:    %[[C1:.*]] = arith.constant 1 : index
// Chain 1:
// SUBBLOCK:        scf.for %[[IV1:.*]] =
// SUBBLOCK:          ttl.compute
// SUBBLOCK:            %[[I_DIM0_A:.*]] = ttl.iter_index 0 : index
// SUBBLOCK-NEXT:       %[[I_DIM1_A:.*]] = ttl.iter_index 1 : index
// SUBBLOCK:            ttl.copy_tile %{{.*}}[%[[I_DIM0_A]], %[[I_DIM1_A]]] into dst[%[[C0]]]
// SUBBLOCK:            ttl.tile_store %{{.*}}, %{{.*}}[%[[I_DIM0_A]], %[[I_DIM1_A]]]
// SUBBLOCK:        } {ttl.subblock_dim = 0 : index, ttl.subblock_loop_stride = 4 : index}
// Chain 2:
// SUBBLOCK:        scf.for %[[IV2:.*]] =
// SUBBLOCK:          ttl.compute
// SUBBLOCK:            %[[I_DIM0_B:.*]] = ttl.iter_index 0 : index
// SUBBLOCK-NEXT:       %[[I_DIM1_B:.*]] = ttl.iter_index 1 : index
// SUBBLOCK:            ttl.copy_tile %{{.*}}[%[[I_DIM0_B]], %[[I_DIM1_B]]] into dst[%[[C0]]]
// SUBBLOCK:            ttl.tile_store %{{.*}}, %{{.*}}[%[[I_DIM0_B]], %[[I_DIM1_B]]]
// SUBBLOCK:        } {ttl.subblock_dim = 0 : index, ttl.subblock_loop_stride = 4 : index}
// Chain 3:
// SUBBLOCK:        scf.for %[[IV3:.*]] =
// SUBBLOCK:          ttl.compute
// SUBBLOCK:            %[[I_DIM0_C:.*]] = ttl.iter_index 0 : index
// SUBBLOCK-NEXT:       %[[I_DIM1_C:.*]] = ttl.iter_index 1 : index
// SUBBLOCK:            ttl.copy_tile %{{.*}}[%[[I_DIM0_C]], %[[I_DIM1_C]]] into dst[%[[C0]]]
// SUBBLOCK:            ttl.tile_store %{{.*}}, %{{.*}}[%[[I_DIM0_C]], %[[I_DIM1_C]]]
// SUBBLOCK:        } {ttl.subblock_dim = 0 : index, ttl.subblock_loop_stride = 4 : index}

// Verify that lower-to-loops produces an outer subblock scf.for with unrolled
// inner tile copies (inner tile loops are fully unrolled). Each chain has one
// scf.for with subblock attributes and 4 unrolled copies (1x4 subblock).

// Purpose: Compute function with 3 input CBs, 3 output CBs, and 20 fused ops
// across 3 store chains. Each chain reads from different input CBs and stores
// to a different output CB. This tests that the subblock pass handles multiple
// independent compute regions within a single function.
module {
  func.func @fused_compute() attributes {ttl.base_cta_index = 6 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
    %0 = ttl.bind_cb{cb_index = 0, block_count = 2} : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
    %1 = ttl.bind_cb{cb_index = 1, block_count = 2} : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
    %2 = ttl.bind_cb{cb_index = 2, block_count = 2} : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
    %3 = ttl.bind_cb{cb_index = 3, block_count = 2} : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
    %4 = ttl.bind_cb{cb_index = 4, block_count = 2} : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
    %5 = ttl.bind_cb{cb_index = 5, block_count = 2} : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
    %6 = ttl.cb_wait %0 : <[4, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %7 = ttl.attach_cb %6, %0 : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %8 = ttl.cb_wait %1 : <[4, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %9 = ttl.attach_cb %8, %1 : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %10 = ttl.cb_wait %2 : <[4, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %11 = ttl.attach_cb %10, %2 : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %12 = ttl.cb_reserve %3 : <[4, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %13 = ttl.attach_cb %12, %3 : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %14 = ttl.cb_reserve %4 : <[4, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %15 = ttl.attach_cb %14, %4 : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %16 = ttl.cb_reserve %5 : <[4, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %17 = ttl.attach_cb %16, %5 : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    // Chain 1: a -> sigmoid -> tanh -> add(b) -> sigmoid -> tanh -> abs -> relu -> store(out1)
    %18 = ttl.sigmoid %7 : tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %19 = ttl.tanh %18 : tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %20 = ttl.add %19, %9 : tensor<4x4x!ttcore.tile<32x32, bf16>>, tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %21 = ttl.sigmoid %20 : tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %22 = ttl.tanh %21 : tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %23 = ttl.abs %22 : tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %24 = ttl.relu %23 : tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    ttl.store %24, %12 : tensor<4x4x!ttcore.tile<32x32, bf16>>, tensor<4x4x!ttcore.tile<32x32, bf16>>
    %25 = ttl.attach_cb %24, %3 : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    // Chain 2: b -> tanh -> sigmoid -> add(c) -> tanh -> neg -> abs -> sigmoid -> store(out2)
    %26 = ttl.tanh %9 : tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %27 = ttl.sigmoid %26 : tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %28 = ttl.add %27, %11 : tensor<4x4x!ttcore.tile<32x32, bf16>>, tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %29 = ttl.tanh %28 : tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %30 = ttl.neg %29 : tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %31 = ttl.abs %30 : tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %32 = ttl.sigmoid %31 : tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    ttl.store %32, %14 : tensor<4x4x!ttcore.tile<32x32, bf16>>, tensor<4x4x!ttcore.tile<32x32, bf16>>
    %33 = ttl.attach_cb %32, %4 : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    // Chain 3: a -> relu -> sigmoid -> add(c) -> tanh -> abs -> sigmoid -> store(out3)
    %34 = ttl.relu %7 : tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %35 = ttl.sigmoid %34 : tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %36 = ttl.add %35, %11 : tensor<4x4x!ttcore.tile<32x32, bf16>>, tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %37 = ttl.tanh %36 : tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %38 = ttl.abs %37 : tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %39 = ttl.sigmoid %38 : tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    ttl.store %39, %16 : tensor<4x4x!ttcore.tile<32x32, bf16>>, tensor<4x4x!ttcore.tile<32x32, bf16>>
    %40 = ttl.attach_cb %39, %5 : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %5 : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_push %4 : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_push %3 : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %2 : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %1 : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %0 : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
