// RUN: ttlang-opt --ttkernel-cost-estimate='enable=1 detail=1' %s -o /dev/null 2>&1 | FileCheck %s

// The unary datacopy path, which the eltwise test in this directory does not
// reach. Its three compute ops are the ones perf_copy_tile measures, and every
// row it left behind is keyed on `unpack_to_dest` -- a per-buffer decision, not
// a kernel-wide one, and worth a factor of three on unpack. This kernel lists no
// buffer in `ttl.unpack_to_dest_fp32`, so the answer is false for each of them
// and copy_tile costs 42 on unpack rather than the 121 a listed buffer would.

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @read() attributes {ttkernel.thread = #ttkernel.thread<noc>, ttl.noc_index = 0 : i32} {
    %c4_i32 = arith.constant 4 : i32
    %0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>
    ttkernel.cb_reserve_back(%0, %c4_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, i32) -> ()
    ttkernel.noc_async_read_barrier() : () -> ()
    ttkernel.cb_push_back(%0, %c4_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, i32) -> ()
    return
  }

  func.func @compute() attributes {dst_full_sync_en = false, fp32_dest_acc_en = false, ttkernel.thread = #ttkernel.thread<compute>} {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c4_i32 = arith.constant 4 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>
    %1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>
    ttkernel.cb_wait_front(%0, %c4_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, i32) -> ()
    ttkernel.cb_reserve_back(%1, %c4_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, i32) -> ()
    ttkernel.compute_kernel_hw_startup(%0, %1) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.copy_tile_init(%0) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%0, %c0, %c0) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    ttkernel.copy_tile(%0, %c1, %c1) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    ttkernel.copy_tile(%0, %c2, %c2) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    ttkernel.copy_tile(%0, %c3, %c3) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %1, %c0, true) : (index, !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.pack_tile(%c1, %1, %c1, true) : (index, !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.pack_tile(%c2, %1, %c2, true) : (index, !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.pack_tile(%c3, %1, %c3, true) : (index, !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
    ttkernel.cb_push_back(%1, %c4_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, i32) -> ()
    ttkernel.cb_pop_front(%0, %c4_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, i32) -> ()
    return
  }

  func.func @write() attributes {ttkernel.thread = #ttkernel.thread<noc>, ttl.noc_index = 1 : i32} {
    %c4_i32 = arith.constant 4 : i32
    %1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>
    ttkernel.cb_wait_front(%1, %c4_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, i32) -> ()
    ttkernel.noc_async_write_barrier() : () -> ()
    ttkernel.cb_pop_front(%1, %c4_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, i32) -> ()
    return
  }
}

// CHECK: cost estimate: {{[0-9]+}} of {{[0-9]+}} placements measured
// CHECK-NEXT: 0 unmatched {{.*}} untimed
// CHECK: kernels: read compute write

// Every compute op on every lane is measured. That is what splitting the
// benchmark's init zone bought: an init whose halves are separately attributed,
// where a lumped zone leaves one of these two columns unattributed.
// CHECK: TRISC0 unpack
// CHECK: ttkernel.compute_kernel_hw_startup {{.*}} meas
// CHECK: ttkernel.copy_tile_init {{.*}} meas
// CHECK: ttkernel.copy_tile {{.*}} 42 {{.*}} meas

// CHECK: TRISC1 math
// CHECK: ttkernel.compute_kernel_hw_startup {{.*}} meas
// CHECK: ttkernel.copy_tile_init {{.*}} meas
// CHECK: ttkernel.copy_tile {{.*}} meas

// The pack lane's init zone was left unsplit because copy_tile_init has no pack
// half; the zone is compute_kernel_hw_startup's three pack calls exactly, which
// is why it is attributable without a split.
// CHECK: TRISC2 pack
// CHECK: ttkernel.compute_kernel_hw_startup {{.*}} meas
// CHECK: ttkernel.pack_tile {{.*}} meas
