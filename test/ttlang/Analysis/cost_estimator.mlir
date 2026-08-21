// RUN: ttlang-opt --ttkernel-cost-estimate='enable=1 detail=1' %s -o /dev/null 2>&1 | FileCheck %s
// The detail view is part of the report, so asking for it without the
// estimate is refused rather than ignored.
// RUN: not ttlang-opt --ttkernel-cost-estimate='detail=1' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOENABLE
// NOENABLE: error: cost estimate detail was requested with the estimate disabled
// NOENABLE-SAME: pass 'enable'
// NOENABLE-NOT: cost estimate:

// Three kernel threads over two input CBs and one output, four tiles per block.
// The reader and writer only move credits, which is all the compute kernel needs
// to get past cb_wait_front; the point is the cost table and the credit model,
// not the data movement.
//
// Every op here is in the cost table, so the estimate is produced rather than
// failed, and the per-op `src` column shows what backs each cost: `meas` a
// measurement keyed to this kernel, `nokey` one taken in a configuration this
// kernel cannot key, `untimed` an engine no sweep timed. This is the regression
// test for the table wiring: a lookup that stops matching shows up as a column
// flipping away from meas.

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @read() attributes {ttkernel.thread = #ttkernel.thread<noc>, ttl.noc_index = 0 : i32} {
    %c4_i32 = arith.constant 4 : i32
    %0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>
    %2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>
    ttkernel.cb_reserve_back(%0, %c4_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, i32) -> ()
    ttkernel.cb_reserve_back(%2, %c4_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, i32) -> ()
    ttkernel.noc_async_read_barrier() : () -> ()
    ttkernel.cb_push_back(%0, %c4_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, i32) -> ()
    ttkernel.cb_push_back(%2, %c4_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, i32) -> ()
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
    %2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>
    ttkernel.cb_wait_front(%0, %c4_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, i32) -> ()
    ttkernel.cb_wait_front(%2, %c4_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, i32) -> ()
    ttkernel.cb_reserve_back(%1, %c4_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, i32) -> ()
    ttkernel.binary_op_init_common(%0, %2, %1) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.add_tiles_init(%0, %2) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.add_tiles(%0, %2, %c0, %c0, %c0) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
    ttkernel.add_tiles(%0, %2, %c1, %c1, %c1) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
    ttkernel.add_tiles(%0, %2, %c2, %c2, %c2) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
    ttkernel.add_tiles(%0, %2, %c3, %c3, %c3) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %1, %c0, true) : (index, !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.pack_tile(%c1, %1, %c1, true) : (index, !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.pack_tile(%c2, %1, %c2, true) : (index, !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.pack_tile(%c3, %1, %c3, true) : (index, !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
    ttkernel.cb_push_back(%1, %c4_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, i32) -> ()
    ttkernel.cb_pop_front(%0, %c4_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, i32) -> ()
    ttkernel.cb_pop_front(%2, %c4_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, i32) -> ()
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

// The three threads are recognized and none of them deadlocks. Nothing here is
// keyed on a field this kernel cannot supply, so the placements that carry no
// cost are all untimed ones -- the circular-buffer and DST handshakes.
// CHECK: cost estimate: {{[0-9]+}} of {{[0-9]+}} placements measured
// CHECK-NEXT: 0 unmatched {{.*}} untimed
// CHECK: kernels: read compute write

// Both halves of the eltwise add are measured on both engines. That is what
// splitting the benchmark's init zones bought: an init whose halves are
// separately attributed, where a lumped zone leaves one of these columns
// unattributed.
// CHECK: TRISC0 unpack
// CHECK: ttkernel.binary_op_init_common {{.*}} meas
// CHECK: ttkernel.add_tiles_init {{.*}} meas
// CHECK: ttkernel.add_tiles {{.*}} meas

// MATH trails UNPACK by the Src bank credit rather than running with it.
// CHECK: TRISC1 math
// CHECK: ttkernel.binary_op_init_common {{.*}} meas
// CHECK: ttkernel.add_tiles {{.*}} meas

// The credit operations are the untimed ones: no perf source isolates a
// handshake, so they cost nothing and are counted apart from the measured work.
// CHECK: ttkernel.tile_regs_commit {{.*}} untimed

// CHECK: TRISC2 pack
// CHECK: ttkernel.binary_op_init_common {{.*}} meas
// CHECK: ttkernel.pack_tile {{.*}} meas
