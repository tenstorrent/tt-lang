// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttkernel-specialize-and-annotate-dfb-use)' | FileCheck %s
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttkernel-cleanup-and-finalize-runtime-args)' | FileCheck %s
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline | FileCheck %s

// Summary: Record-loop expansion resolves destination tables before write-state
// cleanup, with and without core specialization enabled.

// Each expanded record configures its own destination before waiting. Cleanup
// before unrolling cannot move the record-dependent destination out of the loop.
// CHECK-LABEL: func.func @record_sends
// CHECK-NOT: scf.for
// CHECK-NOT: constant_table_lookup
// CHECK: %[[ADDR0:.*]] = ttkernel.get_noc_addr
// CHECK-NEXT: ttkernel.noc_async_write_one_packet_set_state(%[[ADDR0]], {{.*}}) posted true
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: ttkernel.noc_semaphore_set
// CHECK: ttkernel.noc_async_write_one_packet_with_state(%arg0, %arg1, {{.*}}) posted true
// CHECK: %[[ADDR1:.*]] = ttkernel.get_noc_addr
// CHECK-NEXT: ttkernel.noc_async_write_one_packet_set_state(%[[ADDR1]], {{.*}}) posted true
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: ttkernel.noc_semaphore_set
// CHECK: ttkernel.noc_async_write_one_packet_with_state(%arg0, %arg1, {{.*}}) posted true
// CHECK-NOT: ttkernel.noc_async_write %
// CHECK: return
module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @record_sends(%src: i32, %dst: i32) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %noc = arith.constant 0 : i8
    %ready = arith.constant 1 : i32
    %bytes = arith.constant 1024 : i32
    %semaphore = ttkernel.get_semaphore(%one)
        : (index) -> !ttkernel.local_semaphore
    %address = ttkernel.reinterpret_cast(%semaphore)
        : (!ttkernel.local_semaphore) -> !ttkernel.l1_addr_ptr
    scf.for %record = %zero to %two step %one {
      %logical_x = ttkernel.experimental.constant_table_lookup %record, [0, 1] : index
      %destination_x = ttkernel.experimental.convert_logical_x_to_translated(%logical_x)
          : (index) -> index
      %destination_y = ttkernel.experimental.convert_logical_y_to_translated(%zero)
          : (index) -> index
      ttkernel.experimental.semaphore_wait(%address, %ready)
          : (!ttkernel.l1_addr_ptr, i32) -> ()
      ttkernel.noc_semaphore_set(%address, %zero)
          : (!ttkernel.l1_addr_ptr, index) -> ()
      ttkernel.noc_async_write %src, core[%destination_x, %destination_y], %dst,
          %bytes, noc %noc posted true
          : (i32, index, index, i32, i32, i8) -> ()
    } {ttl.pipenet_local_record_loop}
    return
  }
}
