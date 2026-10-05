// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttkernel-batch-static-pipenet-receives))' | FileCheck %s --implicit-check-not=ttl.pipenet_initial_receive_capacity

// Verify batching order, private counter preservation, and conservative rejection of unsafe schedules.

// Reserve both initial slots and post both senders before either completion wait.
// CHECK-LABEL: func.func @batch_two
// CHECK: %[[TOTAL:.*]] = arith.constant 2 : i32
// CHECK-NEXT: ttkernel.cb_reserve_back(%[[DFB:.*]], %[[TOTAL]])
// CHECK-NOT: semaphore_wait_min
// CHECK: ttkernel.noc_semaphore_inc
// CHECK-NOT: semaphore_wait_min
// CHECK: ttkernel.noc_semaphore_inc
// CHECK-NEXT: ttkernel.noc_async_atomic_barrier
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK-NEXT: ttkernel.cb_push_back(%[[DFB]],
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK-NEXT: ttkernel.cb_push_back(%[[DFB]],
// CHECK-NEXT: return
func.func @batch_two(%runtime_upper: index, %external: memref<4xi32>, %condition: i1) {
  %lower = arith.constant 0 : index
  %upper = arith.constant 2 : index
  %step = arith.constant 1 : index
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %pages = arith.constant 1 : i32
  %sequence = arith.constant 1 : i32
  %zero_i32 = arith.constant 0 : i32
  %noc = arith.constant 0 : i8
  %dfb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %ready_semaphore = ttkernel.get_semaphore(%zero) : (index) -> !ttkernel.local_semaphore
  scf.for %record = %lower to %upper step %step {
    ttkernel.cb_reserve_back(%dfb, %pages) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
    %ready = ttkernel.get_noc_addr(%record, %zero, %ready_semaphore, %noc) : (index, index, !ttkernel.local_semaphore, i8) -> !ttkernel.noc_addr
    ttkernel.noc_semaphore_inc(%ready, %one, %noc) : (!ttkernel.noc_addr, index, i8) -> ()
    %completion = ttkernel.get_semaphore(%record) : (index) -> !ttkernel.local_semaphore
    %completion_ptr = ttkernel.reinterpret_cast(%completion) : (!ttkernel.local_semaphore) -> !ttkernel.l1_addr_ptr
    ttkernel.experimental.semaphore_wait_min(%completion_ptr, %sequence) : (!ttkernel.l1_addr_ptr, i32) -> ()
    ttkernel.cb_push_back(%dfb, %pages) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
  } {ttl.pipenet_initial_receive_capacity = 2 : i64}
  return
}

// -----

// Preserve a nonzero lower bound and step while combining multi-page reservations.
// CHECK-LABEL: func.func @strided_multi_page
// CHECK: %[[TOTAL:.*]] = arith.constant 4 : i32
// CHECK-NEXT: ttkernel.cb_reserve_back(%[[DFB:.*]], %[[TOTAL]])
// CHECK-NEXT: %[[FIRST:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[FIRST_ADDRESS:.*]] = ttkernel.get_noc_addr(%[[FIRST]],
// CHECK-NEXT: ttkernel.noc_semaphore_inc(%[[FIRST_ADDRESS]],
// CHECK-NEXT: %[[SECOND:.*]] = arith.constant 3 : index
// CHECK-NEXT: %[[SECOND_ADDRESS:.*]] = ttkernel.get_noc_addr(%[[SECOND]],
// CHECK-NEXT: ttkernel.noc_semaphore_inc(%[[SECOND_ADDRESS]],
// CHECK-NEXT: ttkernel.noc_async_atomic_barrier
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK-NEXT: ttkernel.cb_push_back(%[[DFB]],
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK-NEXT: ttkernel.cb_push_back(%[[DFB]],
// CHECK-NEXT: return
func.func @strided_multi_page(%runtime_upper: index, %external: memref<4xi32>, %condition: i1) {
  %lower = arith.constant 1 : index
  %upper = arith.constant 5 : index
  %step = arith.constant 2 : index
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %pages = arith.constant 2 : i32
  %sequence = arith.constant 1 : i32
  %zero_i32 = arith.constant 0 : i32
  %noc = arith.constant 0 : i8
  %dfb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %ready_semaphore = ttkernel.get_semaphore(%zero) : (index) -> !ttkernel.local_semaphore
  scf.for %record = %lower to %upper step %step {
    ttkernel.cb_reserve_back(%dfb, %pages) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
    %ready = ttkernel.get_noc_addr(%record, %zero, %ready_semaphore, %noc) : (index, index, !ttkernel.local_semaphore, i8) -> !ttkernel.noc_addr
    ttkernel.noc_semaphore_inc(%ready, %one, %noc) : (!ttkernel.noc_addr, index, i8) -> ()
    %completion = ttkernel.get_semaphore(%record) : (index) -> !ttkernel.local_semaphore
    %completion_ptr = ttkernel.reinterpret_cast(%completion) : (!ttkernel.local_semaphore) -> !ttkernel.l1_addr_ptr
    ttkernel.experimental.semaphore_wait_min(%completion_ptr, %sequence) : (!ttkernel.l1_addr_ptr, i32) -> ()
    ttkernel.cb_push_back(%dfb, %pages) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  } {ttl.pipenet_initial_receive_capacity = 4 : i64}
  return
}

// -----

// Keep nonescaping local sequence-counter updates in completion order.
// CHECK-LABEL: func.func @private_sequence_counters
// CHECK: %[[TOTAL:.*]] = arith.constant 2 : i32
// CHECK-NEXT: ttkernel.cb_reserve_back(%[[DFB:.*]], %[[TOTAL]])
// CHECK-NOT: semaphore_wait_min
// CHECK: ttkernel.noc_semaphore_inc
// CHECK-NOT: semaphore_wait_min
// CHECK: ttkernel.noc_semaphore_inc
// CHECK-NEXT: ttkernel.noc_async_atomic_barrier
// CHECK: %[[COUNT:.*]] = memref.load
// CHECK-NEXT: %[[NEXT:.*]] = arith.addi %[[COUNT]],
// CHECK-NEXT: memref.store %[[NEXT]],
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK-NEXT: ttkernel.cb_push_back(%[[DFB]],
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK-NEXT: ttkernel.cb_push_back(%[[DFB]],
// CHECK-NEXT: return
func.func @private_sequence_counters(%runtime_upper: index, %external: memref<4xi32>, %condition: i1) {
  %lower = arith.constant 0 : index
  %upper = arith.constant 2 : index
  %step = arith.constant 1 : index
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %pages = arith.constant 1 : i32
  %sequence = arith.constant 1 : i32
  %zero_i32 = arith.constant 0 : i32
  %noc = arith.constant 0 : i8
  %dfb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %ready_semaphore = ttkernel.get_semaphore(%zero) : (index) -> !ttkernel.local_semaphore
  %counter = memref.alloca() : memref<4xi32>
  memref.store %zero_i32, %counter[%zero] : memref<4xi32>
  memref.store %zero_i32, %counter[%one] : memref<4xi32>
  scf.for %record = %lower to %upper step %step {
    ttkernel.cb_reserve_back(%dfb, %pages) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
    %ready = ttkernel.get_noc_addr(%record, %zero, %ready_semaphore, %noc) : (index, index, !ttkernel.local_semaphore, i8) -> !ttkernel.noc_addr
    ttkernel.noc_semaphore_inc(%ready, %one, %noc) : (!ttkernel.noc_addr, index, i8) -> ()
    %count = memref.load %counter[%record] : memref<4xi32>
    %next = arith.addi %count, %sequence : i32
    memref.store %next, %counter[%record] : memref<4xi32>
    %completion = ttkernel.get_semaphore(%record) : (index) -> !ttkernel.local_semaphore
    %completion_ptr = ttkernel.reinterpret_cast(%completion) : (!ttkernel.local_semaphore) -> !ttkernel.l1_addr_ptr
    ttkernel.experimental.semaphore_wait_min(%completion_ptr, %next) : (!ttkernel.l1_addr_ptr, i32) -> ()
    ttkernel.cb_push_back(%dfb, %pages) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
  } {ttl.pipenet_initial_receive_capacity = 2 : i64}
  return
}

// -----

// Retain incremental reservations when the complete batch exceeds available pages.
// CHECK-LABEL: func.func @insufficient_capacity
// CHECK: scf.for
// CHECK: ttkernel.cb_reserve_back
// CHECK: ttkernel.noc_semaphore_inc
// CHECK-NOT: noc_async_atomic_barrier
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: ttkernel.cb_push_back
// CHECK: return
func.func @insufficient_capacity(%runtime_upper: index, %external: memref<4xi32>, %condition: i1) {
  %lower = arith.constant 0 : index
  %upper = arith.constant 2 : index
  %step = arith.constant 1 : index
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %pages = arith.constant 1 : i32
  %sequence = arith.constant 1 : i32
  %zero_i32 = arith.constant 0 : i32
  %noc = arith.constant 0 : i8
  %dfb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  %ready_semaphore = ttkernel.get_semaphore(%zero) : (index) -> !ttkernel.local_semaphore
  scf.for %record = %lower to %upper step %step {
    ttkernel.cb_reserve_back(%dfb, %pages) : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, i32) -> ()
    %ready = ttkernel.get_noc_addr(%record, %zero, %ready_semaphore, %noc) : (index, index, !ttkernel.local_semaphore, i8) -> !ttkernel.noc_addr
    ttkernel.noc_semaphore_inc(%ready, %one, %noc) : (!ttkernel.noc_addr, index, i8) -> ()
    %completion = ttkernel.get_semaphore(%record) : (index) -> !ttkernel.local_semaphore
    %completion_ptr = ttkernel.reinterpret_cast(%completion) : (!ttkernel.local_semaphore) -> !ttkernel.l1_addr_ptr
    ttkernel.experimental.semaphore_wait_min(%completion_ptr, %sequence) : (!ttkernel.l1_addr_ptr, i32) -> ()
    ttkernel.cb_push_back(%dfb, %pages) : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, i32) -> ()
  } {ttl.pipenet_initial_receive_capacity = 1 : i64}
  return
}

// -----

// Do not batch an empty receive sequence.
// CHECK-LABEL: func.func @zero_records
// CHECK: scf.for
// CHECK: ttkernel.cb_reserve_back
// CHECK: ttkernel.noc_semaphore_inc
// CHECK-NOT: noc_async_atomic_barrier
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: ttkernel.cb_push_back
// CHECK: return
func.func @zero_records(%runtime_upper: index, %external: memref<4xi32>, %condition: i1) {
  %lower = arith.constant 0 : index
  %upper = arith.constant 0 : index
  %step = arith.constant 1 : index
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %pages = arith.constant 1 : i32
  %sequence = arith.constant 1 : i32
  %zero_i32 = arith.constant 0 : i32
  %noc = arith.constant 0 : i8
  %dfb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %ready_semaphore = ttkernel.get_semaphore(%zero) : (index) -> !ttkernel.local_semaphore
  scf.for %record = %lower to %upper step %step {
    ttkernel.cb_reserve_back(%dfb, %pages) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
    %ready = ttkernel.get_noc_addr(%record, %zero, %ready_semaphore, %noc) : (index, index, !ttkernel.local_semaphore, i8) -> !ttkernel.noc_addr
    ttkernel.noc_semaphore_inc(%ready, %one, %noc) : (!ttkernel.noc_addr, index, i8) -> ()
    %completion = ttkernel.get_semaphore(%record) : (index) -> !ttkernel.local_semaphore
    %completion_ptr = ttkernel.reinterpret_cast(%completion) : (!ttkernel.local_semaphore) -> !ttkernel.l1_addr_ptr
    ttkernel.experimental.semaphore_wait_min(%completion_ptr, %sequence) : (!ttkernel.l1_addr_ptr, i32) -> ()
    ttkernel.cb_push_back(%dfb, %pages) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
  } {ttl.pipenet_initial_receive_capacity = 2 : i64}
  return
}

// -----

// A single receive needs no grouped notification.
// CHECK-LABEL: func.func @one_record
// CHECK: scf.for
// CHECK: ttkernel.cb_reserve_back
// CHECK: ttkernel.noc_semaphore_inc
// CHECK-NOT: noc_async_atomic_barrier
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: ttkernel.cb_push_back
// CHECK: return
func.func @one_record(%runtime_upper: index, %external: memref<4xi32>, %condition: i1) {
  %lower = arith.constant 0 : index
  %upper = arith.constant 1 : index
  %step = arith.constant 1 : index
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %pages = arith.constant 1 : i32
  %sequence = arith.constant 1 : i32
  %zero_i32 = arith.constant 0 : i32
  %noc = arith.constant 0 : i8
  %dfb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %ready_semaphore = ttkernel.get_semaphore(%zero) : (index) -> !ttkernel.local_semaphore
  scf.for %record = %lower to %upper step %step {
    ttkernel.cb_reserve_back(%dfb, %pages) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
    %ready = ttkernel.get_noc_addr(%record, %zero, %ready_semaphore, %noc) : (index, index, !ttkernel.local_semaphore, i8) -> !ttkernel.noc_addr
    ttkernel.noc_semaphore_inc(%ready, %one, %noc) : (!ttkernel.noc_addr, index, i8) -> ()
    %completion = ttkernel.get_semaphore(%record) : (index) -> !ttkernel.local_semaphore
    %completion_ptr = ttkernel.reinterpret_cast(%completion) : (!ttkernel.local_semaphore) -> !ttkernel.l1_addr_ptr
    ttkernel.experimental.semaphore_wait_min(%completion_ptr, %sequence) : (!ttkernel.l1_addr_ptr, i32) -> ()
    ttkernel.cb_push_back(%dfb, %pages) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
  } {ttl.pipenet_initial_receive_capacity = 2 : i64}
  return
}

// -----

// Leave a runtime record count sequential.
// CHECK-LABEL: func.func @dynamic_records
// CHECK: scf.for
// CHECK: ttkernel.cb_reserve_back
// CHECK: ttkernel.noc_semaphore_inc
// CHECK-NOT: noc_async_atomic_barrier
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: ttkernel.cb_push_back
// CHECK: return
func.func @dynamic_records(%runtime_upper: index, %external: memref<4xi32>, %condition: i1) {
  %lower = arith.constant 0 : index
  %upper = arith.constant 2 : index
  %step = arith.constant 1 : index
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %pages = arith.constant 1 : i32
  %sequence = arith.constant 1 : i32
  %zero_i32 = arith.constant 0 : i32
  %noc = arith.constant 0 : i8
  %dfb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %ready_semaphore = ttkernel.get_semaphore(%zero) : (index) -> !ttkernel.local_semaphore
  scf.for %record = %lower to %runtime_upper step %step {
    ttkernel.cb_reserve_back(%dfb, %pages) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
    %ready = ttkernel.get_noc_addr(%record, %zero, %ready_semaphore, %noc) : (index, index, !ttkernel.local_semaphore, i8) -> !ttkernel.noc_addr
    ttkernel.noc_semaphore_inc(%ready, %one, %noc) : (!ttkernel.noc_addr, index, i8) -> ()
    %completion = ttkernel.get_semaphore(%record) : (index) -> !ttkernel.local_semaphore
    %completion_ptr = ttkernel.reinterpret_cast(%completion) : (!ttkernel.local_semaphore) -> !ttkernel.l1_addr_ptr
    ttkernel.experimental.semaphore_wait_min(%completion_ptr, %sequence) : (!ttkernel.l1_addr_ptr, i32) -> ()
    ttkernel.cb_push_back(%dfb, %pages) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
  } {ttl.pipenet_initial_receive_capacity = 2 : i64}
  return
}

// -----

// Do not move later notifications ahead of an opaque call.
// CHECK-LABEL: func.func @unknown_effect
// CHECK: scf.for
// CHECK: ttkernel.cb_reserve_back
// CHECK: ttkernel.noc_semaphore_inc
// CHECK-NOT: noc_async_atomic_barrier
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: ttkernel.cb_push_back
// CHECK: return
func.func private @opaque()
func.func @unknown_effect(%runtime_upper: index, %external: memref<4xi32>, %condition: i1) {
  %lower = arith.constant 0 : index
  %upper = arith.constant 2 : index
  %step = arith.constant 1 : index
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %pages = arith.constant 1 : i32
  %sequence = arith.constant 1 : i32
  %zero_i32 = arith.constant 0 : i32
  %noc = arith.constant 0 : i8
  %dfb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %ready_semaphore = ttkernel.get_semaphore(%zero) : (index) -> !ttkernel.local_semaphore
  scf.for %record = %lower to %upper step %step {
    ttkernel.cb_reserve_back(%dfb, %pages) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
    %ready = ttkernel.get_noc_addr(%record, %zero, %ready_semaphore, %noc) : (index, index, !ttkernel.local_semaphore, i8) -> !ttkernel.noc_addr
    ttkernel.noc_semaphore_inc(%ready, %one, %noc) : (!ttkernel.noc_addr, index, i8) -> ()
    func.call @opaque() : () -> ()
    %completion = ttkernel.get_semaphore(%record) : (index) -> !ttkernel.local_semaphore
    %completion_ptr = ttkernel.reinterpret_cast(%completion) : (!ttkernel.local_semaphore) -> !ttkernel.l1_addr_ptr
    ttkernel.experimental.semaphore_wait_min(%completion_ptr, %sequence) : (!ttkernel.l1_addr_ptr, i32) -> ()
    ttkernel.cb_push_back(%dfb, %pages) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
  } {ttl.pipenet_initial_receive_capacity = 2 : i64}
  return
}

// -----

// Externally visible memory updates cannot be reordered with notifications.
// CHECK-LABEL: func.func @external_counter
// CHECK: scf.for
// CHECK: ttkernel.cb_reserve_back
// CHECK: ttkernel.noc_semaphore_inc
// CHECK-NOT: noc_async_atomic_barrier
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: ttkernel.cb_push_back
// CHECK: return
func.func @external_counter(%runtime_upper: index, %external: memref<4xi32>, %condition: i1) {
  %lower = arith.constant 0 : index
  %upper = arith.constant 2 : index
  %step = arith.constant 1 : index
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %pages = arith.constant 1 : i32
  %sequence = arith.constant 1 : i32
  %zero_i32 = arith.constant 0 : i32
  %noc = arith.constant 0 : i8
  %dfb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %ready_semaphore = ttkernel.get_semaphore(%zero) : (index) -> !ttkernel.local_semaphore
  scf.for %record = %lower to %upper step %step {
    ttkernel.cb_reserve_back(%dfb, %pages) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
    %ready = ttkernel.get_noc_addr(%record, %zero, %ready_semaphore, %noc) : (index, index, !ttkernel.local_semaphore, i8) -> !ttkernel.noc_addr
    ttkernel.noc_semaphore_inc(%ready, %one, %noc) : (!ttkernel.noc_addr, index, i8) -> ()
    %count = memref.load %external[%record] : memref<4xi32>
    %next = arith.addi %count, %sequence : i32
    memref.store %next, %external[%record] : memref<4xi32>
    %completion = ttkernel.get_semaphore(%record) : (index) -> !ttkernel.local_semaphore
    %completion_ptr = ttkernel.reinterpret_cast(%completion) : (!ttkernel.local_semaphore) -> !ttkernel.l1_addr_ptr
    ttkernel.experimental.semaphore_wait_min(%completion_ptr, %next) : (!ttkernel.l1_addr_ptr, i32) -> ()
    ttkernel.cb_push_back(%dfb, %pages) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
  } {ttl.pipenet_initial_receive_capacity = 2 : i64}
  return
}

// -----

// A matching low-level protocol alone is not an initial-storage proof.
// CHECK-LABEL: func.func @unmarked_loop
// CHECK: scf.for
// CHECK: ttkernel.cb_reserve_back
// CHECK: ttkernel.noc_semaphore_inc
// CHECK-NOT: noc_async_atomic_barrier
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: ttkernel.cb_push_back
// CHECK: return
func.func @unmarked_loop(%runtime_upper: index, %external: memref<4xi32>, %condition: i1) {
  %lower = arith.constant 0 : index
  %upper = arith.constant 2 : index
  %step = arith.constant 1 : index
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %pages = arith.constant 1 : i32
  %sequence = arith.constant 1 : i32
  %zero_i32 = arith.constant 0 : i32
  %noc = arith.constant 0 : i8
  %dfb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %ready_semaphore = ttkernel.get_semaphore(%zero) : (index) -> !ttkernel.local_semaphore
  scf.for %record = %lower to %upper step %step {
    ttkernel.cb_reserve_back(%dfb, %pages) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
    %ready = ttkernel.get_noc_addr(%record, %zero, %ready_semaphore, %noc) : (index, index, !ttkernel.local_semaphore, i8) -> !ttkernel.noc_addr
    ttkernel.noc_semaphore_inc(%ready, %one, %noc) : (!ttkernel.noc_addr, index, i8) -> ()
    %completion = ttkernel.get_semaphore(%record) : (index) -> !ttkernel.local_semaphore
    %completion_ptr = ttkernel.reinterpret_cast(%completion) : (!ttkernel.local_semaphore) -> !ttkernel.l1_addr_ptr
    ttkernel.experimental.semaphore_wait_min(%completion_ptr, %sequence) : (!ttkernel.l1_addr_ptr, i32) -> ()
    ttkernel.cb_push_back(%dfb, %pages) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
  }
  return
}

// -----

// Conditional callback work requires a separate scheduling proof.
// CHECK-LABEL: func.func @nested_control
// CHECK: scf.for
// CHECK: ttkernel.cb_reserve_back
// CHECK: ttkernel.noc_semaphore_inc
// CHECK-NOT: noc_async_atomic_barrier
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: ttkernel.cb_push_back
// CHECK: return
func.func private @opaque()
func.func @nested_control(%runtime_upper: index, %external: memref<4xi32>, %condition: i1) {
  %lower = arith.constant 0 : index
  %upper = arith.constant 2 : index
  %step = arith.constant 1 : index
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %pages = arith.constant 1 : i32
  %sequence = arith.constant 1 : i32
  %zero_i32 = arith.constant 0 : i32
  %noc = arith.constant 0 : i8
  %dfb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %ready_semaphore = ttkernel.get_semaphore(%zero) : (index) -> !ttkernel.local_semaphore
  scf.for %record = %lower to %upper step %step {
    ttkernel.cb_reserve_back(%dfb, %pages) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
    %ready = ttkernel.get_noc_addr(%record, %zero, %ready_semaphore, %noc) : (index, index, !ttkernel.local_semaphore, i8) -> !ttkernel.noc_addr
    ttkernel.noc_semaphore_inc(%ready, %one, %noc) : (!ttkernel.noc_addr, index, i8) -> ()
    scf.if %condition {
      func.call @opaque() : () -> ()
    }
    %completion = ttkernel.get_semaphore(%record) : (index) -> !ttkernel.local_semaphore
    %completion_ptr = ttkernel.reinterpret_cast(%completion) : (!ttkernel.local_semaphore) -> !ttkernel.l1_addr_ptr
    ttkernel.experimental.semaphore_wait_min(%completion_ptr, %sequence) : (!ttkernel.l1_addr_ptr, i32) -> ()
    ttkernel.cb_push_back(%dfb, %pages) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
  } {ttl.pipenet_initial_receive_capacity = 2 : i64}
  return
}
