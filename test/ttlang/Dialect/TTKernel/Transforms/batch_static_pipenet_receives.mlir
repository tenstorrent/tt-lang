// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttkernel-batch-static-pipenet-receives,canonicalize)' | FileCheck %s

func.func private @observable_callback()

// CHECK-LABEL: func.func @batch_static
// CHECK-NOT:     scf.for
// CHECK:         %[[THREE:.*]] = arith.constant 3 : i32
// CHECK:         ttkernel.cb_reserve_back(%{{.*}}, %[[THREE]])
// CHECK:         ttkernel.noc_semaphore_inc
// CHECK:         ttkernel.noc_semaphore_inc
// CHECK:         ttkernel.noc_semaphore_inc
// CHECK-NEXT:    ttkernel.noc_async_atomic_barrier
// CHECK:         ttkernel.experimental.semaphore_wait_min
// CHECK-NEXT:    ttkernel.cb_push_back
// CHECK:         ttkernel.experimental.semaphore_wait_min
// CHECK-NEXT:    ttkernel.cb_push_back
// CHECK:         ttkernel.experimental.semaphore_wait_min
// CHECK-NEXT:    ttkernel.cb_push_back
// CHECK-NOT:     scf.for
// CHECK:         return
func.func @batch_static() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %cb = ttkernel.get_compile_time_arg_val(0)
      : () -> !ttkernel.cb<3, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c1_i32 = arith.constant 1 : i32
  %noc = arith.constant 0 : i8
  %address = arith.constant 1024 : i32
  %completion = ttkernel.reinterpret_cast(%address)
      : (i32) -> !ttkernel.l1_addr_ptr
  scf.for %record = %c0 to %c3 step %c1 {
    ttkernel.cb_reserve_back(%cb, %c1_i32)
        : (!ttkernel.cb<3, !ttcore.tile<32x32, bf16>>, i32) -> ()
    %ready = ttkernel.get_noc_addr(%record, %record, %address, %noc)
        : (index, index, i32, i8) -> !ttkernel.noc_addr
    ttkernel.noc_semaphore_inc(%ready, %c1, %noc)
        : (!ttkernel.noc_addr, index, i8) -> ()
    ttkernel.experimental.semaphore_wait_min(%completion, %c1_i32)
        : (!ttkernel.l1_addr_ptr, i32) -> ()
    ttkernel.cb_push_back(%cb, %c1_i32)
        : (!ttkernel.cb<3, !ttcore.tile<32x32, bf16>>, i32) -> ()
  } {ttl.pipenet_local_record_loop, ttl.pipenet_receive_record_loop}
  return
}

// CHECK-LABEL: func.func @retain_observable_callback
// CHECK:         scf.for
// CHECK:           ttkernel.cb_reserve_back
// CHECK:           func.call @observable_callback
// CHECK:           ttkernel.noc_semaphore_inc
// CHECK:           ttkernel.experimental.semaphore_wait_min
// CHECK:           ttkernel.cb_push_back
func.func @retain_observable_callback()
    attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %cb = ttkernel.get_compile_time_arg_val(0)
      : () -> !ttkernel.cb<3, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c1_i32 = arith.constant 1 : i32
  %noc = arith.constant 0 : i8
  %address = arith.constant 1024 : i32
  %ready = ttkernel.get_noc_addr(%c0, %c0, %address, %noc)
      : (index, index, i32, i8) -> !ttkernel.noc_addr
  %completion = ttkernel.reinterpret_cast(%address)
      : (i32) -> !ttkernel.l1_addr_ptr
  scf.for %record = %c0 to %c3 step %c1 {
    ttkernel.cb_reserve_back(%cb, %c1_i32)
        : (!ttkernel.cb<3, !ttcore.tile<32x32, bf16>>, i32) -> ()
    func.call @observable_callback() : () -> ()
    ttkernel.noc_semaphore_inc(%ready, %c1, %noc)
        : (!ttkernel.noc_addr, index, i8) -> ()
    ttkernel.experimental.semaphore_wait_min(%completion, %c1_i32)
        : (!ttkernel.l1_addr_ptr, i32) -> ()
    ttkernel.cb_push_back(%cb, %c1_i32)
        : (!ttkernel.cb<3, !ttcore.tile<32x32, bf16>>, i32) -> ()
  } {ttl.pipenet_local_record_loop, ttl.pipenet_receive_record_loop}
  return
}
