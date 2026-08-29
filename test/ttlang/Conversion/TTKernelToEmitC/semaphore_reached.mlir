// RUN: ttlang-opt --convert-ttkernel-to-emitc -o %t.emitc.mlir %s
// RUN: FileCheck %s --input-file=%t.emitc.mlir --check-prefix=EMITC
// RUN: ttlang-translate --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP

// Verifies that a nonblocking semaphore threshold test lowers to the
// cache-coherent experimental implementation.

// EMITC-LABEL: func.func @kernel_main
// EMITC: %[[REACHED:.*]] = emitc.call_opaque "experimental::semaphore_reached"

// CPP: bool semaphore_reached(
// CPP: invalidate_l1_cache();
// CPP-NEXT: return *sem_addr >= val;
// CPP-LABEL: void kernel_main()
// CPP: bool {{.*}} = experimental::semaphore_reached(
func.func @kernel_main() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %address = arith.constant 4096 : i32
  %sequence = arith.constant 3 : i32
  %typed_address = ttkernel.cast_to_l1_addr %address : i32 to !ttkernel.l1_addr
  %pointer = ttkernel.reinterpret_cast(%typed_address)
      : (!ttkernel.l1_addr) -> !ttkernel.l1_addr_ptr
  %reached = ttkernel.experimental.semaphore_reached(%pointer, %sequence)
      : (!ttkernel.l1_addr_ptr, i32) -> i1
  return
}
