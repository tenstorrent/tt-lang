// RUN: ttlang-opt --convert-ttkernel-to-emitc %s -o %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.emitc.mlir --check-prefix=EMITC
// RUN: ttlang-translate --ttkernel-to-cpp %t.emitc.mlir 2>&1 | FileCheck %s --check-prefix=CPP

// Verifies that local and runtime-provided semaphore addresses can share one
// SSA address without introducing a C++ conversion.

// EMITC-LABEL: func.func @kernel_main
// EMITC-NOT: cast_to_l1_addr
// EMITC: emitc.conditional

// CPP-LABEL: void kernel_main()
// CPP: get_semaphore
// CPP: get_common_arg_val
// CPP: ? {{.*}} : {{.*}}
func.func @kernel_main() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %semaphore_index = arith.constant 0 : index
  %common_arg_index = arith.constant 1 : index
  %logical_x = "ttkernel.my_logical_x_"() : () -> index
  %condition = arith.cmpi ne, %logical_x, %semaphore_index : index
  %local_address = ttkernel.get_semaphore(%semaphore_index)
      : (index) -> !ttkernel.local_semaphore
  %global_address = ttkernel.get_common_arg_val(%common_arg_index)
      : (index) -> i32
  %typed_local_address = ttkernel.cast_to_l1_addr %local_address
      : !ttkernel.local_semaphore to !ttkernel.l1_addr
  %typed_global_address = ttkernel.cast_to_l1_addr %global_address
      : i32 to !ttkernel.l1_addr
  %selected_address = arith.select %condition, %typed_global_address,
      %typed_local_address : !ttkernel.l1_addr
  %selected_pointer = ttkernel.reinterpret_cast(%selected_address)
      : (!ttkernel.l1_addr) -> !ttkernel.l1_addr_ptr
  %sequence = arith.constant 1 : i32
  ttkernel.experimental.semaphore_wait_min(%selected_pointer, %sequence)
      : (!ttkernel.l1_addr_ptr, i32) -> ()
  return
}
