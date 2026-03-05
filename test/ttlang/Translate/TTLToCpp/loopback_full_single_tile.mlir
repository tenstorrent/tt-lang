// RUN: ttlang-opt --ttl-to-ttkernel-pipeline --canonicalize %s -o %t.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// Test: Full loopback pattern (read from DRAM, write back to DRAM)
// Validates complete pattern matching production kernel structure

#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                      buffer = dram, grid = [1, 1], memory = interleaved>

// CHECK: // loopback
// CHECK: void kernel_main() {
// CHECK-DAG:   int32_t [[ZERO:v[0-9]+]] = 0;
// CHECK-DAG:   int32_t [[ADDR:v[0-9]+]] = 4096;
// CHECK-DAG:   size_t [[STEP:v[0-9]+]] = 1;
// CHECK-DAG:   size_t [[UB:v[0-9]+]] = 4;
// CHECK-DAG:   size_t [[LB:v[0-9]+]] = 0;
// CHECK:   for (size_t [[IV:i[0-9]+]] = [[LB]]; [[IV]] < [[UB]]; [[IV]] += [[STEP]]) {
// Read: tensor -> CB (uses get_write_ptr for CB destination, with cast chain)
// CHECK:     int32_t [[RT_ARG_R:v[0-9]+]] = get_common_arg_val<uint32_t>([[LB]]);
// CHECK:     auto [[ARGS_READ:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<1, 0>();
// CHECK:     TensorAccessor [[ACC_READ:v[0-9]+]] = TensorAccessor([[ARGS_READ]], [[RT_ARG_R]], [[ADDR]]);
// CHECK:     int32_t [[CB_WRITE_PTR:v[0-9]+]] = get_write_ptr(get_compile_time_arg_val(0));
// CHECK-NEXT:     ptrdiff_t [[RC1:v[0-9]+]] = (ptrdiff_t) [[CB_WRITE_PTR]];
// CHECK-NEXT:     size_t [[RC2:v[0-9]+]] = (size_t) [[RC1]];
// CHECK-NEXT:     ptrdiff_t [[RC3:v[0-9]+]] = (ptrdiff_t) [[RC2]];
// CHECK-NEXT:     int32_t [[RC4:v[0-9]+]] = (int32_t) [[RC3]];
// CHECK-NEXT:     noc_async_read_tile([[ZERO]], [[ACC_READ]], [[RC4]]);
// CHECK:     noc_async_read_barrier();
// Write: CB -> tensor (uses get_read_ptr for CB source, with cast chain)
// CHECK:     int32_t [[RT_ARG_W:v[0-9]+]] = get_common_arg_val<uint32_t>([[STEP]]);
// CHECK:     auto [[ARGS_WRITE:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<2, 1>();
// CHECK:     TensorAccessor [[ACC_WRITE:v[0-9]+]] = TensorAccessor([[ARGS_WRITE]], [[RT_ARG_W]], [[ADDR]]);
// CHECK:     int32_t [[CB_READ_PTR:v[0-9]+]] = get_read_ptr(get_compile_time_arg_val(0));
// CHECK-NEXT:     ptrdiff_t [[WC1:v[0-9]+]] = (ptrdiff_t) [[CB_READ_PTR]];
// CHECK-NEXT:     size_t [[WC2:v[0-9]+]] = (size_t) [[WC1]];
// CHECK-NEXT:     ptrdiff_t [[WC3:v[0-9]+]] = (ptrdiff_t) [[WC2]];
// CHECK-NEXT:     int32_t [[WC4:v[0-9]+]] = (int32_t) [[WC3]];
// CHECK-NEXT:     noc_async_write_tile([[ZERO]], [[ACC_WRITE]], [[WC4]]);
// CHECK:     noc_async_write_barrier();
// CHECK:   }
// CHECK:   return;
// CHECK-NEXT: }
module {
  func.func @loopback(%src: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, %dst: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0, 1], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index

    %src_slice = ttl.tensor_slice %src[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %dst_slice = ttl.tensor_slice %dst[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    scf.for %i = %c0 to %c4 step %c1 {
      %xf_read = ttl.copy %src_slice, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
      ttl.wait %xf_read : !ttl.transfer_handle<read>
      %xf_write = ttl.copy %cb, %dst_slice : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<write>
      ttl.wait %xf_write : !ttl.transfer_handle<write>
    }
    func.return
  }
}
