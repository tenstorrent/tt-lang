// RUN: ttlang-opt --convert-ttl-to-ttkernel --canonicalize -cse %s -o %t.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

#layout = #ttl.layout<shape = [5, 7168], element_type = bf16,
                      buffer = dram, grid = [1, 1], memory = interleaved>

// CHECK-LABEL: void kernel_main()
// CHECK-DAG: int32_t [[TRANSFER_SIZE:v[0-9]+]] = 14336;
// CHECK: .async_read({{.*}}, CoreLocalMem<uint32_t>({{.*}}), [[TRANSFER_SIZE]], {{.*}})
// CHECK-NOT: .async_read({{.*}}get_aligned_page_size()
module {
  func.func @kernel_main(%source: tensor<5x7168xbf16, #layout>)
      attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0],
                  ttl.kernel_thread = #ttkernel.thread<noc>, ttl.noc_index = 0 : i64} {
    %page = arith.constant 4 : index
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 224], !ttcore.tile<1x32, bf16>, 2>
    %xf = ttl.copy_tensor_page %source[%page], %dfb
        : tensor<5x7168xbf16, #layout>,
          !ttl.cb<[1, 224], !ttcore.tile<1x32, bf16>, 2>
        -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    return
  }
}
