// dm_write
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  int32_t v1 = 0;
  int32_t v2 = 17;
  int32_t v3 = 2048;
  int32_t v4 = 1;
  size_t v5 = 0;
  cb_wait_front(get_compile_time_arg_val(12), v4);
  int32_t v6 = get_common_arg_val<uint32_t>(v5);
  auto tensor_accessor_args_9 = TensorAccessorArgs<17, 0>();
  TensorAccessor v7 = TensorAccessor(tensor_accessor_args_9, v6, v3);
  int32_t v8 = get_read_ptr(get_compile_time_arg_val(12));
  noc_async_write_tile(v1, v7, v8);
  noc_async_write_barrier();
  cb_pop_front(get_compile_time_arg_val(12), v4);
  return;
}

