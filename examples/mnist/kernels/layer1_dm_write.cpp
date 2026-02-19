// dm_write
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  size_t v1 = 1;
  size_t v2 = 2048;
  int32_t v3 = 0;
  int32_t v4 = 8;
  int32_t v5 = 2048;
  int32_t v6 = 4;
  size_t v7 = 4;
  size_t v8 = 0;
  size_t v9 = get_absolute_logical_x();
  size_t v10 = v9 * v7;
  cb_wait_front(get_compile_time_arg_val(4), v6);
  int32_t v11 = get_common_arg_val<uint32_t>(v8);
  auto tensor_accessor_args_16 = TensorAccessorArgs<8, 0>();
  TensorAccessor v12 = TensorAccessor(tensor_accessor_args_16, v11, v5);
  int32_t v13 = get_read_ptr(get_compile_time_arg_val(4));
  ptrdiff_t v14 = (ptrdiff_t) v13;
  size_t v15 = (size_t) v14;
  for (size_t i16 = v8; i16 < v7; i16 += v1) {
    size_t v17 = v10 + i16;
    size_t v18 = i16 * v2;
    size_t v19 = v15 + v18;
    ptrdiff_t v20 = (ptrdiff_t) v17;
    int32_t v21 = (int32_t) v20;
    ptrdiff_t v22 = (ptrdiff_t) v19;
    int32_t v23 = (int32_t) v22;
    noc_async_write_tile(v21, v12, v23);
  }
  noc_async_write_barrier();
  cb_pop_front(get_compile_time_arg_val(4), v6);
  return;
}

