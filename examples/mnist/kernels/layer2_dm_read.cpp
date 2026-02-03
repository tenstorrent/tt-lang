// dm_read
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  int32_t v1 = 2;
  int32_t v2 = 16;
  size_t v3 = 2;
  int32_t v4 = 15;
  int32_t v5 = 3;
  int32_t v6 = 14;
  size_t v7 = 3;
  int32_t v8 = 0;
  size_t v9 = 2048;
  int32_t v10 = 1;
  int32_t v11 = 13;
  int32_t v12 = 2048;
  int32_t v13 = 4;
  size_t v14 = 4;
  size_t v15 = 8;
  size_t v16 = 1;
  size_t v17 = 0;
  for (size_t i18 = v17; i18 < v15; i18 += v16) {
    size_t v19 = i18 * v14;
    cb_reserve_back(get_compile_time_arg_val(0), v13);
    int32_t v20 = get_common_arg_val<uint32_t>(v16);
    auto tensor_accessor_args_64 = TensorAccessorArgs<13, 1>();
    TensorAccessor v21 = TensorAccessor(tensor_accessor_args_64, v20, v12);
    int32_t v22 = get_write_ptr(get_compile_time_arg_val(0));
    ptrdiff_t v23 = (ptrdiff_t) v22;
    size_t v24 = (size_t) v23;
    for (size_t j25 = v17; j25 < v14; j25 += v16) {
      size_t v26 = v19 + j25;
      size_t v27 = j25 * v9;
      size_t v28 = v24 + v27;
      ptrdiff_t v29 = (ptrdiff_t) v26;
      int32_t v30 = (int32_t) v29;
      ptrdiff_t v31 = (ptrdiff_t) v28;
      int32_t v32 = (int32_t) v31;
      noc_async_read_tile(v30, v21, v32);
    }
    noc_async_read_barrier();
    cb_push_back(get_compile_time_arg_val(0), v13);
    cb_reserve_back(get_compile_time_arg_val(1), v13);
    int32_t v33 = get_common_arg_val<uint32_t>(v7);
    auto tensor_accessor_args_75 = TensorAccessorArgs<14, 3>();
    TensorAccessor v34 = TensorAccessor(tensor_accessor_args_75, v33, v12);
    int32_t v35 = get_write_ptr(get_compile_time_arg_val(1));
    ptrdiff_t v36 = (ptrdiff_t) v35;
    size_t v37 = (size_t) v36;
    for (size_t j38 = v17; j38 < v14; j38 += v16) {
      size_t v39 = v19 + j38;
      size_t v40 = j38 * v9;
      size_t v41 = v37 + v40;
      ptrdiff_t v42 = (ptrdiff_t) v39;
      int32_t v43 = (int32_t) v42;
      ptrdiff_t v44 = (ptrdiff_t) v41;
      int32_t v45 = (int32_t) v44;
      noc_async_read_tile(v43, v34, v45);
    }
    noc_async_read_barrier();
    cb_push_back(get_compile_time_arg_val(1), v13);
  }
  cb_reserve_back(get_compile_time_arg_val(4), v10);
  int32_t v46 = get_common_arg_val<uint32_t>(v17);
  auto tensor_accessor_args_54 = TensorAccessorArgs<15, 0>();
  TensorAccessor v47 = TensorAccessor(tensor_accessor_args_54, v46, v12);
  int32_t v48 = get_write_ptr(get_compile_time_arg_val(4));
  noc_async_read_tile(v8, v47, v48);
  noc_async_read_barrier();
  cb_push_back(get_compile_time_arg_val(4), v10);
  cb_reserve_back(get_compile_time_arg_val(6), v10);
  int32_t v49 = get_common_arg_val<uint32_t>(v3);
  auto tensor_accessor_args_62 = TensorAccessorArgs<16, 2>();
  TensorAccessor v50 = TensorAccessor(tensor_accessor_args_62, v49, v12);
  int32_t v51 = get_write_ptr(get_compile_time_arg_val(6));
  noc_async_read_tile(v8, v50, v51);
  noc_async_read_barrier();
  cb_push_back(get_compile_time_arg_val(6), v10);
  return;
}

