// dm_read
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  int32_t v1 = 7;
  int32_t v2 = 4;
  size_t v3 = 32;
  int32_t v4 = 1;
  int32_t v5 = 6;
  int32_t v6 = 100;
  int32_t v7 = 0;
  size_t v8 = 25;
  size_t v9 = 1;
  size_t v10 = 2048;
  int32_t v11 = 2;
  int32_t v12 = 5;
  int32_t v13 = 2048;
  size_t v14 = 2;
  int32_t v15 = 25;
  size_t v16 = 4;
  size_t v17 = 0;
  size_t v18 = get_absolute_logical_x();
  cb_reserve_back(get_compile_time_arg_val(0), v15);
  int32_t v19 = get_common_arg_val<uint32_t>(v14);
  auto tensor_accessor_args_27 = TensorAccessorArgs<5, 2>();
  TensorAccessor v20 = TensorAccessor(tensor_accessor_args_27, v19, v13);
  int32_t v21 = get_write_ptr(get_compile_time_arg_val(0));
  ptrdiff_t v22 = (ptrdiff_t) v21;
  size_t v23 = (size_t) v22;
  for (size_t i24 = v17; i24 < v8; i24 += v9) {
    size_t v25 = i24 * v10;
    size_t v26 = v23 + v25;
    ptrdiff_t v27 = (ptrdiff_t) i24;
    int32_t v28 = (int32_t) v27;
    ptrdiff_t v29 = (ptrdiff_t) v26;
    int32_t v30 = (int32_t) v29;
    noc_async_read_tile(v28, v20, v30);
  }
  noc_async_read_barrier();
  cb_push_back(get_compile_time_arg_val(0), v15);
  size_t v31 = v18 * v16;
  cb_reserve_back(get_compile_time_arg_val(1), v6);
  int32_t v32 = get_common_arg_val<uint32_t>(v9);
  auto tensor_accessor_args_78 = TensorAccessorArgs<6, 1>();
  TensorAccessor v33 = TensorAccessor(tensor_accessor_args_78, v32, v13);
  int32_t v34 = get_write_ptr(get_compile_time_arg_val(1));
  ptrdiff_t v35 = (ptrdiff_t) v34;
  size_t v36 = (size_t) v35;
  for (size_t i37 = v17; i37 < v8; i37 += v9) {
    for (size_t j38 = v17; j38 < v16; j38 += v9) {
      size_t v39 = v31 + j38;
      size_t v40 = i37 * v3;
      size_t v41 = v40 + v39;
      size_t v42 = i37 * v16;
      size_t v43 = v42 + j38;
      size_t v44 = v43 * v10;
      size_t v45 = v36 + v44;
      ptrdiff_t v46 = (ptrdiff_t) v41;
      int32_t v47 = (int32_t) v46;
      ptrdiff_t v48 = (ptrdiff_t) v45;
      int32_t v49 = (int32_t) v48;
      noc_async_read_tile(v47, v33, v49);
    }
  }
  noc_async_read_barrier();
  cb_push_back(get_compile_time_arg_val(1), v6);
  cb_reserve_back(get_compile_time_arg_val(2), v2);
  int32_t v50 = get_common_arg_val<uint32_t>(v17);
  auto tensor_accessor_args_89 = TensorAccessorArgs<7, 0>();
  TensorAccessor v51 = TensorAccessor(tensor_accessor_args_89, v50, v13);
  int32_t v52 = get_write_ptr(get_compile_time_arg_val(2));
  ptrdiff_t v53 = (ptrdiff_t) v52;
  size_t v54 = (size_t) v53;
  for (size_t i55 = v17; i55 < v16; i55 += v9) {
    size_t v56 = v31 + i55;
    size_t v57 = i55 * v10;
    size_t v58 = v54 + v57;
    ptrdiff_t v59 = (ptrdiff_t) v56;
    int32_t v60 = (int32_t) v59;
    ptrdiff_t v61 = (ptrdiff_t) v58;
    int32_t v62 = (int32_t) v61;
    noc_async_read_tile(v60, v51, v62);
  }
  noc_async_read_barrier();
  cb_push_back(get_compile_time_arg_val(2), v2);
  return;
}

