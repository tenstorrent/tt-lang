// compute
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"
inline uint32_t float_to_bits(const float f) { uint32_t r; __builtin_memcpy(&r, &f, sizeof(r)); return r; }
#ifndef INFINITY
#define INFINITY __builtin_inff()
#endif
void kernel_main() {
  float v1 = 2.000000000e+00f;
  size_t v2 = 7;
  size_t v3 = 6;
  size_t v4 = 5;
  size_t v5 = 4;
  size_t v6 = 3;
  size_t v7 = 2;
  int32_t v8 = 4;
  size_t v9 = 1;
  size_t v10 = 0;
  CircularBuffer cb_ctarg_0(get_compile_time_arg_val(0));
  CircularBuffer cb_ctarg_1(get_compile_time_arg_val(1));
  CircularBuffer cb_ctarg_2(get_compile_time_arg_val(2));
  cb_ctarg_0.wait_front(v8);
  cb_ctarg_2.reserve_back(v8);
  init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(2));
  tile_regs_acquire();
  fill_tile_init();
  fill_tile(v10, v1);
  fill_tile(v7, v1);
  fill_tile(v5, v1);
  fill_tile(v3, v1);
  copy_tile_init(get_compile_time_arg_val(0));
  copy_tile(get_compile_time_arg_val(0), v10, v9);
  copy_tile(get_compile_time_arg_val(0), v9, v6);
  copy_tile(get_compile_time_arg_val(0), v7, v4);
  copy_tile(get_compile_time_arg_val(0), v6, v2);
  mul_binary_tile_init();
  mul_binary_tile(v9, v10, v10);
  mul_binary_tile(v6, v7, v7);
  mul_binary_tile(v4, v5, v5);
  mul_binary_tile(v2, v3, v3);
  tile_regs_commit();
  tile_regs_wait();
  pack_tile<true>(v10, get_compile_time_arg_val(2), v10);
  pack_tile<true>(v7, get_compile_time_arg_val(2), v9);
  pack_tile<true>(v5, get_compile_time_arg_val(2), v7);
  pack_tile<true>(v3, get_compile_time_arg_val(2), v6);
  tile_regs_release();
  cb_ctarg_0.pop_front(v8);
  cb_ctarg_1.reserve_back(v8);
  init_sfpu(get_compile_time_arg_val(2), get_compile_time_arg_val(1));
  tile_regs_acquire();
  transpose_wh_init(get_compile_time_arg_val(2), get_compile_time_arg_val(1));
  transpose_wh_tile(get_compile_time_arg_val(2), v10, v10);
  transpose_wh_tile(get_compile_time_arg_val(2), v9, v9);
  transpose_wh_tile(get_compile_time_arg_val(2), v7, v7);
  transpose_wh_tile(get_compile_time_arg_val(2), v6, v6);
  tile_regs_commit();
  tile_regs_wait();
  pack_tile_block(v10, get_compile_time_arg_val(1), v5);
  tile_regs_release();
  cb_ctarg_1.push_back(v8);
  return;
}

