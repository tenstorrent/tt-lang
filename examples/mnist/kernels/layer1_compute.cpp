// compute
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "llk_defs.h"
#include "compute_kernel_api/binary_max_min.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/activations.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "compute_kernel_api/eltwise_unary/rounding.h"
#include "compute_kernel_api/eltwise_unary/trigonometry.h"
#include "compute_kernel_api/eltwise_unary/gelu.h"
#include "compute_kernel_api/eltwise_unary/erf_erfc.h"
#include "compute_kernel_api/eltwise_unary/logical_not_noti.h"
#include "compute_kernel_api/eltwise_unary/comp.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"
#include "compute_kernel_api/eltwise_unary/typecast.h"
#include "compute_kernel_api/binary_bitwise_sfpu.h"
#include "compute_kernel_api/eltwise_unary/bitwise_not.h"
#include "compute_kernel_api/eltwise_unary/relu.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/where.h"
inline uint32_t float_to_bits(float f) { uint32_t r; __builtin_memcpy(&r, &f, sizeof(r)); return r; }
#ifndef INFINITY
#define INFINITY __builtin_inff()
#endif
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_COL
#include "compute_kernel_api/reduce.h"
namespace NAMESPACE {
void kernel_main() {
  int32_t v1 = 0;
  int32_t v2 = 4;
  int32_t v3 = 100;
  int32_t v4 = 25;
  size_t v5 = 4;
  size_t v6 = 25;
  size_t v7 = 1;
  size_t v8 = 0;
  cb_wait_front(get_compile_time_arg_val(0), v4);
  cb_wait_front(get_compile_time_arg_val(1), v3);
  cb_reserve_back(get_compile_time_arg_val(3), v2);
  init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(3));
  for (size_t i9 = v8; i9 < v6; i9 += v7) {
    for (size_t j10 = v8; j10 < v5; j10 += v7) {
      tile_regs_acquire();
      mm_init_short(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v1);
      for (size_t k11 = v8; k11 < v6; k11 += v7) {
        size_t v12 = i9 * v6;
        size_t v13 = v12 + k11;
        size_t v14 = k11 * v5;
        size_t v15 = v14 + j10;
        matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v13, v15, v8);
      }
      tile_regs_commit();
      tile_regs_wait();
      size_t v16 = i9 * v5;
      size_t v17 = v16 + j10;
      pack_tile<false>(v8, get_compile_time_arg_val(3), v17);
      tile_regs_release();
    }
  }
  cb_push_back(get_compile_time_arg_val(3), v2);
  cb_pop_front(get_compile_time_arg_val(1), v3);
  cb_pop_front(get_compile_time_arg_val(0), v4);
  cb_wait_front(get_compile_time_arg_val(3), v2);
  cb_wait_front(get_compile_time_arg_val(2), v2);
  cb_reserve_back(get_compile_time_arg_val(4), v2);
  init_sfpu(get_compile_time_arg_val(3), get_compile_time_arg_val(4));
  for (size_t i18 = v8; i18 < v5; i18 += v7) {
    tile_regs_acquire();
    copy_tile_init(get_compile_time_arg_val(3));
    copy_tile(get_compile_time_arg_val(3), i18, v8);
    copy_tile_init(get_compile_time_arg_val(2));
    copy_tile(get_compile_time_arg_val(2), i18, v7);
    add_binary_tile_init();
    add_binary_tile(v8, v7, v8);
    relu_tile_init();
    relu_tile(v8);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<false>(v8, get_compile_time_arg_val(4), i18);
    tile_regs_release();
  }
  cb_push_back(get_compile_time_arg_val(4), v2);
  cb_pop_front(get_compile_time_arg_val(2), v2);
  cb_pop_front(get_compile_time_arg_val(3), v2);
  return;
}
void MAIN { kernel_main(); }
}

