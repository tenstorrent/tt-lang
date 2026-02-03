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
  int32_t v2 = 1;
  int32_t v3 = 4;
  size_t v4 = 4;
  size_t v5 = 7;
  size_t v6 = 1;
  size_t v7 = 0;
  cb_wait_front(get_compile_time_arg_val(0), v3);
  cb_wait_front(get_compile_time_arg_val(1), v3);
  cb_reserve_back(get_compile_time_arg_val(2), v2);
  init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(2));
  for (size_t i8 = v7; i8 < v4; i8 += v6) {
    tile_regs_acquire();
    mm_init_short(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v1);
    for (size_t j9 = v7; j9 < v4; j9 += v6) {
      size_t v10 = i8 * v4;
      size_t v11 = v10 + j9;
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v11, j9, v7);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<false>(v7, get_compile_time_arg_val(2), i8);
    tile_regs_release();
  }
  cb_push_back(get_compile_time_arg_val(2), v2);
  cb_pop_front(get_compile_time_arg_val(1), v3);
  cb_pop_front(get_compile_time_arg_val(0), v3);
  for (size_t i12 = v7; i12 < v5; i12 += v6) {
    cb_wait_front(get_compile_time_arg_val(0), v3);
    cb_wait_front(get_compile_time_arg_val(1), v3);
    cb_reserve_back(get_compile_time_arg_val(3), v2);
    init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(3));
    for (size_t j13 = v7; j13 < v4; j13 += v6) {
      tile_regs_acquire();
      mm_init_short(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v1);
      for (size_t k14 = v7; k14 < v4; k14 += v6) {
        size_t v15 = j13 * v4;
        size_t v16 = v15 + k14;
        matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v16, k14, v7);
      }
      tile_regs_commit();
      tile_regs_wait();
      pack_tile<false>(v7, get_compile_time_arg_val(3), j13);
      tile_regs_release();
    }
    cb_push_back(get_compile_time_arg_val(3), v2);
    cb_pop_front(get_compile_time_arg_val(1), v3);
    cb_pop_front(get_compile_time_arg_val(0), v3);
    cb_wait_front(get_compile_time_arg_val(3), v2);
    cb_wait_front(get_compile_time_arg_val(2), v2);
    cb_reserve_back(get_compile_time_arg_val(2), v2);
    init_sfpu(get_compile_time_arg_val(2), get_compile_time_arg_val(2));
    tile_regs_acquire();
    copy_tile_init(get_compile_time_arg_val(2));
    copy_tile(get_compile_time_arg_val(2), v7, v7);
    copy_tile_init(get_compile_time_arg_val(3));
    copy_tile(get_compile_time_arg_val(3), v7, v6);
    add_binary_tile_init();
    add_binary_tile(v7, v6, v7);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<false>(v7, get_compile_time_arg_val(2), v7);
    tile_regs_release();
    cb_push_back(get_compile_time_arg_val(2), v2);
    cb_pop_front(get_compile_time_arg_val(2), v2);
    cb_pop_front(get_compile_time_arg_val(3), v2);
  }
  cb_wait_front(get_compile_time_arg_val(2), v2);
  cb_wait_front(get_compile_time_arg_val(4), v2);
  cb_reserve_back(get_compile_time_arg_val(5), v2);
  init_sfpu(get_compile_time_arg_val(2), get_compile_time_arg_val(5));
  tile_regs_acquire();
  copy_tile_init(get_compile_time_arg_val(2));
  copy_tile(get_compile_time_arg_val(2), v7, v7);
  copy_tile_init(get_compile_time_arg_val(4));
  copy_tile(get_compile_time_arg_val(4), v7, v6);
  add_binary_tile_init();
  add_binary_tile(v7, v6, v7);
  tile_regs_commit();
  tile_regs_wait();
  pack_tile<false>(v7, get_compile_time_arg_val(5), v7);
  tile_regs_release();
  cb_push_back(get_compile_time_arg_val(5), v2);
  cb_pop_front(get_compile_time_arg_val(4), v2);
  cb_pop_front(get_compile_time_arg_val(2), v2);
  cb_wait_front(get_compile_time_arg_val(5), v2);
  cb_wait_front(get_compile_time_arg_val(6), v2);
  cb_reserve_back(get_compile_time_arg_val(7), v2);
  init_sfpu(get_compile_time_arg_val(5), get_compile_time_arg_val(7));
  tile_regs_acquire();
  reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW, false>(get_compile_time_arg_val(5), get_compile_time_arg_val(6), get_compile_time_arg_val(7));
  reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW, false>(get_compile_time_arg_val(5), get_compile_time_arg_val(6), v7, v7, v7);
  tile_regs_commit();
  tile_regs_wait();
  pack_tile<false>(v7, get_compile_time_arg_val(7), v7);
  tile_regs_release();
  cb_push_back(get_compile_time_arg_val(7), v2);
  cb_wait_front(get_compile_time_arg_val(7), v2);
  cb_reserve_back(get_compile_time_arg_val(8), v2);
  init_sfpu(get_compile_time_arg_val(7), get_compile_time_arg_val(8));
  tile_regs_acquire();
  unary_bcast_init<BroadcastType::COL>(get_compile_time_arg_val(7), get_compile_time_arg_val(8));
  unary_bcast<BroadcastType::COL>(get_compile_time_arg_val(7), v7, v7);
  tile_regs_commit();
  tile_regs_wait();
  pack_tile<false>(v7, get_compile_time_arg_val(8), v7);
  tile_regs_release();
  cb_push_back(get_compile_time_arg_val(8), v2);
  cb_pop_front(get_compile_time_arg_val(7), v2);
  cb_wait_front(get_compile_time_arg_val(8), v2);
  cb_reserve_back(get_compile_time_arg_val(9), v2);
  init_sfpu(get_compile_time_arg_val(5), get_compile_time_arg_val(9));
  tile_regs_acquire();
  copy_tile_init(get_compile_time_arg_val(5));
  copy_tile(get_compile_time_arg_val(5), v7, v7);
  copy_tile_init(get_compile_time_arg_val(8));
  copy_tile(get_compile_time_arg_val(8), v7, v6);
  sub_binary_tile_init();
  sub_binary_tile(v7, v6, v7);
  exp_tile_init();
  exp_tile(v7);
  tile_regs_commit();
  tile_regs_wait();
  pack_tile<false>(v7, get_compile_time_arg_val(9), v7);
  tile_regs_release();
  cb_push_back(get_compile_time_arg_val(9), v2);
  cb_wait_front(get_compile_time_arg_val(9), v2);
  cb_reserve_back(get_compile_time_arg_val(10), v2);
  init_sfpu(get_compile_time_arg_val(9), get_compile_time_arg_val(10));
  tile_regs_acquire();
  reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW, false>(get_compile_time_arg_val(9), get_compile_time_arg_val(6), get_compile_time_arg_val(10));
  reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW, false>(get_compile_time_arg_val(9), get_compile_time_arg_val(6), v7, v7, v7);
  tile_regs_commit();
  tile_regs_wait();
  pack_tile<false>(v7, get_compile_time_arg_val(10), v7);
  tile_regs_release();
  cb_push_back(get_compile_time_arg_val(10), v2);
  cb_pop_front(get_compile_time_arg_val(9), v2);
  cb_wait_front(get_compile_time_arg_val(10), v2);
  cb_reserve_back(get_compile_time_arg_val(11), v2);
  init_sfpu(get_compile_time_arg_val(10), get_compile_time_arg_val(11));
  tile_regs_acquire();
  unary_bcast_init<BroadcastType::COL>(get_compile_time_arg_val(10), get_compile_time_arg_val(11));
  unary_bcast<BroadcastType::COL>(get_compile_time_arg_val(10), v7, v7);
  tile_regs_commit();
  tile_regs_wait();
  pack_tile<false>(v7, get_compile_time_arg_val(11), v7);
  tile_regs_release();
  cb_push_back(get_compile_time_arg_val(11), v2);
  cb_pop_front(get_compile_time_arg_val(10), v2);
  cb_wait_front(get_compile_time_arg_val(11), v2);
  cb_reserve_back(get_compile_time_arg_val(12), v2);
  init_sfpu(get_compile_time_arg_val(5), get_compile_time_arg_val(12));
  tile_regs_acquire();
  copy_tile_init(get_compile_time_arg_val(5));
  copy_tile(get_compile_time_arg_val(5), v7, v7);
  copy_tile_init(get_compile_time_arg_val(8));
  copy_tile(get_compile_time_arg_val(8), v7, v6);
  sub_binary_tile_init();
  sub_binary_tile(v7, v6, v7);
  exp_tile_init();
  exp_tile(v7);
  copy_tile_init(get_compile_time_arg_val(11));
  copy_tile(get_compile_time_arg_val(11), v7, v6);
  div_binary_tile_init();
  div_binary_tile(v7, v6, v7);
  tile_regs_commit();
  tile_regs_wait();
  pack_tile<false>(v7, get_compile_time_arg_val(12), v7);
  tile_regs_release();
  cb_push_back(get_compile_time_arg_val(12), v2);
  cb_pop_front(get_compile_time_arg_val(11), v2);
  cb_pop_front(get_compile_time_arg_val(8), v2);
  cb_pop_front(get_compile_time_arg_val(6), v2);
  cb_pop_front(get_compile_time_arg_val(5), v2);
  return;
}
void MAIN { kernel_main(); }
}

