// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Standalone metal compute kernel for the resident-intermediate PACK->UNPACK
// question. Loops NITERS times, each: cb_in -> DST (copy + *2) -> pack -> cb2
// (resident scratch, reused) -> transpose_wh -> cb_out. cb2 is never pushed/
// popped, so a PACK->UNPACK race would read the PRIOR iteration's stale cb2.
//
// Toggles (-D): HANDSHAKE (full cb2 push/wait/pop), BARRIER (UNPACK STALLWAIT),
// NOPS=N (N unpacker NOPs). None => bare scratchpad.

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

namespace {
// Deterministic PACK->UNPACK ordering: stall the unpacker pipe until the
// packers are idle (the pending pack has flushed to L1). Same primitive
// cb_push_back uses (STALL_THCON, PACK), retargeted to the unpack pipe.
// Wrapped in a function so UNPACK((...)) sees a call expression, not asm.
inline void unpack_after_pack_barrier() {
  TTI_STALLWAIT(ckernel::p_stall::STALL_UNPACK, ckernel::p_stall::PACK);
}
inline void unpack_nops(int n) {
  for (int i = 0; i < n; ++i) {
    TTI_NOP;
  }
}
}  // namespace

void kernel_main() {
  constexpr uint32_t cb_in = get_compile_time_arg_val(0);
  constexpr uint32_t cb_out = get_compile_time_arg_val(1);
  constexpr uint32_t cb_scratch = get_compile_time_arg_val(2);
  constexpr uint32_t NITERS = get_compile_time_arg_val(3);
  constexpr float two = 2.0f;
  constexpr uint32_t NT = 4;

  CircularBuffer cb_i(cb_in);
  CircularBuffer cb_o(cb_out);
  CircularBuffer cb_s(cb_scratch);

  for (uint32_t it = 0; it < NITERS; it++) {
    cb_i.wait_front(NT);
    cb_s.reserve_back(NT);

    // Producer: cb_in -> DST, *2 (SFPU), pack to resident scratch L1.
    init_sfpu(cb_in, cb_scratch);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, two);
    fill_tile(1, two);
    fill_tile(2, two);
    fill_tile(3, two);
    copy_tile_init(cb_in);
    copy_tile(cb_in, 0, 4);
    copy_tile(cb_in, 1, 5);
    copy_tile(cb_in, 2, 6);
    copy_tile(cb_in, 3, 7);
    mul_binary_tile_init();
    mul_binary_tile(4, 0, 0);
    mul_binary_tile(5, 1, 1);
    mul_binary_tile(6, 2, 2);
    mul_binary_tile(7, 3, 3);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, cb_scratch, 0);
    pack_tile<true>(1, cb_scratch, 1);
    pack_tile<true>(2, cb_scratch, 2);
    pack_tile<true>(3, cb_scratch, 3);
    tile_regs_release();
    cb_i.pop_front(NT);

#if defined(HANDSHAKE)
    cb_s.push_back(NT);
    cb_s.wait_front(NT);
#elif defined(BARRIER)
    UNPACK((unpack_after_pack_barrier()));
#elif defined(NOPS)
    UNPACK((unpack_nops(NOPS)));
#endif

    cb_o.reserve_back(NT);

    // Consumer: read scratch L1 via transpose unpack -> DST -> pack to cb_out.
    init_sfpu(cb_scratch, cb_out);
    tile_regs_acquire();
    transpose_wh_init(cb_scratch, cb_out);
    transpose_wh_tile(cb_scratch, 0, 0);
    transpose_wh_tile(cb_scratch, 1, 1);
    transpose_wh_tile(cb_scratch, 2, 2);
    transpose_wh_tile(cb_scratch, 3, 3);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile_block(0, cb_out, NT);
    tile_regs_release();

#if defined(HANDSHAKE)
    cb_s.pop_front(NT);
#endif
    cb_o.push_back(NT);
  }
}
