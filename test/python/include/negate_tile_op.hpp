// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// A minimal struct-based compute operation that negates a single tile.
//
// Demonstrates the pattern used by external op libraries: a config struct
// with constexpr CB indices feeds a templated Op class that drives the
// tile register pipeline.  The shim function at the bottom provides the
// plain template-arg interface that ttl.call_extern_func expects.

#pragma once

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#endif

namespace test_ops {

// Config struct -- constexpr fields mirror the pattern used by external op
// libraries that encode CB routing as compile-time args.
template <uint32_t InputCB, uint32_t OutputCB>
struct NegateTileConfig {
  static constexpr uint32_t input_cb = InputCB;
  static constexpr uint32_t output_cb = OutputCB;
};

// Op class -- lifecycle methods consume the config.
template <typename Config>
struct NegateTileOp {
  void init() {
#if defined(COMPILE_FOR_TRISC)
    using namespace ckernel;
    unary_op_init_common(Config::input_cb, Config::output_cb);
    copy_tile_init(Config::input_cb);
#endif
  }

  void run() {
#if defined(COMPILE_FOR_TRISC)
    using namespace ckernel;
    tile_regs_acquire();
    copy_tile(Config::input_cb, 0, 0);

    negative_tile_init();
    negative_tile(0);

    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, Config::output_cb);
    tile_regs_release();
#endif
  }
};

} // namespace test_ops

// The function operands declare the DFB access set; template arguments retain
// the constexpr configuration required by the external operation.
template <uint32_t InCB, uint32_t OutCB>
void negate_tile_shim(int inDfb, int outDfb) {
  (void)inDfb;
  (void)outDfb;
  using Config = test_ops::NegateTileConfig<InCB, OutCB>;
  test_ops::NegateTileOp<Config> op;
  op.init();
  op.run();
}
