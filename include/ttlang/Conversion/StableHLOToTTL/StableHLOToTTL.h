// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_CONVERSION_STABLEHLOTOTLL_STABLEHLOTOTLL_H
#define TTLANG_CONVERSION_STABLEHLOTOTLL_STABLEHLOTOTLL_H

#include "mlir/Pass/Pass.h"

namespace mlir::tt::ttl {

std::unique_ptr<mlir::Pass> createConvertStableHLOToTTL();

} // namespace mlir::tt::ttl

#endif // TTLANG_CONVERSION_STABLEHLOTOTLL_STABLEHLOTOTLL_H
