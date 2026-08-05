// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_DFBANALYSISFAILURE_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_DFBANALYSISFAILURE_H

#include "mlir/IR/Operation.h"

#include <string>
#include <utility>

namespace mlir::tt::ttl {

/// First failure discovered while computing DFB analysis or allocation results.
///
/// Analysis helpers do not emit diagnostics. Retaining the first failure
/// preserves the operation that first violated the input contract.
struct DFBAnalysisFailure {
  Operation *operation = nullptr;
  std::string message;

  /// Records the first failure so traversal order selects diagnostics.
  void set(Operation *failureOperation, std::string failureMessage) {
    if (!message.empty()) {
      return;
    }
    operation = failureOperation;
    message = std::move(failureMessage);
  }
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBANALYSISFAILURE_H
