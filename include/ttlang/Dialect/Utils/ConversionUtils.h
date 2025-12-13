// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_UTILS_CONVERSIONUTILS_H
#define TTLANG_DIALECT_UTILS_CONVERSIONUTILS_H

#include "mlir/IR/Diagnostics.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Twine.h"

namespace mlir::tt::ttl::utils {

/// Runs applyPartialConversion while capturing the first diagnostic emitted
/// during conversion. Returns true on failure and populates `capturedDiag`
/// with either the captured diagnostic or a generic message that includes the
/// pass name.
inline bool
applyPartialConversionWithDiag(Operation *root, ConversionTarget &target,
                               const FrozenRewritePatternSet &patterns,
                               StringRef passName, std::string &capturedDiag) {
  bool failedConv = false;
  {
    ScopedDiagnosticHandler handler(root->getContext(), [&](Diagnostic &diag) {
      if (capturedDiag.empty()) {
        capturedDiag = diag.str();
      }
      return success();
    });
    failedConv = failed(applyPartialConversion(root, target, patterns));
  }

  if (failedConv && capturedDiag.empty()) {
    capturedDiag =
        (llvm::Twine(passName) + " failed during legalization").str();
  }
  return failedConv;
}

} // namespace mlir::tt::ttl::utils

#endif // TTLANG_DIALECT_UTILS_CONVERSIONUTILS_H
