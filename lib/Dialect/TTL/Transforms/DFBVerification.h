// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_LIB_DIALECT_TTL_TRANSFORMS_DFBVERIFICATION_H
#define TTLANG_LIB_DIALECT_TTL_TRANSFORMS_DFBVERIFICATION_H

#include "mlir/IR/BuiltinOps.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"

#include <cstdlib>

namespace mlir::tt::ttl {

/// Returns whether protocol-domain verification is relaxed, recording one
/// module marker and warning so the output cannot appear fully verified.
inline bool applyDFBProtocolDomainVerificationRelaxation(ModuleOp module) {
  if (std::getenv("TTL_RELAX_DFB_SPSC") == nullptr) {
    return false;
  }
  if (!module->hasAttr(kRelaxedDFBProtocolDomainVerificationAttrName)) {
    module->setAttr(kRelaxedDFBProtocolDomainVerificationAttrName,
                    UnitAttr::get(module.getContext()));
    module.emitWarning()
        << "`TTL_RELAX_DFB_SPSC` disables per-launch-node DFB producer, "
           "consumer, and wait correspondence checks; the program must "
           "enforce the omitted ownership and synchronization requirements";
  }
  return true;
}

} // namespace mlir::tt::ttl

#endif // TTLANG_LIB_DIALECT_TTL_TRANSFORMS_DFBVERIFICATION_H
