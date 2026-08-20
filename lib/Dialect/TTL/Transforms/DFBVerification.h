// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_LIB_DIALECT_TTL_TRANSFORMS_DFBVERIFICATION_H
#define TTLANG_LIB_DIALECT_TTL_TRANSFORMS_DFBVERIFICATION_H

#include <cstdlib>

namespace mlir::tt::ttl {

/// Treats any defined `TTL_RELAX_DFB_SPSC` value as enabling relaxation.
inline bool isDFBProtocolDomainVerificationRelaxed() {
  return std::getenv("TTL_RELAX_DFB_SPSC") != nullptr;
}

} // namespace mlir::tt::ttl

#endif // TTLANG_LIB_DIALECT_TTL_TRANSFORMS_DFBVERIFICATION_H
