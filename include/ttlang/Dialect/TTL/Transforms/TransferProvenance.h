// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_TRANSFERPROVENANCE_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_TRANSFERPROVENANCE_H

#include "ttlang/Analysis/ValueOriginAnalysis.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttl {

/// Return the transfer creation op shared by every possible origin.
FailureOr<PipeTransferCreateOp>
findPipeTransferCreateForTransfer(ValueOriginAnalysis &analysis,
                                  Value transfer);

/// Validate non-local transfer, handle, and token provenance.
LogicalResult verifyTransferProvenance(ModuleOp module);
LogicalResult verifyTransferProvenance(ModuleOp module,
                                       ValueOriginAnalysis &analysis);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_TRANSFERPROVENANCE_H
