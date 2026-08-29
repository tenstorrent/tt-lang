// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Pipe Transfer Provenance
//===----------------------------------------------------------------------===//
//
// This file declares SSA provenance queries for pipe transfer handles, tokens,
// and transfer objects, plus module verification for malformed provenance.
//
//===----------------------------------------------------------------------===//

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_TRANSFERPROVENANCE_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_TRANSFERPROVENANCE_H

#include "ttlang/Analysis/ValueOriginAnalysis.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::tt::ttl {

/// Returns the transfer creation op shared by every possible origin.
FailureOr<PipeTransferCreateOp>
findPipeTransferCreateForTransfer(ValueOriginAnalysis &analysis,
                                  Value transfer);

/// Returns the unique pipe receive whose handle may reach `value`. Returns no
/// receive when none of the possible origins is a pipe receive, and failure
/// when pipe and non-pipe origins are mixed or distinct receives are possible.
FailureOr<std::optional<CopyOp>>
findUniquePipeReceiveCopy(ValueOriginAnalysis &analysis, Value value);

/// Returns every high-level pipe receive whose request may reach `value`.
/// Fails unless at least one receive and no other origin reaches the value.
FailureOr<SmallVector<CopyOp>>
findPipeReceiveCopies(ValueOriginAnalysis &analysis, Value value);

/// Returns every internal receive post whose token may reach `token`. Fails
/// unless at least one post reaches the token.
FailureOr<SmallVector<PipeTransferPostOp>>
findPipeTransferPostsForToken(ValueOriginAnalysis &analysis, Value token);

/// Returns the transfer creation shared by `posts`.
FailureOr<PipeTransferCreateOp>
findPipeTransferCreateForPosts(ValueOriginAnalysis &analysis,
                               ArrayRef<PipeTransferPostOp> posts);

/// Validate non-local transfer, handle, and token provenance.
LogicalResult verifyTransferProvenance(ModuleOp module);
LogicalResult verifyTransferProvenance(ModuleOp module,
                                       ValueOriginAnalysis &analysis);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_TRANSFERPROVENANCE_H
