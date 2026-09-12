// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_DFBALLOCATIONDEBUGREPORT_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_DFBALLOCATIONDEBUGREPORT_H

#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::ttl {

class DFBConcurrentKernelLivenessAnalysis;
class DFBPhysicalConflictModel;

/// Print deterministic lifetime and conflict facts used by DFB allocation.
void printDFBAllocationDebugReport(
    llvm::raw_ostream &output,
    const DFBConcurrentKernelLivenessAnalysis &liveness,
    const DFBPhysicalConflictModel &conflictModel);

/// Print deterministic conflict facts used only for storage assignment.
void printDFBStorageConflictDebugReport(
    llvm::raw_ostream &output,
    const DFBPhysicalConflictModel &storageConflictModel);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBALLOCATIONDEBUGREPORT_H
