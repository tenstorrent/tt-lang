// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPENETFOREACHLOWERING_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPENETFOREACHLOWERING_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir::tt::ttl {

struct PipeForeachLoweringInfo;

void lowerPipeNetForeachOps(ModuleOp module,
                            PipeForeachLoweringInfo &foreachLoweringInfo);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPENETFOREACHLOWERING_H
