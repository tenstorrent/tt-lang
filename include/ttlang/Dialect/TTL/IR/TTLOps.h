// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPS_H
#define TTLANG_DIALECT_TTL_IR_TTLOPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"

#define GET_OP_CLASSES
#include "ttlang/Dialect/TTL/IR/TTLOps.h.inc"

#endif // TTLANG_DIALECT_TTL_IR_TTLOPS_H
