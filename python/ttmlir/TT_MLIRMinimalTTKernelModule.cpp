// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Wrapper that compiles tt-mlir's TTKernelModule.cpp with our minimal header.

#define TTMLIR_BINDINGS_PYTHON_TTMLIRMODULE_H
#include "TTMLIRMinimalModule.h"

// Include the original implementation from tt-mlir
#include "TTKernelModule.cpp"
