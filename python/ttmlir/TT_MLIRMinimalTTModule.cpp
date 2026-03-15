// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Wrapper that compiles tt-mlir's TTModule.cpp with our minimal header.
// The original includes ttmlir/Bindings/Python/TTMLIRModule.h which pulls
// in the full tt-mlir (TTNN, StableHLO, etc.). We redirect the include.

// Redirect the header include
#define TTMLIR_BINDINGS_PYTHON_TTMLIRMODULE_H
#include "TTMLIRMinimalModule.h"

// Now include the original implementation
#include "TTModule.cpp"
