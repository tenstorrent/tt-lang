// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "ttlang/Transforms/Passes.h"

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;

NB_MODULE(_ttlang, m) {
  m.doc() = "tt-lang Python bindings";
  mlir::tt::d2m::registerD2MPasses();
}
