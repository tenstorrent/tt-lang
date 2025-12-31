// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

using namespace mlir;

namespace mlir::tt::ttkernel {
void registerTTKernelToCpp();
} // namespace mlir::tt::ttkernel

static void registerCustomTranslations() {
  static bool initOnce = []() {
    mlir::tt::ttkernel::registerTTKernelToCpp();
    return true;
  }();
  (void)initOnce;
}

int main(int argc, char **argv) {
  registerAllTranslations();
  registerCustomTranslations();

  return failed(mlirTranslateMain(argc, argv, "tt-lang translation driver"));
}
