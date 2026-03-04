// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

namespace mlir::tt::ttkernel {
void registerTTKernelToCpp();
} // namespace mlir::tt::ttkernel

int main(int argc, char **argv) {
  mlir::tt::ttkernel::registerTTKernelToCpp();

  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "tt-lang translation driver"));
}
