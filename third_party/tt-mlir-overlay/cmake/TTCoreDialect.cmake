# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Custom TTCore dialect build that adds FlatBuffers dependency
# The upstream CMakeLists.txt doesn't have this dependency, but TTCoreOpsTypes.cpp
# includes Target.h which requires FlatBuffers-generated headers.

add_mlir_dialect_library(MLIRTTCoreDialect
    ${ttmlir_SOURCE_DIR}/lib/Dialect/TTCore/IR/TTCoreDialect.cpp
    ${ttmlir_SOURCE_DIR}/lib/Dialect/TTCore/IR/TTCoreOps.cpp
    ${ttmlir_SOURCE_DIR}/lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp
    ${ttmlir_SOURCE_DIR}/lib/Dialect/TTCore/IR/Utils.cpp

    ADDITIONAL_HEADER_DIRS
    ${ttmlir_SOURCE_DIR}/include/ttmlir

    DEPENDS
    MLIRTTCoreOpsIncGen
    COMMON_FBS

    LINK_LIBS PUBLIC
    MLIRQuantDialect
)
