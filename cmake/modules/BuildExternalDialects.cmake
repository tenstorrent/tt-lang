# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# BuildExternalDialects.cmake - Build StableHLO and Shardy dialect IR
# definitions from submodules.
#
# StableHLO has native CMake support and uses STABLEHLO_BUILD_EMBEDDED mode.
# Shardy is Bazel-only, so we provide CMake TableGen + library targets here.
#
# Only dialect IR definitions are built (op/type/attribute classes).
# No transform passes from either project.

# ---------------------------------------------------------------------------
# StableHLO (has CMake, use add_subdirectory in embedded mode)
# ---------------------------------------------------------------------------
set(STABLEHLO_BUILD_EMBEDDED ON CACHE BOOL "" FORCE)
set(STABLEHLO_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "" FORCE)

set(STABLEHLO_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third-party/stablehlo")
ttlang_ensure_submodules(third-party/stablehlo)

include_directories(SYSTEM "${STABLEHLO_SOURCE_DIR}")

add_subdirectory(
  "${STABLEHLO_SOURCE_DIR}"
  "${CMAKE_BINARY_DIR}/stablehlo"
  EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# Shardy (Bazel-only upstream, we provide CMake targets)
# ---------------------------------------------------------------------------
set(SHARDY_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third-party/shardy")
ttlang_ensure_submodules(third-party/shardy)

include_directories(SYSTEM "${SHARDY_SOURCE_DIR}")

set(SHARDY_IR_DIR "${SHARDY_SOURCE_DIR}/shardy/dialect/sdy/ir")
set(SHARDY_BINARY_DIR "${CMAKE_BINARY_DIR}/shardy/dialect/sdy/ir")

# TableGen include paths for Shardy .td files
set(SHARDY_TABLEGEN_FLAGS
  "-I${SHARDY_IR_DIR}"
  "-I${SHARDY_SOURCE_DIR}"
  "-I${STABLEHLO_SOURCE_DIR}")

# --- TableGen targets ---
# Each uses mlir_tablegen() with SHARDY_BINARY_DIR as output location.

set(LLVM_TARGET_DEFINITIONS "${SHARDY_IR_DIR}/dialect.td")
set(SAVED_TABLEGEN_FLAGS ${MLIR_TABLEGEN_FLAGS})
list(APPEND MLIR_TABLEGEN_FLAGS ${SHARDY_TABLEGEN_FLAGS})
mlir_tablegen("${SHARDY_BINARY_DIR}/dialect.h.inc" -gen-dialect-decls)
mlir_tablegen("${SHARDY_BINARY_DIR}/dialect.cc.inc" -gen-dialect-defs)
add_public_tablegen_target(ShardyDialectIncGen)
add_dependencies(mlir-headers ShardyDialectIncGen)

set(LLVM_TARGET_DEFINITIONS "${SHARDY_IR_DIR}/ops.td")
mlir_tablegen("${SHARDY_BINARY_DIR}/ops.h.inc" -gen-op-decls)
mlir_tablegen("${SHARDY_BINARY_DIR}/ops.cc.inc" -gen-op-defs)
add_public_tablegen_target(ShardyOpsIncGen)
add_dependencies(mlir-headers ShardyOpsIncGen)

set(LLVM_TARGET_DEFINITIONS "${SHARDY_IR_DIR}/attrs.td")
mlir_tablegen("${SHARDY_BINARY_DIR}/attrs.h.inc" -gen-attrdef-decls)
mlir_tablegen("${SHARDY_BINARY_DIR}/attrs.cc.inc" -gen-attrdef-defs)
add_public_tablegen_target(ShardyAttrsIncGen)
add_dependencies(mlir-headers ShardyAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS "${SHARDY_IR_DIR}/enums.td")
mlir_tablegen("${SHARDY_BINARY_DIR}/enums.h.inc" -gen-enum-decls)
mlir_tablegen("${SHARDY_BINARY_DIR}/enums.cc.inc" -gen-enum-defs)
add_public_tablegen_target(ShardyEnumsIncGen)
add_dependencies(mlir-headers ShardyEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS "${SHARDY_IR_DIR}/op_interface.td")
mlir_tablegen("${SHARDY_BINARY_DIR}/op_interface.h.inc" -gen-op-interface-decls)
mlir_tablegen("${SHARDY_BINARY_DIR}/op_interface.cc.inc" -gen-op-interface-defs)
add_public_tablegen_target(ShardyOpInterfaceIncGen)
add_dependencies(mlir-headers ShardyOpInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS "${SHARDY_IR_DIR}/canonicalization.td")
mlir_tablegen("${SHARDY_BINARY_DIR}/canonicalization.cc.inc" -gen-rewriters)
add_public_tablegen_target(ShardyCanonicalizationIncGen)
add_dependencies(mlir-headers ShardyCanonicalizationIncGen)

set(LLVM_TARGET_DEFINITIONS "${SHARDY_IR_DIR}/bytecode.td")
mlir_tablegen("${SHARDY_BINARY_DIR}/bytecode.cc.inc" -gen-bytecode "-bytecode-dialect=sdy")
add_public_tablegen_target(ShardyBytecodeIncGen)
add_dependencies(mlir-headers ShardyBytecodeIncGen)

# Restore original tablegen flags
set(MLIR_TABLEGEN_FLAGS ${SAVED_TABLEGEN_FLAGS})

# --- Shardy common/logging library ---
add_library(ShardyCommonLogging STATIC
  "${SHARDY_SOURCE_DIR}/shardy/common/logging.cc"
)
target_include_directories(ShardyCommonLogging PUBLIC "${SHARDY_SOURCE_DIR}")
target_link_libraries(ShardyCommonLogging PUBLIC LLVMSupport MLIRSupport)

# --- Shardy dialect library ---
# Generated .inc files go to SHARDY_BINARY_DIR. Source files include them
# relative to the source tree, so we add both as include dirs.
add_mlir_dialect_library(MLIRSdyDialect
  "${SHARDY_IR_DIR}/bytecode.cc"
  "${SHARDY_IR_DIR}/canonicalization.cc"
  "${SHARDY_IR_DIR}/compatibility.cc"
  "${SHARDY_IR_DIR}/dialect.cc"
  "${SHARDY_IR_DIR}/extensions/stablehlo_extensions.cc"
  "${SHARDY_IR_DIR}/parsers.cc"
  "${SHARDY_IR_DIR}/printers.cc"
  "${SHARDY_IR_DIR}/utils.cc"
  "${SHARDY_IR_DIR}/verifiers.cc"
  "${SHARDY_IR_DIR}/register.cc"

  DEPENDS
  ShardyDialectIncGen
  ShardyOpsIncGen
  ShardyAttrsIncGen
  ShardyEnumsIncGen
  ShardyOpInterfaceIncGen
  ShardyCanonicalizationIncGen
  ShardyBytecodeIncGen

  LINK_LIBS PUBLIC
  ShardyCommonLogging
  StablehloOps
  MLIRIR
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRShapeDialect
  MLIRSupport
  MLIRTransformUtils
)

target_include_directories(MLIRSdyDialect PUBLIC
  "${SHARDY_SOURCE_DIR}"
  "${CMAKE_BINARY_DIR}"
)

# Suppress warnings in third-party code we cannot fix
foreach(_target obj.MLIRSdyDialect ShardyCommonLogging)
  if(TARGET ${_target})
    target_compile_options(${_target} PRIVATE
      -Wno-deprecated-declarations
      -Wno-covered-switch-default)
  endif()
endforeach()
