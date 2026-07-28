# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Support wiring for the tt-lang-owned ttcore/ttkernel dialects, the
# TTKernelToEmitC conversion, and the TTKernelToCpp translation. Their sources
# live under include/ttlang, lib/Dialect, lib/Conversion, and lib/Target and are
# compiled by the normal add_subdirectory(include)/add_subdirectory(lib) tree.

# add_mlir_dialect/add_mlir_doc reference these aggregate targets; create them
# when building against a pre-built MLIR install (find_package) that lacks them.
if(NOT TARGET mlir-headers)
  add_custom_target(mlir-headers)
endif()
if(NOT TARGET mlir-doc)
  add_custom_target(mlir-doc)
endif()

# tt-lang does not use flatbuffers; keep the system-descriptor flatbuffer loader
# compiled out (preserves the previous behavior).
add_compile_definitions(TTLANG_NO_FLATBUFFERS)

# Dialect/conversion/translation libraries linked by ttlang-opt and
# ttlang-translate. Defined later under add_subdirectory(lib); listed here by
# name so the tool CMakeLists (added at this scope) can reference them.
set(TTLANG_DIALECT_LIBS
  MLIRTTCoreDialect
  MLIRTTKernelDialect
  TTKernelTargetCpp
  TTLangTTKernelToEmitC
)
