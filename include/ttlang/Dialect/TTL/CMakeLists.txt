add_subdirectory(IR)
add_subdirectory(Pipelines)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TTL)
add_public_tablegen_target(TTLangTTLTransformsIncGen)
add_dependencies(mlir-headers TTLangTTLTransformsIncGen)
