add_subdirectory(IR)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TTL)
add_public_tablegen_target(TTLangTTLTransformsIncGen)
add_dependencies(mlir-headers TTLangTTLTransformsIncGen)
