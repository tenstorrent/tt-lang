set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name D2M)
add_public_tablegen_target(TTLangD2MTransformsIncGen)
add_dependencies(mlir-headers TTLangD2MTransformsIncGen)
