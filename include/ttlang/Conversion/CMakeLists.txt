set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TTLangConversion)
add_public_tablegen_target(TTLangConversionPassIncGen)
add_dependencies(mlir-headers TTLangConversionPassIncGen)
