set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TTLang)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix TTLang)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix TTLang)
add_public_tablegen_target(TTLangTransformsIncGen)
add_dependencies(mlir-headers TTLangTransformsIncGen)

add_mlir_doc(Passes TTLangPasses ./ -gen-pass-doc)
