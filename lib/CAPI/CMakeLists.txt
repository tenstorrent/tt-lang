add_mlir_library(TTLangCAPI
  Dialects.cpp
  TTLAttrs.cpp

  EXCLUDE_FROM_LIBMLIR
  ENABLE_AGGREGATION

  DEPENDS
  MLIRTTLOpsAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRTTLDialect
  TTLangTTLTransforms
  TTLangTTLPipelines

  # Minimal upstream dialects + passes the tt-lang pipeline uses, registered in
  # Dialects.cpp in place of MLIR's RegisterEverything.
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRControlFlowDialect
  MLIRAffineDialect
  MLIREmitCDialect
  MLIRIndexDialect
  MLIRUBDialect
  MLIRMathDialect
  MLIRTransforms
  MLIRAffineToStandard
)
