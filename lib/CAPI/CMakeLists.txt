add_mlir_library(TTLangCAPI
  Dialects.cpp
  TTCoreAttrs.cpp
  TTKernelTypes.cpp
  TTLAttrs.cpp

  EXCLUDE_FROM_LIBMLIR
  ENABLE_AGGREGATION

  DEPENDS
  MLIRTTLOpsAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineToStandard
  MLIRArithDialect
  MLIRCAPIIR
  MLIRCAPITransforms
  MLIRControlFlowDialect
  MLIREmitCDialect
  MLIRFuncDialect
  MLIRIR
  MLIRIndexDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTTCoreDialect
  MLIRTTKernelDialect
  MLIRTTLDialect
  MLIRTransforms
  MLIRUBDialect
  TTKernelTargetCpp
  TTLangTTKernelToEmitC
  TTLangTTLPipelines
  TTLangTTLTransforms
)

target_compile_options(TTLangCAPI PRIVATE
  -Wno-covered-switch-default
  -Wno-deprecated-declarations
  -Wno-switch
)
