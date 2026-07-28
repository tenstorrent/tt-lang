add_compile_options(-Wno-deprecated-declarations -Wno-switch -Wno-covered-switch-default)
add_mlir_conversion_library(TTLangTTKernelToEmitC
  TTKernelToEmitC.cpp
  RemoveDeadEmitCExpressions.cpp

  DEPENDS
  TTLangConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithToEmitC
  MLIREmitCDialect
  MLIRIR
  MLIRMemRefToEmitC
  MLIRPass
  MLIRSCFToEmitC
  MLIRTargetCpp
  MLIRTTKernelDialect
  MLIRTransformUtils
)
