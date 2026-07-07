add_compile_options(-Wno-deprecated-declarations -Wno-switch -Wno-covered-switch-default)
add_mlir_conversion_library(TTMLIRTTKernelToEmitC
  TTKernelToEmitC.cpp
  RemoveDeadEmitCExpressions.cpp

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRArithToEmitC
  MLIRMemRefToEmitC
  MLIRSCFToEmitC
  MLIREmitCDialect
  MLIRTargetCpp
  MLIRTransformUtils
  MLIRTTKernelDialect
)
