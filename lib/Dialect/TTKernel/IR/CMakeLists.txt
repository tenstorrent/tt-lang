add_compile_options(-Wno-deprecated-declarations -Wno-switch -Wno-covered-switch-default)
add_mlir_dialect_library(MLIRTTKernelDialect
  TTKernelDialect.cpp
  TTKernelOps.cpp
  TTKernelOpsTypes.cpp

  DEPENDS
  MLIRTTKernelOpsIncGen
  MLIRTTKernelOpsAttributesIncGen
  MLIRTTKernelOpsEnumsIncGen
  MLIRTTCoreOpsIncGen

  LINK_LIBS PUBLIC
  MLIRSCFDialect
  MLIRTTCoreDialect
)
