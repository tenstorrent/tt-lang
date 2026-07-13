add_compile_options(-Wno-deprecated-declarations -Wno-switch -Wno-covered-switch-default)
add_mlir_dialect_library(MLIRTTCoreDialect
  TTCoreDialect.cpp
  TTCoreOps.cpp
  TTCoreOpsTypes.cpp
  Utils.cpp

  DEPENDS
  MLIRTTCoreOpsIncGen
  TTCoreAttrInterfacesIncGen
  MLIRTTCoreOpsEnumsIncGen
  MLIRTTCoreOpsAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRQuantDialect
)
