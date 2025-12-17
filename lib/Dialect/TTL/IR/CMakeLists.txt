add_mlir_dialect_library(MLIRTTLDialect
    TTLDialect.cpp
    TTLOps.cpp
    TTLOpsEnums.cpp
    TTLOpsTypes.cpp
    TTLOpsVerifyUtils.cpp
    ValueUseUtils.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ttlang

    DEPENDS
    MLIRTTLOpsIncGen

    LINK_LIBS PUBLIC
    MLIRTTCoreDialect
)
