add_compile_options(-Wno-deprecated-declarations -Wno-switch -Wno-covered-switch-default)
# LLK raw-string headers embedded into generated C++ by TTKernelToCpp.
set(LLK_SOURCE_DIR "${CMAKE_SOURCE_DIR}/include/ttlang/Target/TTKernel/LLKs")
set(LLK_GEN_DIR "${CMAKE_BINARY_DIR}/include/ttlang/Target/TTKernel/LLKs")
file(MAKE_DIRECTORY "${LLK_GEN_DIR}")

set(LLK_HEADERS
  experimental_tilize_llks
  experimental_untilize_llks
  experimental_pack_untilize_llks
  experimental_invoke_sfpi_llks
  experimental_matmul_llks
  experimental_padding_llks
  experimental_constant_table
  experimental_coord_translation
  experimental_fabric_topology_info
  experimental_fabric_1d_routing
  experimental_fabric_2d_routing
  experimental_fabric_api
  experimental_reg_api
  experimental_semaphore
)

set(GENERATED_LLK_HEADERS)
foreach(header_name ${LLK_HEADERS})
  set(input_file "${LLK_SOURCE_DIR}/${header_name}.h")
  set(output_file "${LLK_GEN_DIR}/${header_name}_generated.h")
  add_custom_command(
    OUTPUT ${output_file}
    COMMAND ${CMAKE_COMMAND}
      -DINPUT_FILE=${input_file}
      -DOUTPUT_FILE=${output_file}
      -DVARIABLE_NAME=${header_name}_generated
      -P "${CMAKE_SOURCE_DIR}/cmake/modules/GenerateRawStringHeader.cmake"
    DEPENDS ${input_file}
    COMMENT "Generating LLK header ${header_name}_generated.h"
    VERBATIM
  )
  list(APPEND GENERATED_LLK_HEADERS ${output_file})
endforeach()

set(DFB_DESCRIPTOR_PRELUDE_SOURCE
  "${CMAKE_SOURCE_DIR}/include/ttlang/Target/TTKernel/DFBDescriptorPrelude.inc")
set(DFB_DESCRIPTOR_PRELUDE_GENERATED
  "${CMAKE_BINARY_DIR}/include/ttlang/Target/TTKernel/DFBDescriptorPrelude_generated.h")
add_custom_command(
  OUTPUT ${DFB_DESCRIPTOR_PRELUDE_GENERATED}
  COMMAND ${CMAKE_COMMAND}
    -DINPUT_FILE=${DFB_DESCRIPTOR_PRELUDE_SOURCE}
    -DOUTPUT_FILE=${DFB_DESCRIPTOR_PRELUDE_GENERATED}
    -DVARIABLE_NAME=dfb_descriptor_prelude_generated
    -P "${CMAKE_SOURCE_DIR}/cmake/modules/GenerateRawStringHeader.cmake"
  DEPENDS ${DFB_DESCRIPTOR_PRELUDE_SOURCE}
  COMMENT "Generating DFB descriptor prelude header"
  VERBATIM
)
list(APPEND GENERATED_LLK_HEADERS ${DFB_DESCRIPTOR_PRELUDE_GENERATED})

add_custom_target(TTKernelGeneratedLLKHeaders DEPENDS ${GENERATED_LLK_HEADERS})

add_mlir_translation_library(TTKernelTargetCpp
  TTKernelToCpp.cpp
  TTKernelToCppRegistration.cpp

  DEPENDS
  TTKernelGeneratedLLKHeaders

  LINK_LIBS PUBLIC
  MLIRTTKernelDialect
  MLIRTTCoreDialect
  MLIRTargetCpp
)
