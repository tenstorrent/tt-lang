cmake_minimum_required(VERSION 3.20)
project(ttlang_docs NONE)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pip install sphinx myst-parser sphinx-rtd-theme sphinxcontrib-mermaid
  RESULT_VARIABLE _sphinx_pip_result
)

get_filename_component(_python_bindir ${Python3_EXECUTABLE} DIRECTORY)
if(_python_bindir)
  list(APPEND CMAKE_PROGRAM_PATH ${_python_bindir})
endif()

find_program(SPHINX_EXECUTABLE NAMES sphinx-build)
if(NOT SPHINX_EXECUTABLE)
  message(WARNING "sphinx-build not found after install attempt; disabling documentation targets.")
  return()
endif()

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/_build)

add_custom_target(ttlang-docs
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SPHINX_BUILD}/html
  COMMAND ${SPHINX_EXECUTABLE} -b html ${SPHINX_SOURCE} ${SPHINX_BUILD}/html
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating Sphinx documentation for tt-lang"
)
