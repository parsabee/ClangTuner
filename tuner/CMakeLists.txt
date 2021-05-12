cmake_minimum_required(VERSION 3.16)
project(tuner)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-std=c++20 -Wall -O0 -fno-rtti ${LLVM_COMPILE_FLAGS}")

set(LLVM_LINK_COMPONENTS support)
set(CMAKE_BUILD_TYPE Debug)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in:  ${MLIR_DIR}")

find_package(LLVM REQUIRED CONFIGS)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(Clang REQUIRED)
message(STATUS "Found Clang ${CLANG_PACKAGE_VERSION}")
message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")
message(STATUS "Found CLANG DEFS${CLANG_DEFINITIONS}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddLLVM)
include(AddMLIR)
include(TableGen)

add_custom_target(ClangTune)
set_target_properties(ClangTune PROPERTIES FOLDER Examples)##FIXME

macro(add_clang_tune_target name)
    add_dependencies(ClangTune ${name})
    add_llvm_example(${name} ${ARGN})
endmacro(add_clang_tune_target name)

llvm_map_components_to_libnames(LLVM_LIBS ${LLVM_TARGETS_TO_BUILD})

set(MLIR_DIR ../mlir)
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${MLIR_DIR}/include/)

add_subdirectory(include)

set(MLIR_LIBS
        MLIRAffineToStandard
        MLIRAffineTransforms
        MLIRLinalgTransforms
        MLIRLLVMToLLVMIRTranslation
        MLIRMathTransforms
        MLIROptLib
        MLIRSCFToStandard
        MLIRShapeToStandard
        )

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
add_definitions(${CLANG_DEFINITIONS})
add_clang_tune_target(clang-tune
        main.cpp
        FindTuneAttr.h
        FindTuneAttr.cpp
        FindAttrStmts.h
        FindAttrStmts.cpp
        mlir/Dialect.cpp
        mlir/MLIRGen.cpp

        DEPENDS
        ClangTuneIncGen

        )


include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

target_link_libraries(
        clang-tune
        PRIVATE
        ${LLVM_AVAILABLE_LIBS}
        ${LLVM_LIBS}
        ${MLIR_LIBS}
        clangAST
        clangBasic
        clangDriver
        clangFrontend
        clangRewriteFrontend
        clangSerialization
        clangTooling

        MLIRSupport
        MLIRIR
        MLIRAnalysis
        MLIRLLVMIR
        MLIRNVVMIR
        MLIRGPU
        MLIRTransforms
        MLIRSCFToStandard
        MLIRStandardToLLVM
        MLIRAffineTransforms
        MLIRAffineToStandard
        MLIRTargetLLVMIRImport

        LLVMAMDGPUCodeGen
        LLVMAMDGPUDesc
        LLVMAMDGPUInfo
        LLVMAMDGPUAsmParser)