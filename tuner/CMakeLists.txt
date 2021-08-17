cmake_minimum_required(VERSION 3.16)
project(tuner)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-std=c++20 -Wall -O0 -fno-rtti ${LLVM_COMPILE_FLAGS}")

set(LLVM_LINK_COMPONENTS support)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CLANG_PATH "/usr/local/bin/clang++")

add_definitions(-DDEBUG -DCLANG_PATH=${CLANG_PATH}) #used in program

find_package(MLIR REQUIRED CONFIG NO_DEFAULT_PATH)
message(STATUS "Using MLIRConfig.cmake in:  ${MLIR_DIR}")

find_package(LLVM REQUIRED CONFIGS NO_DEFAULT_PATH)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(Clang REQUIRED NO_DEFAULT_PATH)
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
include_directories(${MLIR_INCLUDE_DIRS})
message(STATUS "including ${MLIR_INCLUDE_DIRS}")
include_directories(${CLANG_INCLUDE_DIRS})
message(STATUS "including ${CLANG_INCLUDE_DIRS}")
include_directories(${LLVM_INCLUDE_DIRS})
message(STATUS "including ${LLVM_INCLUDE_DIRS}")

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRLoopAnalysis
        MLIRAnalysis
        MLIRDialect
        MLIREDSC
        MLIROptLib
        MLIRParser
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
        MLIRSupport
        MLIRIR
        )

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
        AttrForLoopFinder.h
        AttrForLoopFinder.cpp
        AttrForLoopFunctionDeclarator.cpp
        MLIRCodeGenerator.cpp
        MLIRTypeGenerator.cpp
        TypeCorrector.cpp
        AttrForLoopRefactorer.cpp
        SemaCheck.cpp
        AttrForLoopArgumentFinder.cpp
        OpenMPConfigurer.cpp
        OptimizationDriver.cpp
        Driver.cpp
        mlir/Dialect.cpp
        mlir/MLIRGen.cpp
        ParallelizingPass.cpp

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
        ${LIBS}
        clangAST
        clangBasic
        clangDriver
        clangFrontend
        clangRewriteFrontend
        clangSerialization
        clangTooling
        clangToolingRefactoring
        clangCodeGen

        MLIRSupport
        MLIRIR
        MLIRAnalysis
        MLIRLLVMIR
        MLIRNVVMIR
        MLIRGPU
        MLIRTransforms
        MLIRSCFToStandard
        MLIRSCFToOpenMP
        MLIROpenMPToLLVM
        MLIRStandardToLLVM
        MLIRAffineTransforms
        MLIRAffineToStandard
        MLIRTargetLLVMIRImport
        MLIRToLLVMIRTranslationRegistration
        MLIROptLib
        MLIRParser
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
        MLIRSupport
        MLIRLLVMToLLVMIRTranslation


        LLVMAMDGPUCodeGen
        LLVMAMDGPUDesc
        LLVMAMDGPUInfo
        LLVMAMDGPUAsmParser)
