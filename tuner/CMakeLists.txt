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

# Macro for creating the tuner libraries
macro(add_clang_tuner_library name)
  cmake_parse_arguments(ARG
    "SHARED;STATIC;INSTALL_WITH_TOOLCHAIN"
    ""
    "ADDITIONAL_HEADERS"
    ${ARGN})
  set(srcs)

  ## FIXME Add XCODE and MSVC support here

  if(srcs OR ARG_ADDITIONAL_HEADERS)
    set(srcs
      ADDITIONAL_HEADERS
      ${srcs}
      ${ARG_ADDITIONAL_HEADERS} # It may contain unparsed unknown args.
      )
  endif()

  if(ARG_SHARED AND ARG_STATIC)
    set(LIBTYPE SHARED STATIC)
  elseif(ARG_SHARED)
    set(LIBTYPE SHARED)
  else()
    # llvm_add_library ignores BUILD_SHARED_LIBS if STATIC is explicitly set,
    # so we need to handle it here.
    if(BUILD_SHARED_LIBS)
      set(LIBTYPE SHARED)
    else()
      set(LIBTYPE STATIC)
    endif()
    if(NOT XCODE)
      # The Xcode generator doesn't handle object libraries correctly.
      list(APPEND LIBTYPE OBJECT)
    endif()
    set_property(GLOBAL APPEND PROPERTY CLANG_TUNER_STATIC_LIBS ${name})
  endif()
  llvm_add_library(${name} ${LIBTYPE} ${ARG_UNPARSED_ARGUMENTS} ${srcs})

  set(libs ${name})
  if(ARG_SHARED AND ARG_STATIC)
    list(APPEND libs ${name}_static)
  endif()

  foreach(lib ${libs})
    if(TARGET ${lib})
      target_link_libraries(${lib} INTERFACE ${LLVM_COMMON_LIBS})

      ## Copied + modified from AddClang.cmake
      ##
      if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY OR ARG_INSTALL_WITH_TOOLCHAIN)
        set(export_to_clangtunertargets)
        if(${lib} IN_LIST LLVM_DISTRIBUTION_COMPONENTS OR
            "clang-libraries" IN_LIST LLVM_DISTRIBUTION_COMPONENTS OR
            NOT LLVM_DISTRIBUTION_COMPONENTS)
          set(export_to_clangtunertargets EXPORT ClangTunerTargets)
          set_property(GLOBAL PROPERTY CLANG_HAS_EXPORTS True)
        endif()

        install(TARGETS ${lib}
          COMPONENT ${lib}
          ${export_to_clangtunertargets}
          LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
          ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
          RUNTIME DESTINATION bin)

        if (NOT LLVM_ENABLE_IDE)
          add_llvm_install_targets(install-${lib}
                                   DEPENDS ${lib}
                                   COMPONENT ${lib})
        endif()

        set_property(GLOBAL APPEND PROPERTY CLANG_TUNER_LIBS ${lib})
      endif()
      set_property(GLOBAL APPEND PROPERTY CLANG_TUNER_EXPORTS ${lib})
    else()
      # Add empty "phony" target
      add_custom_target(${lib})
    endif()
  endforeach()

  set_target_properties(${name} PROPERTIES FOLDER "ClangTuner libraries")
#  set_clang_windows_version_resource_properties(${name})
endmacro(add_clang_tuner_library name)

macro(add_clang_tuner_executable name)
  add_llvm_executable( ${name} ${ARGN} )
  set_target_properties(${name} PROPERTIES FOLDER "ClangTuner executables")
  set_clang_tuner_windows_version_resource_properties(${name})
endmacro(add_clang_tuner_executable)

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

#add_subdirectory(include)
add_subdirectory(lib)

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
        ParallelizingPass.cpp
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
