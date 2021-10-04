add_clang_tuner_library(MLIRCodeGen
  MLIRCodeGenerator.cpp
  MLIRTypeGenerator.cpp

  LINK_LIBS
  MLIRAffineToStandard
  MLIRAffineTransforms
  MLIRLinalgTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRMathTransforms
  MLIROptLib
  MLIRSCFToStandard
  MLIRShapeToStandard
)