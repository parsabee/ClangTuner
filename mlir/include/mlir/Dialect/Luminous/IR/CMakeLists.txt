add_mlir_dialect(LuminousOps luminous)
add_mlir_doc(LuminousOps LuminousDialect Dialects/ -gen-dialect-doc -dialect=luminous)

#set(LLVM_TARGET_DEFINITIONS LuminousOps.td)
#mlir_tablegen(LuminousOps.h.inc -gen-op-decls)
#mlir_tablegen(LuminousOps.cpp.inc -gen-op-defs)
#mlir_tablegen(LuminousDialect.h.inc -gen-dialect-decls)
#mlir_tablegen(LuminousDialect.cpp.inc -gen-dialect-defs)
#add_public_tablegen_target(LuminousDialect)