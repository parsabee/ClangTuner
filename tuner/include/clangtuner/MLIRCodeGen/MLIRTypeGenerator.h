//
// Created by parsa on 5/14/21.
//

#ifndef TUNER_TYPEGEN_H
#define TUNER_TYPEGEN_H

#include "../../clang/lib/CodeGen/CodeGenTypes.h"
#include "clang/AST/StmtVisitor.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

// Parsa: The code below is taken from
// llvm-project/mlir/lib/Target/LLVMIR/ConvertFromLLVMIR.cpp,
// it is an internal library for lifting LLVM IR types to LLVM dialect of MLIR.
// This doesn't appear in a header file that's why I pasted the code here.
// The implementation is in libMLIRLLVMToLLVMIRTranslation.a

/// Utility class to translate LLVM IR types to the MLIR LLVM dialect. Stores
/// the translation state, in particular any identified structure types that are
/// reused across translations.
namespace mlir::LLVM {
namespace detail {
class TypeFromLLVMIRTranslatorImpl;
}
class TypeFromLLVMIRTranslator {
public:
  TypeFromLLVMIRTranslator(mlir::MLIRContext &context);
  ~TypeFromLLVMIRTranslator();

  /// Translates the given LLVM IR type to the MLIR LLVM dialect.
  mlir::Type translateType(llvm::Type *type);

private:
  /// Private implementation.
  std::unique_ptr<mlir::LLVM::detail::TypeFromLLVMIRTranslatorImpl> impl;
};
} // namespace mlir::LLVM

namespace clang::tuner {

/// This class generates an mlir type corresponding to types declared in the
/// program. It uses the CodeGenModule to lower types to LLVM types, then uses
/// the above class to lift them to mlir. Arrays are handled differently, Memref
/// dialect is used to represent array types.
class MLIRTypeGenerator : public StmtVisitor<MLIRTypeGenerator, mlir::Type> {
  mlir::MLIRContext *mlirContext;
  CodeGen::CodeGenTypes &CGTypes;
  mlir::LLVM::TypeFromLLVMIRTranslator llvmTypeTranlator;

public:
  MLIRTypeGenerator(mlir::ModuleOp &moduleOp, CodeGen::CodeGenTypes &CGTypes)
      : mlirContext(moduleOp->getContext()), CGTypes(CGTypes),
        llvmTypeTranlator(*mlirContext) {}

  mlir::Type VisitImplicitCastExpr(ImplicitCastExpr *);

  mlir::Type VisitDeclRefExpr(DeclRefExpr *);

  mlir::Type VisitArraySubscriptExpr(ArraySubscriptExpr *);

  mlir::Type VisitIntegerLiteral(IntegerLiteral *);

  mlir::MemRefType getMemRefType(Decl *);

  mlir::MemRefType getMemRefType(VarDecl *);
};
} // namespace clang::tuner
#endif // TUNER_TYPEGEN_H
