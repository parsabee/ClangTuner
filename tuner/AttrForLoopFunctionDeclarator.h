//
// Created by parsa on 5/16/21.
//

#ifndef TUNER_FUNCTIONCREATOR_H
#define TUNER_FUNCTIONCREATOR_H

#include "MLIRTypeGenerator.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

namespace clang::tuner {
namespace utils {
using Decls = std::map<llvm::StringRef, VarDecl *>;

/// Finds all declaration references within a for loop to be passed as the loops
/// arguments
/// TODO: exclude the decl refs that their declaration is within the loop
void findLoopInputs(ForStmt *forStmt, ASTContext &context, Decls &inputArgs);
}
} // namespace clang::tuner
#endif // TUNER_FUNCTIONCREATOR_H
