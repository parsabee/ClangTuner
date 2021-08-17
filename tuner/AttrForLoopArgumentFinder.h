//
// Created by Parsa Bagheri on 6/17/21.
//

#ifndef TUNER__ATTRFORLOOPARGUMENTFINDER_H
#define TUNER__ATTRFORLOOPARGUMENTFINDER_H

#include "clang/AST/RecursiveASTVisitor.h"

namespace clang {
namespace tuner {

using Declarations = std::map<llvm::StringRef, const DeclRefExpr *>;

class DeclCollector : public clang::RecursiveASTVisitor<DeclCollector> {
  Declarations &varDecls;
  Declarations &declRefs;

public:
  DeclCollector(Declarations &varDecls, Declarations &declRefs)
      : varDecls(varDecls), declRefs(declRefs) {}

  bool VisitVarDecl(VarDecl *varDecl) {
    varDecls[varDecl->getName()] = nullptr;
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr *declRefExpr) {
    declRefs[declRefExpr->getDecl()->getName()] = declRefExpr;
    return true;
  }
};

/// Finds all declaration references within a for loop to be passed as the loops
/// arguments
/// TODO: exclude the decl refs that their declaration is within the loop
void findAttrForLoopArguments(ForStmt *forStmt, ASTContext &context,
                           Declarations &inputArgs);

} // namespace tuner
} // namespace clang

#endif // TUNER__ATTRFORLOOPARGUMENTFINDER_H
