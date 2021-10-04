//
// Created by Parsa Bagheri on 6/16/21.
//
/// SourceRewriter is a module for refactoring out the relevant attributed stmts
/// and rewriting the original source code

#ifndef TUNER__ATTRFORLOOPREFACTORER_H
#define TUNER__ATTRFORLOOPREFACTORER_H

#include "clang/AST/Stmt.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "TypeCorrector.h"

namespace clang {
namespace tuner {

using Decls = std::map<llvm::StringRef, VarDecl *>;

/// Finds all declaration references within a for loop to be passed as the loops
/// arguments
/// TODO: exclude the decl refs that their declaration is within the loop
void findLoopInputs(ForStmt *forStmt, ASTContext &context, Decls &inputArgs);

const std::string createFunctionName(ForStmt *forStmt,
                               SourceManager &sourceManager);

class AttrForLoopRefactorer {
  SourceManager &sourceManager;
  ASTContext &astContext;
  Rewriter &rewriter;
  TypeCorrector typeCorrector;

public:
  AttrForLoopRefactorer(SourceManager &sourceManager, ASTContext &astContext,
                 Rewriter &rewriter)
      : sourceManager(sourceManager), astContext(astContext),
        rewriter(rewriter), typeCorrector(rewriter) {}

  void performExtraction(AttributedStmt *stmt);
};
} // namespace tuner
} // namespace clang
#endif // TUNER__ATTRFORLOOPREFACTORER_H
