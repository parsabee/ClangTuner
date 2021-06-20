//
// Created by Parsa Bagheri on 6/16/21.
//

#ifndef TUNER__TYPECORRECTOR_H
#define TUNER__TYPECORRECTOR_H

#include "clang/AST/Type.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/AST/Expr.h"

namespace clang {
namespace tuner {
class TypeCorrector {
  Rewriter &rewriter;

public:
  explicit TypeCorrector(Rewriter &rewriter) : rewriter(rewriter) {}

  void insertCorrection(QualType type,
                        llvm::SmallVector<std::string> &insertedDecls,
                        int offset = 0);

  llvm::SmallVector<std::string> getExpandedTypes(QualType type);

  llvm::SmallVector<std::string> getExpandedArgs(const DeclRefExpr *decl);

};
} // namespace tuner
} // namespace clang

#endif // TUNER__TYPECORRECTOR_H
