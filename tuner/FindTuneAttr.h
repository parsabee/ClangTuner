//
// Created by parsa on 4/24/21.
//

#ifndef CLANG_FINDTUNEATTR_H
#define CLANG_FINDTUNEATTR_H

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Rewrite/Core/Rewriter.h"

namespace clang {
namespace tune {
using namespace ast_matchers;

class MatchTuneAttr : public MatchFinder::MatchCallback {
public:
  void run(const MatchFinder::MatchResult &result) override;
};

class MatchTuneAttrConsumer : public ASTConsumer {
  MatchFinder matcher;
  MatchTuneAttr matchTune;
  ASTContext *context;

public:
  explicit MatchTuneAttrConsumer(ASTContext *);

  void HandleTranslationUnit(ASTContext &Context) override;
};

class MatchTuneAttrAction : public ASTFrontendAction {
public:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &,
                                                 StringRef) override;
};

} // namespace tune
} // namespace clang

#endif // CLANG_FINDTUNEATTR_H
