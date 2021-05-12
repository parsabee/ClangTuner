//
// Created by parsa on 4/24/21.
//

#include "FindTuneAttr.h"
#include <iostream>

#define CUDA_TUNE "tune"

namespace clang {
namespace tune {

MatchTuneAttrConsumer::MatchTuneAttrConsumer(ASTContext *context)
    : context(context) {
  auto tmpMatcher = stmt().bind(CUDA_TUNE);
  matcher.addMatcher(tmpMatcher,
                     &matchTune);
}

void MatchTuneAttr::run(const MatchFinder::MatchResult &result) {
  if (const auto *res = result.Nodes.getNodeAs<Stmt>(CUDA_TUNE)) {
    std::cout << "here\n";
  }
}

void MatchTuneAttrConsumer::HandleTranslationUnit(ASTContext &context) {
  matcher.matchAST(context);
}

std::unique_ptr<ASTConsumer>
MatchTuneAttrAction::CreateASTConsumer(CompilerInstance &CI, StringRef file) {
  return std::make_unique<MatchTuneAttrConsumer>(&CI.getASTContext());

}
} // namespace tune
} // namespace clang