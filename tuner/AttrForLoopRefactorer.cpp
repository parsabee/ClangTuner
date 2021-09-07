//
// Created by Parsa Bagheri on 6/16/21.
//

#include "AttrForLoopRefactorer.h"
#include "AttrForLoopArgumentFinder.h"
#include "SemaCheck.h"

#include <algorithm>
#include <sstream>

namespace clang {
namespace tuner {

static void printListToStream(SmallVector<std::string> &elms,
                              std::stringstream &ss) {
  bool first = true;
  for (const auto &elm : elms) {
    if (!first)
      ss << ", ";
    ss << elm;
    first = false;
  }
}

static std::string createFunctionDecl(Declarations &loopInputs,
                                      const std::string &functionName,
                                      TypeCorrector &typeCorrector) {
  std::stringstream ss;
  bool first = true;

  ss << "extern \"C\" void " << functionName << "(";
  for (const auto &it : loopInputs) {
    assert(it.second && "Can't be null");
    if (!first)
      ss << ", ";
    auto declType = it.second->getDecl()->getType();
    auto args = typeCorrector.getExpandedTypes(declType);

    bool first2 = true;
    for (auto arg : args) {
      if (!first2)
        ss << ", ";
      ss << arg;
      first2 = false;
    }
    first = false;
  }
  ss << ");\n";
  return ss.str();
}

static std::string createFunctionCall(Declarations &loopInputs,
                                      const std::string &functionName,
                                      TypeCorrector &typeCorrector) {
  bool first = true;
  std::stringstream ss;
  ss << functionName << "(";
  for (const auto &it : loopInputs) {
    assert(it.second && "Can't be null");

    if (!first)
      ss << ", ";

    auto args = typeCorrector.getExpandedArgs(it.second);

    bool first2 = true;
    for (auto arg : args) {
      if (!first2)
        ss << ", ";
      ss << arg;
      first2 = false;
    }

    first = false;
  }
  ss << ");\n";
  return ss.str();
}

const std::string createFunctionName(ForStmt *forStmt,
                                     SourceManager &sourceManager) {
  auto suffix = forStmt->getBeginLoc().printToString(sourceManager);
  std::replace(suffix.begin(), suffix.end(), ':', '_');
  std::replace(suffix.begin(), suffix.end(), '-', '_');
  std::replace(suffix.begin(), suffix.end(), '/', '_');
  std::replace(suffix.begin(), suffix.end(), '.', '_');
  return "__forloop_" + suffix;
}

void AttrForLoopRefactorer::performExtraction(AttributedStmt *stmt) {

  ForStmt *forStmt = nullptr;

  auto attrs = stmt->getAttrs();

  for (auto attr : attrs) {
    if (isAnMLIRAttr(attr)) {
      forStmt = dyn_cast<ForStmt>(stmt->getSubStmt());
    }
  }

  assert(forStmt && "attributed stmt doesn't have a tune attribute");

  rewriter.RemoveText(stmt->getSourceRange());

  Declarations loopInputs;
  findAttrForLoopArguments(forStmt, astContext, loopInputs);

  auto fileID = sourceManager.getFileID(stmt->getBeginLoc());

  auto functionName = createFunctionName(forStmt, sourceManager);

  rewriter.InsertText(
      stmt->getBeginLoc(),
      createFunctionCall(loopInputs, functionName, typeCorrector));

  rewriter.InsertText(sourceManager.getLocForStartOfFile(fileID),
                      "#include <memory>\n"); //std::align is here

  // TODO, create the function decl after headers
  rewriter.InsertText(
      sourceManager.getLocForStartOfFile(fileID),
      createFunctionDecl(loopInputs, functionName, typeCorrector));

}

} // namespace tuner
} // namespace clang
