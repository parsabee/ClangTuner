//
// Created by Parsa Bagheri on 6/17/21.
//

#include "LoopInputFinder.h"

namespace clang {
namespace tuner {

void findLoopInputs(ForStmt *forStmt, ASTContext &context,
                           Declarations &inputArgs) {
  Declarations declRefs, varDecls;
  DeclCollector declCollector(varDecls, declRefs);
  declCollector.TraverseForStmt(forStmt);

  for (const auto &it : declRefs) {
    if (varDecls.find(it.first) == varDecls.end()) {
      inputArgs[it.first] = it.second;
    }
  }
}

}
}