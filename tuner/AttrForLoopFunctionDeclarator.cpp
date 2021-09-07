//
// Created by parsa on 5/16/21.
//

#include "AttrForLoopFunctionDeclarator.h"
#include "clang/AST/RecursiveASTVisitor.h"

namespace clang::tuner {
namespace utils {
using namespace ast_matchers;
#define MATCH_DECL_REF "MatchDeclRefs"
#define MATCH_DECL "MatchDecls"

//class MatchDecl : public MatchFinder::MatchCallback {
//  Decls &decls;
//
//public:
//  explicit MatchDecl(Decls &decls) : decls(decls) {}
//  void run(const MatchFinder::MatchResult &result) override {
//    if (const auto res = result.Nodes.getNodeAs<VarDecl>(MATCH_DECL)) {
//      auto name = res->getName();
//      if (decls.find(name) == decls.end()) {
//        decls[name] = const_cast<VarDecl *>(res);
//      }
//    }
//  }
//};
//
//class MatchDeclRefExpr : public MatchFinder::MatchCallback {
//  Decls &decls;
//  Decls localDecls;
//  ASTContext &context;
//  ForStmt *forStmt;
//
//public:
//  explicit MatchDeclRefExpr(Decls &inputArgs, ASTContext &context, ForStmt *forStmt)
//      : decls(inputArgs), context(context), forStmt(forStmt) {
//    findLocalDecls();
//  }
//
//  void findLocalDecls() {
//    MatchDecl matchDecl(localDecls);
//    MatchFinder matcher;
//    auto tmpMatcher = varDecl().bind(MATCH_DECL);
//    matcher.addMatcher(tmpMatcher, &matchDecl);
//    matcher.matchAST(context);
//    for(auto &it: localDecls)
//      llvm::errs() << it.first << "\n";
//  }
//
//  void run(const MatchFinder::MatchResult &result) override {
//    if (const auto res = result.Nodes.getNodeAs<DeclRefExpr>(MATCH_DECL_REF)) {
//      auto decl = res->getDecl();
//      auto name = res->getDecl()->getName();
//      if (decls.find(name) == decls.end() &&
//          localDecls.find(name) == localDecls.end()) {
//
//        if (auto varDecl = dyn_cast<VarDecl>(decl))
//          decls[name] = const_cast<VarDecl *>(varDecl);
//      }
//    }
//  }
//};

class FindLocalDecls: public RecursiveASTVisitor<FindLocalDecls> {
  Decls &decls;
public:
  explicit FindLocalDecls(Decls &decls) : decls(decls) {}

  bool VisitVarDecl(VarDecl *varDecl) {
    auto name = varDecl->getName();
    if (decls.find(name) == decls.end()) {
      decls[name] = const_cast<VarDecl *>(varDecl);
    }
    return true;
  }
};

class FindLocalDeclRefs : public RecursiveASTVisitor<FindLocalDeclRefs> {
  Decls &decls;
  Decls localDecls;
  ForStmt *forStmt;

  void findLocalDecls() {
    FindLocalDecls findLocalDecls(localDecls);
    findLocalDecls.TraverseForStmt(forStmt);
  }

public:
  FindLocalDeclRefs(Decls &inputArgs, ASTContext &context, ForStmt *forStmt)
  : decls(inputArgs), forStmt(forStmt) {
    findLocalDecls();
  }

  bool VisitDeclRefExpr(DeclRefExpr *declRef) {
    auto decl = declRef->getDecl();
    auto name = declRef->getDecl()->getName();
    if (decls.find(name) == decls.end() &&
        localDecls.find(name) == localDecls.end()) {

      if (auto varDecl = dyn_cast<VarDecl>(decl))
        decls[name] = const_cast<VarDecl *>(varDecl);
    }
    return true;
  }
};


void findLoopInputs(ForStmt *forStmt, ASTContext &context, Decls &inputArgs) {
  FindLocalDeclRefs findLocalDeclRefs(inputArgs,context, forStmt);
  findLocalDeclRefs.TraverseForStmt(forStmt);
}
} // namespace utils
} // namespace clang::tuner
