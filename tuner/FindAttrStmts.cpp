//
// Created by parsa on 4/25/21.
//
#include "FindAttrStmts.h"

#include "ClangTune/Dialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

namespace clang::tuner {

static bool isATuneAttr(AttributedStmt *AttributedStmt) {
  const auto attrs = AttributedStmt->getAttrs();
  for (auto attr : attrs) {
    if (std::strcmp(attr->getSpelling(), "block_dim") == 0) {
      return true;
    }
  }
  return false;
}

bool FindAttrStmtsVisitor::VisitAttributedStmt(AttributedStmt *attributedStmt) {
  std::cout << "finally here god!\n";
  auto attrs = attributedStmt->getAttrs();
  for (auto attr : attrs) {
    if (std::strcmp(attr->getSpelling(), "block_dim") == 0) {
      if (auto blockDim = dyn_cast<TuneBlockDimAttr>(attr)) {
        for (auto i : blockDim->blockDim()) {
          std::cout << i << "\n";
        }
      }
    }
  }
  return true;
}

bool FindAttrStmtsVisitor::TraverseAttributedStmt(
    AttributedStmt *attributedStmt) {
  bool isCurAttrTune = isATuneAttr(attributedStmt);
  if (isCurAttrTune) {
    isInTuneAttr = true;
    theModule = mlir::ModuleOp::create(opBuilder.getUnknownLoc());
    theModule->getContext()
        ->getOrLoadDialect<mlir::clang_tune::ClangTuneDialect>();
    theModule->getContext()->getOrLoadDialect<mlir::AffineDialect>();
  }
  if (!clang::RecursiveASTVisitor<FindAttrStmtsVisitor>::TraverseAttributedStmt(
          attributedStmt))
    return false;
  if (isCurAttrTune)
    isInTuneAttr = false;
  return true;
}

using InitDecl =
    std::pair<std::pair<const QualType, llvm::StringRef>, const Expr *>;
using InitDecls = llvm::SmallVector<InitDecl>;

static void handleLoopDecls(const VarDecl *loopDecl, InitDecls &initDecls) {
  assert(loopDecl && "loop decl is not a var decl");
  const QualType type = loopDecl->getType();
  assert(type->isBuiltinType() && "var decl doesn't have a builtin type");
  const llvm::StringRef varName = loopDecl->getName();
  const Expr *init = loopDecl->getInit();
  assert(init && "var decl is not initialized");
  initDecls.push_back({{type, varName}, init});
}

mlir::Value mlirGen(BinaryOperator *binOp) {

  switch (binOp->getOpcode()) {
  case BinaryOperator::Opcode::BO_LT:
  case BinaryOperator::Opcode::BO_Assign:
  default:
    break;
  }
}

class ForLoopBodyBuilder : public StmtVisitor<ForLoopBodyBuilder, mlir::Value> {
  ForStmt *forStmt;
  ASTContext &astContext;
  mlir::MLIRContext *mlirContext;
  mlir::ModuleOp &module;

public:
  ForLoopBodyBuilder(ForStmt *forStmt, ASTContext &context,
                     mlir::ModuleOp &moduleOp)
      : forStmt(forStmt), astContext(context), module(moduleOp),
        mlirContext(moduleOp->getContext()) {}

  void operator()(mlir::OpBuilder &, mlir::Location, mlir::Value,
                  mlir::ValueRange) {
    const auto body = forStmt->getBody();
    if (const auto compStmt = dyn_cast<CompoundStmt>(body)) {
      for (const auto it : compStmt->body()) {
        auto res = Visit(it);
      }
    } else
      auto res = Visit(body);
  }

  mlir::Value VisitImplicitCastExpr(ImplicitCastExpr *implicitCastExpr) {
    return Visit(implicitCastExpr->getSubExpr());
  }

  mlir::Value VisitArraySubscriptExpr(ArraySubscriptExpr *arraySubscriptExpr) {
    auto lhsRes = Visit(arraySubscriptExpr->getLHS());
    auto rhsRes = Visit(arraySubscriptExpr->getRHS());
    return nullptr;
  }
  mlir::Value VisitBinaryOperator(BinaryOperator *binOp) {
    auto lhsRes = Visit(binOp->getLHS());
    auto rhsRes = Visit(binOp->getRHS());
    return nullptr;
  }
  mlir::Value VisitDeclRefExpr(DeclRefExpr *declRef) {
    declRef->dump();
    return nullptr;
  }

private:
  //  mlir::Type mlirGen(QualType *qualType) {
  //    auto type = qualType->getTypePtr();
  //    mlir::Type t;
  //    if (type->isFloatingType()) {
  //      if (type->isArrayType()) {
  //        //        if (type->)
  //        type->dump();
  //        //        t = mlir::MemRefType::get({})
  //      } else
  //        t = mlir::Float32Type::get(mlirContext);
  //    }
  //    return t;
  //  }
  //
  //  mlir::Value mlirGen(DeclRefExpr *declRef) {
  //    auto decl = declRef->getDecl();
  //    auto type = decl->getType();
  //    auto t = mlirGen(&type);
  //    if (type->isBuiltinType()) {
  //    }
  //
  //    if (type->isArrayType()) {
  //    }
  //  }
  //
  //  mlir::Value mlirGen(Expr *expr) {
  //    if (auto declRef = dyn_cast<DeclRefExpr>(expr)) {
  //      return mlirGen(declRef);
  //    }
  //  }
};

static bool isAssignment(Stmt *stmt) {
  if (const auto binOp = dyn_cast<BinaryOperator>(stmt)) {
    if (binOp->getOpcode() == BinaryOperator::Opcode::BO_Assign ||
        binOp->getOpcode() == BinaryOperator::Opcode::BO_AddAssign ||
        binOp->getOpcode() == BinaryOperator::Opcode::BO_SubAssign ||
        binOp->getOpcode() == BinaryOperator::Opcode::BO_DivAssign ||
        binOp->getOpcode() == BinaryOperator::Opcode::BO_MulAssign) {
      return true;
    }
  }
  return false;
}

using namespace ast_matchers;

#define MATCH_DECL_REF "MatchDeclRefs"
using LoopInputArgs = std::map<llvm::StringRef, const Expr *>;

class MatchDeclRefExpr : public MatchFinder::MatchCallback {
  LoopInputArgs &inputArgs;
  const bool &isInAttr;

public:
  explicit MatchDeclRefExpr(LoopInputArgs &inputArgs, const bool &isInAttr)
      : inputArgs(inputArgs), isInAttr(isInAttr) {}

  void run(const MatchFinder::MatchResult &result) override {
    if (const auto res = result.Nodes.getNodeAs<DeclRefExpr>(MATCH_DECL_REF)) {
      if (isInAttr) {
        auto name = res->getDecl()->getName();
        if (inputArgs.find(name) == inputArgs.end()) {
          inputArgs[name] = res;
        }
      }
    }
  }
};

static void findLoopInputs(ForStmt *forStmt, ASTContext &context,
                           LoopInputArgs &inputArgs, const bool &isInAttr) {
  MatchDeclRefExpr matchDeclRef(inputArgs, isInAttr);
  MatchFinder matcher;
  auto tmpMatcher = declRefExpr().bind(MATCH_DECL_REF);
  matcher.addMatcher(tmpMatcher, &matchDeclRef);
  matcher.matchAST(context);
}

bool FindAttrStmtsVisitor::VisitForStmt(ForStmt *forStmt) {
  if (!isInTuneAttr)
    return true;

  const auto initStmt = forStmt->getInit();
  const auto declStmt = dyn_cast<DeclStmt>(initStmt);
  assert(declStmt && "init stmt is not a decl stmt");

  InitDecls initDecls;
  if (declStmt->isSingleDecl()) {
    const auto varDecl = dyn_cast<VarDecl>(declStmt->getSingleDecl());
    handleLoopDecls(varDecl, initDecls);
  }
  //  else {
  //    const auto groupDecl = declStmt->getDeclGroup();
  //    for (const auto decl : groupDecl) {
  //      handleLoopDecls(dyn_cast<VarDecl>(decl), initDecls);
  //    }
  //  }
  const Expr *lowerBound = nullptr;
  for (const auto &it : initDecls) {
    lowerBound = it.second;
  }

  const auto cond = forStmt->getCond();
  assert(cond && "for loop doesn't have a conditional stmt");

  const Expr *upperBound = nullptr;
  const Expr *condLHS = nullptr;
  if (auto *const binOp = dyn_cast<BinaryOperator>(cond)) {
    if (binOp->getOpcode() == BinaryOperator::Opcode::BO_LT ||
        binOp->getOpcode() == BinaryOperator::Opcode::BO_LE) {
      upperBound = binOp->getRHS();
      condLHS = binOp->getLHS();
    }
  }

  assert(condLHS && upperBound && "invalid loop condition");
  bool isCondLHSInInitDecls = false;
  auto checkInitDeclsForExpr =
      [&initDecls, &isCondLHSInInitDecls](llvm::StringRef condVar) {
        for (const auto &it : initDecls) {
          if (it.first.second == condVar) {
            isCondLHSInInitDecls = true;
          }
        }
      };

  if (const auto castExpr = dyn_cast<ImplicitCastExpr>(condLHS)) {
    if (const auto expr = dyn_cast<DeclRefExpr>(castExpr->getSubExpr())) {
      checkInitDeclsForExpr(expr->getDecl()->getName());
    }
  } else if (const auto expr = dyn_cast<DeclRefExpr>(condLHS)) {
    checkInitDeclsForExpr(expr->getDecl()->getName());
  }
  assert(isCondLHSInInitDecls && "condition is not declared in the init stmt");

  bool isIncrementByOne = false;
  const auto incr = forStmt->getInc();
  if (const auto unaryOp = dyn_cast<UnaryOperator>(incr)) {
    if (unaryOp->isIncrementOp())
      isIncrementByOne = true;
  } else if (const auto binaryOp = dyn_cast<BinaryOperator>(incr)) {
    if (binaryOp->getOpcode() == BinaryOperator::Opcode::BO_AddAssign) {
      const auto rhs = binaryOp->getRHS();
      Expr::EvalResult evalResult;
      if (rhs->EvaluateAsConstantExpr(evalResult, astContext)) {
        if (evalResult.Val.isInt()) {
          auto val = evalResult.Val.getInt();
          if (val == 1)
            isIncrementByOne = true;
        }
      }
    }
  }
  assert(isIncrementByOne && "The loop variable needs to increment by one");

  ForLoopBodyBuilder loopBodyBuilder(forStmt, astContext, theModule);
  Expr::EvalResult lowerBoundResult, upperBoundResult;
  if (lowerBound->EvaluateAsConstantExpr(lowerBoundResult, astContext)) {
    uint64_t lb, ub;
    if (lowerBoundResult.Val.isInt())
      lb = lowerBoundResult.Val.getInt().getExtValue();
    if (upperBound->EvaluateAsConstantExpr(upperBoundResult, astContext)) {
      if (upperBoundResult.Val.isInt()) {
        ub = upperBoundResult.Val.getInt().getExtValue();
        auto forLoopOp = opBuilder.create<mlir::clang_tune::ForLoopOp>(
            opBuilder.getUnknownLoc(), lb, ub, 1, llvm::None, loopBodyBuilder);

        LoopInputArgs loopInputArgs;
        findLoopInputs(forStmt, astContext, loopInputArgs, isInTuneAttr);
        theModule.push_back(forLoopOp);
        theModule->dump();
        return true;
      }
    }
  }

  //  lowerBound->dump();
  //  mlir::Value lb = mlirGen(lowerBound);
  //  mlir::Value ub = mlirGen(upperBound);
  //  auto forLoopOp =
  //      opBuilder.create<mlir::clang_tune::ForLoopOp>(opBuilder.getUnknownLoc(),
  //                                                    {lowerBound},
  //                                                    {upperBound});
  return true;
}
} // namespace clang::tuner