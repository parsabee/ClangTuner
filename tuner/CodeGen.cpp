//
// Created by parsa on 5/14/21.
//

#include "CodeGen.h"
#include "ClangTune/Dialect.h"
#include "FunctionCreator.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

namespace clang::tuner {

#define MATCH_DECL_REF "MatchDeclRefs"
using LoopInputArgs = std::map<llvm::StringRef, const Expr *>;

#define UNKNOWN_LOC opBuilder.getUnknownLoc()

using namespace ast_matchers;
class MatchDeclRefExpr : public MatchFinder::MatchCallback {
  LoopInputArgs &inputArgs;

public:
  explicit MatchDeclRefExpr(LoopInputArgs &inputArgs) : inputArgs(inputArgs) {}

  void run(const MatchFinder::MatchResult &result) override {
    if (const auto res = result.Nodes.getNodeAs<DeclRefExpr>(MATCH_DECL_REF)) {
      auto name = res->getDecl()->getName();
      if (inputArgs.find(name) == inputArgs.end()) {
        inputArgs[name] = res;
      }
    }
  }
};

/// Finds all declaration references within a for loop to be passed as the loops
/// arguments
/// TODO: exclude the decl refs that their declaration is within the loop
static void findLoopInputs(ForStmt *forStmt, ASTContext &context,
                           LoopInputArgs &inputArgs) {
  MatchDeclRefExpr matchDeclRef(inputArgs);
  MatchFinder matcher;
  auto tmpMatcher = declRefExpr().bind(MATCH_DECL_REF);
  matcher.addMatcher(tmpMatcher, &matchDeclRef);
  matcher.matchAST(context);
}

/// forloop inits
using ForLoopInitDecl =
    std::pair<std::pair<const QualType, llvm::StringRef>, Expr *>;
using ForLoopInitDecls = llvm::SmallVector<ForLoopInitDecl>;

static void handleLoopDecls(const VarDecl *loopDecl,
                            ForLoopInitDecls &initDecls) {
  assert(loopDecl && "loop decl is not a var decl");
  const QualType type = loopDecl->getType();
  assert(type->isBuiltinType() && "var decl doesn't have a builtin type");
  const llvm::StringRef varName = loopDecl->getName();
  Expr *init = const_cast<Expr *>(loopDecl->getInit());
  assert(init && "var decl is not initialized");
  initDecls.push_back({{type, varName}, init});
}

mlir::Value CodeGen::VisitDeclStmt(DeclStmt *declStmt) {
  for (auto decl : declStmt->decls()) {
    if (auto varDecl = dyn_cast<VarDecl>(decl)) {
      auto type = typeGen.getType(varDecl);
      assert(type && "type must be valid at this point");
      auto alloca = opBuilder.create<mlir::memref::AllocaOp>(UNKNOWN_LOC, type);
      if (varDecl->hasInit()) {
        auto val = Visit(varDecl->getInit());
        /// Fixme this only covers ints
        auto store = opBuilder.create<mlir::memref::StoreOp>(UNKNOWN_LOC, val, alloca);
      }
      scopedHashTable.insert(varDecl->getName(), alloca);
    } else
      assert(false && "only var decl is supported at the momemnt");
  }
  return nullptr; // nothing to return
}

mlir::Value CodeGen::VisitForStmt(ForStmt *forStmt) {
  llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> forScope(
      scopedHashTable);
  const auto forLoopInitStmt = forStmt->getInit();
  const auto forLoopInitDeclStmt = dyn_cast<DeclStmt>(forLoopInitStmt);

  assert(forLoopInitDeclStmt && "init stmt is not a decl stmt");
  Visit(forLoopInitDeclStmt);

  ForLoopInitDecls initDecls;
  assert(forLoopInitDeclStmt->isSingleDecl() && "loop must have a single decl");

  const auto varDecl = dyn_cast<VarDecl>(forLoopInitDeclStmt->getSingleDecl());
  handleLoopDecls(varDecl, initDecls);

  Expr *lowerBound = nullptr;
  for (const auto &it : initDecls) {
    lowerBound = it.second;
  }

  const auto cond = forStmt->getCond();
  assert(cond && "for loop doesn't have a conditional stmt");

  Expr *upperBound = nullptr;
  Expr *condLHS = nullptr;
  if (auto *const binOp = dyn_cast<BinaryOperator>(cond)) {
    if (binOp->getOpcode() == BinaryOperator::Opcode::BO_LT ||
        binOp->getOpcode() == BinaryOperator::Opcode::BO_LE) {
      upperBound = binOp->getRHS();
      condLHS = binOp->getLHS();
    }
  }

  assert(condLHS && upperBound && "invalid loop condition");
  bool isCondLHSInInitDecls = false;
  auto checkInitDeclsForExpr = [&initDecls](llvm::StringRef condVar) -> bool {
    for (const auto &it : initDecls) {
      if (it.first.second == condVar) {
        return true;
      }
    }
    return false;
  };

  if (const auto castExpr = dyn_cast<ImplicitCastExpr>(condLHS)) {
    if (const auto expr = dyn_cast<DeclRefExpr>(castExpr->getSubExpr())) {
      isCondLHSInInitDecls = checkInitDeclsForExpr(expr->getDecl()->getName());
    }
  } else if (const auto expr = dyn_cast<DeclRefExpr>(condLHS)) {
    isCondLHSInInitDecls = checkInitDeclsForExpr(expr->getDecl()->getName());
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

  /// this lambda will be passed to the ForLoopOp to build the body of forloop
  auto forLoopBodyBuilder = [&forStmt, this](mlir::OpBuilder &, mlir::Location,
                                             mlir::Value, mlir::ValueRange) {
    const auto body = forStmt->getBody();
    if (const auto compStmt = dyn_cast<CompoundStmt>(body)) {
      for (const auto it : compStmt->body()) {
        auto res = this->Visit(it);
      }
    } else
      auto res = this->Visit(body);
  };

  Expr::EvalResult lowerBoundResult, upperBoundResult;
  if (lowerBound->EvaluateAsConstantExpr(lowerBoundResult, astContext)) {
    uint64_t lb, ub;
    if (lowerBoundResult.Val.isInt())
      lb = lowerBoundResult.Val.getInt().getExtValue();
    if (upperBound->EvaluateAsConstantExpr(upperBoundResult, astContext)) {
      if (upperBoundResult.Val.isInt()) {
        ub = upperBoundResult.Val.getInt().getExtValue();
        auto mlirUB = opBuilder.create<mlir::ConstantIndexOp>(UNKNOWN_LOC, ub);
        auto mlirLB = opBuilder.create<mlir::ConstantIndexOp>(UNKNOWN_LOC, lb);
        auto step = opBuilder.create<mlir::ConstantIndexOp>(UNKNOWN_LOC, 1);
        auto forLoopOp = opBuilder.create<mlir::scf::ForOp>(
            UNKNOWN_LOC, mlirLB, mlirUB, step, llvm::None, forLoopBodyBuilder);

        //        LoopInputArgs loopInputArgs;
        //        findLoopInputs(forStmt, astContext, loopInputArgs);
//        return forLoopOp;
//        moduleOp->dump();
      }
    }
  }
}

mlir::Value CodeGen::VisitImplicitCastExpr(ImplicitCastExpr *implicitCastExpr) {
  return Visit(implicitCastExpr->getSubExpr());
}

mlir::Value
CodeGen::VisitArraySubscriptExpr(ArraySubscriptExpr *arraySubscriptExpr) {
  const auto indexValue = Visit(arraySubscriptExpr->getRHS());
  auto mlirType = typeGen.Visit(arraySubscriptExpr);
  //  auto load =
  //  opBuilder.create<mlir::memref::LoadOp>(UNKNOWN_LOC, )
  return nullptr;
}

mlir::Value CodeGen::VisitBinaryOperator(BinaryOperator *binOp) {
  auto lhsRes = Visit(binOp->getLHS());
  auto rhsRes = Visit(binOp->getRHS());
  return nullptr;
}

mlir::Value CodeGen::VisitDeclRefExpr(DeclRefExpr *declRef) {
  auto declName = declRef->getDecl()->getName();
  auto allocaVal = scopedHashTable.lookup(declName);
  assert(
      allocaVal &&
      "declRefExpr should be in the symbol table at this point, but it's not");
  auto memrefType = typeGen.VisitDeclRefExpr(declRef);
  //  auto memref = opBuilder.create<mlir::>
  auto loadVal = opBuilder.create<mlir::memref::LoadOp>(
      UNKNOWN_LOC, allocaVal.getType(), allocaVal);
  return loadVal;
}

mlir::Value CodeGen::VisitIntegerLiteral(IntegerLiteral *integerLiteral) {
  uint64_t val = *(integerLiteral->getValue().getRawData());
  auto type = mlir::IntegerType::get(mlirContext,
                                     integerLiteral->getValue().getBitWidth());
  return opBuilder.create<mlir::ConstantOp>(UNKNOWN_LOC, type,
                                            mlir::IntegerAttr::get(type, val));
}

// using NamedAttrs = llvm::SmallVector<mlir::NamedAttribute>;
// static void generateFunctionAttributes(NamedAttrs& namedAttrs, utils::Decls
// &decls, mlir::MLIRContext *mlirContext) {
//  for (auto &pair : decls) {
//    namedAttrs.push_back({mlir::Identifier::get(pair.first, mlirContext),})
//  }
//}
mlir::LogicalResult CodeGen::declare(llvm::StringRef var, mlir::Value value) {
  if (scopedHashTable.count(var))
    return mlir::failure();
  scopedHashTable.insert(var, value);
  return mlir::success();
}

void CodeGen::run() {
  /// First creating a function enclosing the loop,
  /// this function will take as argument the DeclRefExprs found inside the loop
  utils::Decls loopInputs;
  auto forS = const_cast<ForStmt *>(forStmt);
  utils::findLoopInputs(forS, astContext, loopInputs);

  /// creating the function

  SmallVector<mlir::Type> args;
  SmallVector<StringRef> argNames;
  for (auto pair : loopInputs) {
    args.push_back(typeGen.getType(pair.second));
    argNames.push_back(pair.first);
  }
  auto funcType = opBuilder.getFunctionType(args, llvm::None);

  llvm::ScopedHashTableScope<StringRef, mlir::Value> scope(scopedHashTable);

  /// TODO find a better name (mangled) for the function
  auto funcOp =
      opBuilder.create<mlir::FuncOp>(UNKNOWN_LOC, "the_for_loop", funcType);

  auto &entryBlock = *funcOp.addEntryBlock();
//  auto &region = funcOp.region();

  for (const auto p : llvm::zip(argNames, entryBlock.getArguments())) {
    if (mlir::failed(declare(std::get<0>(p), std::get<1>(p)))) {
      assert(false && "shouldn't have failed");
      return;
    }
  }

  opBuilder.setInsertionPointToEnd(&entryBlock);

  auto forOp = Visit(forS);

  moduleOp.push_back(funcOp);
  moduleOp.dump();
}

} // namespace clang::tuner