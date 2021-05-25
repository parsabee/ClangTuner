//
// Created by parsa on 5/14/21.
//

#include "CodeGen.h"
#include "ClangTune/Dialect.h"
#include "FunctionCreator.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"

namespace clang::tuner {

#define MATCH_DECL_REF "MatchDeclRefs"
using LoopInputArgs = std::map<llvm::StringRef, const Expr *>;

#define UNKNOWN_LOC opBuilder.getUnknownLoc()

static StringRef getArrayName(ArraySubscriptExpr *arraySubscriptExpr) {
  auto *base = arraySubscriptExpr->getBase();
  while (auto *arr = dyn_cast<ArraySubscriptExpr>(base)) {
    base = arr->getLHS();
  }
  if (auto *cast = dyn_cast<ImplicitCastExpr>(base)) {
    if (auto *decl = dyn_cast<DeclRefExpr>(cast->getSubExpr())) {
      return decl->getNameInfo().getAsString();
    }
  }
  return "";
}

mlir::LogicalResult CodeGen::declare(llvm::StringRef var, mlir::Value value) {
  if (scopedHashTable.count(var))
    return mlir::failure();
  scopedHashTable.insert(var, value);
  return mlir::success();
}

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

mlir::Value CodeGen::createConstantIndex(unsigned int index) {
  auto retval = constants.lookup(index);
  if (retval != nullptr)
    return retval;
  auto val = opBuilder.create<mlir::ConstantIndexOp>(UNKNOWN_LOC, index);
  constants.insert(index, val);
  return val;
}

mlir::Value CodeGen::handleVarDecl(VarDecl *varDecl) {
  auto type = typeGen.getType(varDecl);
  assert(type && "type must be valid at this point");
  auto alloca = opBuilder.create<mlir::memref::AllocaOp>(UNKNOWN_LOC, type);
  if (varDecl->hasInit()) {
    auto val = Visit(varDecl->getInit());
    /// Fixme this only covers ints
    opBuilder.create<mlir::memref::StoreOp>(UNKNOWN_LOC, val, alloca,
                                            createConstantIndex(0));
  }
  scopedHashTable.insert(varDecl->getName(), alloca);
}

mlir::Value CodeGen::VisitDeclStmt(DeclStmt *declStmt) {
  for (auto decl : declStmt->decls()) {
    if (auto varDecl = dyn_cast<VarDecl>(decl)) {
      handleVarDecl(varDecl);
    } else
      assert(false && "only var decl is supported at the momemnt");
  }
  return nullptr; // nothing to return
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

mlir::Value CodeGen::VisitForStmt(ForStmt *forStmt) {
  llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> forScope(
      scopedHashTable);

  llvm::ScopedHashTableScope<int, mlir::Value> constScope(constants);

  const auto forLoopInitStmt = forStmt->getInit();
  const auto forLoopInitDeclStmt = dyn_cast<DeclStmt>(forLoopInitStmt);

  assert(forLoopInitDeclStmt && "init stmt is not a decl stmt");

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

  Expr *upperBound = nullptr, *condLHS = nullptr;
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

  Expr::EvalResult lowerBoundResult, upperBoundResult;
  if (lowerBound->EvaluateAsConstantExpr(lowerBoundResult, astContext)) {
    uint64_t lb, ub;
    if (lowerBoundResult.Val.isInt())
      lb = lowerBoundResult.Val.getInt().getExtValue();
    if (upperBound->EvaluateAsConstantExpr(upperBoundResult, astContext)) {
      if (upperBoundResult.Val.isInt()) {

        /// generating the upper/lower bounds and step
        ub = upperBoundResult.Val.getInt().getExtValue();
        auto mlirUB = opBuilder.create<mlir::ConstantIndexOp>(UNKNOWN_LOC, ub);
        auto mlirLB = opBuilder.create<mlir::ConstantIndexOp>(UNKNOWN_LOC, lb);
        auto step = opBuilder.create<mlir::ConstantIndexOp>(UNKNOWN_LOC, 1);

        /// generating the forloop and saving its induction variable
        auto forLoopOp = opBuilder.create<mlir::scf::ForOp>(
            UNKNOWN_LOC, mlirLB, mlirUB, step, llvm::None);
        //        forLoopBodyBuilder();
        auto inductionVar = forLoopOp.getInductionVar();
        auto didDeclare = declare(varDecl->getName(), inductionVar);
        assert(mlir::succeeded(didDeclare) &&
               "induction variable must not be present in the hash table");

        /// generating the body of the for loop
        auto &forLoopRegion = forLoopOp.getLoopBody();
        auto &forLoopBlock = forLoopRegion.back();
        opBuilder.setInsertionPointToStart(&forLoopBlock);

        const auto body = forStmt->getBody();
        if (const auto compStmt = dyn_cast<CompoundStmt>(body)) {
          for (const auto it : compStmt->body()) {
            this->Visit(it);
          }
        } else
          this->Visit(body);

//        assert(mlir::succeeded(mlir::verify(forLoopOp)) && "invalid forloop");
      }
    }
  }
  return nullptr;
}

mlir::Value CodeGen::VisitImplicitCastExpr(ImplicitCastExpr *implicitCastExpr) {
  return Visit(implicitCastExpr->getSubExpr());
}

mlir::Value
CodeGen::VisitArraySubscriptExpr(ArraySubscriptExpr *arraySubscriptExpr) {
  const auto idx = Visit(arraySubscriptExpr->getIdx());
  auto name = getArrayName(arraySubscriptExpr);
  mlir::Value SSAVal = scopedHashTable.lookup(name);
  assert(SSAVal && "array must be present in the hash table");
  return opBuilder.create<mlir::memref::LoadOp>(UNKNOWN_LOC, SSAVal, idx);
}

template <typename F1, typename F2>
static mlir::Value genericBinOpHandler(
    mlir::Value &lhs, mlir::Value &rhs, F1 &&ifInt,
    F2 &&ifFloat) { // pass by rvalue to restrict F1 and F2 to closures
  auto lhsT = lhs.getType();
  auto rhsT = rhs.getType();
  if ((lhsT.isSignedInteger() || lhsT.isUnsignedInteger()) &&
      (rhsT.isSignedInteger() || rhsT.isUnsignedInteger())) {
    return ifInt();
  } else {
    return ifFloat();
  }
}

mlir::Value CodeGen::handleAssignment(mlir::Value &lhs, mlir::Value &rhs) {
  opBuilder.create<mlir::memref::StoreOp>(UNKNOWN_LOC, lhs, rhs);
  return nullptr;
}

mlir::Value CodeGen::handleAssignment(Expr *lhs, Expr *rhs) {
  auto rhsVal = Visit(rhs);
  if (auto *arraySubscript = dyn_cast<ArraySubscriptExpr>(lhs)) {
    const auto idx = Visit(arraySubscript->getIdx());
    auto name = getArrayName(arraySubscript);
    mlir::Value SSAVal = scopedHashTable.lookup(name);
    assert(SSAVal);
    opBuilder.create<mlir::memref::StoreOp>(UNKNOWN_LOC, rhsVal, SSAVal, idx);
  }
  return nullptr;
}

mlir::Value CodeGen::handleSubtraction(mlir::Value &lhs, mlir::Value &rhs) {
  return genericBinOpHandler(
      lhs, rhs,
      [this, &lhs, &rhs]() -> mlir::Value {
        return opBuilder.create<mlir::SubIOp>(UNKNOWN_LOC, lhs.getType(), lhs,
                                              rhs);
      },
      [this, &lhs, &rhs]() -> mlir::Value {
        return opBuilder.create<mlir::SubFOp>(UNKNOWN_LOC, lhs.getType(), lhs,
                                              rhs);
      });
}
mlir::Value CodeGen::handleMultiplication(mlir::Value &lhs, mlir::Value &rhs) {
  return genericBinOpHandler(
      lhs, rhs,
      [this, &lhs, &rhs]() -> mlir::Value {
        return opBuilder.create<mlir::MulFOp>(UNKNOWN_LOC, lhs.getType(), lhs,
                                              rhs);
      },
      [this, &lhs, &rhs]() -> mlir::Value {
        return opBuilder.create<mlir::MulIOp>(UNKNOWN_LOC, lhs.getType(), lhs,
                                              rhs);
      });
}
mlir::Value CodeGen::handleDivision(mlir::Value &lhs, mlir::Value &rhs) {
  return genericBinOpHandler(
      lhs, rhs,
      [this, &lhs, &rhs]() -> mlir::Value {
        return opBuilder.create<mlir::DivFOp>(UNKNOWN_LOC, lhs.getType(), lhs,
                                              rhs);
      },
      [this, &lhs, &rhs]() -> mlir::Value {
        return opBuilder.create<mlir::DivFOp>(UNKNOWN_LOC, lhs.getType(), lhs,
                                              rhs);
      });
}
mlir::Value CodeGen::handleAddition(mlir::Value &lhs, mlir::Value &rhs) {

  return genericBinOpHandler(
      lhs, rhs,
      [this, &lhs, &rhs]() -> mlir::Value {
        return opBuilder.create<mlir::AddIOp>(UNKNOWN_LOC, lhs.getType(), lhs,
                                              rhs);
      },
      [this, &lhs, &rhs]() -> mlir::Value {
        return opBuilder.create<mlir::AddIOp>(UNKNOWN_LOC, lhs.getType(), lhs,
                                              rhs);
      });
}

mlir::Value CodeGen::VisitBinaryOperator(BinaryOperator *binOp) {
  if (binOp->isAssignmentOp()) {
    handleAssignment(binOp->getLHS(), binOp->getRHS());
    return nullptr;
  }

  auto lhs = Visit(binOp->getLHS());
  auto rhs = Visit(binOp->getRHS());
  assert(lhs != nullptr && rhs != nullptr);
  switch (binOp->getOpcode()) {
  case BO_Add:
    return handleAddition(lhs, rhs);
  case BO_Mul:
    return handleMultiplication(lhs, rhs);
  case BO_Div:
    return handleDivision(lhs, rhs);
  case BO_Sub:
    return handleSubtraction(lhs, rhs);
  default:
    return nullptr;
  }
}

mlir::Value CodeGen::VisitDeclRefExpr(DeclRefExpr *declRef) {
  auto declName = declRef->getDecl()->getName();
  auto val = scopedHashTable.lookup(declName);
  assert(
      val &&
      "declRefExpr should be in the symbol table at this point, but it's not");
  return val;
  //  mlir::MemRefType memrefType = typeGen.VisitDeclRefExpr(declRef);
  //  //  auto memref = opBuilder.create<mlir::>
  //  auto loadVal =
  //      opBuilder.create<mlir::memref::LoadOp>(UNKNOWN_LOC, memrefType,
  //      createConstantIndex(0));
  //  return loadVal;
}

mlir::Value CodeGen::VisitIntegerLiteral(IntegerLiteral *integerLiteral) {
  uint64_t val = *(integerLiteral->getValue().getRawData());
  auto type = mlir::IntegerType::get(mlirContext,
                                     integerLiteral->getValue().getBitWidth());
  return opBuilder.create<mlir::ConstantOp>(UNKNOWN_LOC, type,
                                            mlir::IntegerAttr::get(type, val));
}

bool ForLoopArgs::append(mlir::Type type, StringRef name) {
  if (nameArgMap.find(name) == nameArgMap.end()) {
    nameArgMap[name] = nameArgMap.size();
    argTypes.push_back(type);
    argNames.push_back(name);
    return true;
  }
  return false;
}

const mlir::Type ForLoopArgs::lookUp(StringRef name) const {
  if (nameArgMap.find(name) != nameArgMap.end()) {
    return argTypes[nameArgMap.lookup(name)];
  }
  return nullptr;
}

const SmallVector<mlir::Type> &ForLoopArgs::getArgTypes() const {
  return argTypes;
}

const SmallVector<StringRef> &ForLoopArgs::getArgNames() const {
  return argNames;
}

void CodeGen::run() {
  /// First creating a function enclosing the loop,
  /// this function will take as argument the DeclRefExprs found inside the loop
  utils::Decls loopInputs;
  auto forS = const_cast<ForStmt *>(forStmt);
  utils::findLoopInputs(forS, astContext, loopInputs);

  /// creating the function

  for (auto pair : loopInputs) {
    loopArgs.append(typeGen.getType(pair.second), pair.first);
  }
  auto funcType = opBuilder.getFunctionType(loopArgs.getArgTypes(), llvm::None);
  llvm::ScopedHashTableScope<StringRef, mlir::Value> scope(scopedHashTable);
  llvm::ScopedHashTableScope<int, mlir::Value> constScope(constants);

  /// TODO find a better name (mangled) for the function
  auto funcOp =
      opBuilder.create<mlir::FuncOp>(UNKNOWN_LOC, "the_for_loop", funcType);
  auto &entryBlock = *funcOp.addEntryBlock();
  for (auto it : llvm::zip(loopArgs.getArgNames(), entryBlock.getArguments())) {
    auto val = declare(std::get<0>(it), std::get<1>(it));
    assert(mlir::succeeded(val));
  }
  opBuilder.setInsertionPointToEnd(&entryBlock);

  auto forOp = Visit(forS);

  moduleOp.push_back(funcOp);
  opBuilder.setInsertionPointToEnd(&entryBlock);
  opBuilder.create<mlir::ReturnOp>(UNKNOWN_LOC);
  if (mlir::failed(mlir::verify(moduleOp))) {
    moduleOp.emitError("module verification error");
  }

  mlir::PassManager pm(mlirContext);
  pm.addPass(mlir::createLowerToCFGPass());
  pm.addPass(mlir::createLowerToLLVMPass());
  if (mlir::failed(pm.run(moduleOp)))
    moduleOp->emitError("failed to lower module");

  moduleOp.dump();

}

} // namespace clang::tuner