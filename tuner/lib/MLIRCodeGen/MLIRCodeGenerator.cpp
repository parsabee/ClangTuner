//
// Created by parsa on 5/14/21.
//

#include "clangtuner/MLIRCodeGen/MLIRCodeGenerator.h"
#include "clangtuner/AttrForLoopFunctionDeclarator.h"
#include "AttrForLoopRefactorer.h"
#include "CommandLineOpts.h"
#include "IOUtils.h"
#include "OpenMPConfigurer.h"
#include "ParallelizingPass.h"

#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Program.h"

#include <mutex>

namespace clang::tuner {

using Declarations = std::map<llvm::StringRef, const Expr *>;

#define UNKNOWN_LOC opBuilder.getUnknownLoc()

static std::string getArrayName(ArraySubscriptExpr *arraySubscriptExpr) {

  clang::Expr *base = arraySubscriptExpr->getBase();

  // An array subscript can be multi-dimensional
  // in that case we need to iteratively get the base of the array subscript to
  // get to its identifier
  while (true) {
    if (auto *arrSub = dyn_cast<ArraySubscriptExpr>(base)) {
      base = arrSub->getBase();
    } else if (auto *implCast = dyn_cast<ImplicitCastExpr>(base)) {
      base = implCast->getSubExpr();
    } else {
      break;
    }
  }

  if (auto *decl = dyn_cast<DeclRefExpr>(base)) {
    return decl->getNameInfo().getAsString();
  }

  assert(false && "couldn't find declaration");
  return "";
}

void MLIRCodeGenerator::declare(llvm::StringRef var, mlir::Value value) {
  // We know that it is a valid declaration because clang has semantic checked
  scopedHashTable.insert(var, value);
}

mlir::Value MLIRCodeGenerator::createConstantIndex(unsigned int index) {
  auto retval = indexConstants.lookup(index);
  if (retval != nullptr)
    return retval;
  auto val = opBuilder.create<mlir::ConstantIndexOp>(UNKNOWN_LOC, index);
  indexConstants.insert({index, val});
  return val;
}

mlir::Value MLIRCodeGenerator::handleVarDecl(VarDecl *varDecl) {
  mlir::MemRefType memrefType = typeGen.getMemRefType(varDecl);
  assert(memrefType && "type must be valid at this point");

  mlir::Value allocation =
      opBuilder.create<mlir::memref::AllocaOp>(UNKNOWN_LOC, memrefType);
  scopedHashTable.insert(varDecl->getName(), allocation);

  if (varDecl->hasInit()) {
    auto val = Visit(varDecl->getInit());
    /// Fixme this only covers ints
    opBuilder.create<mlir::memref::StoreOp>(UNKNOWN_LOC, val, allocation,
                                            createConstantIndex(0));
  }
  return allocation;
}

mlir::Value MLIRCodeGenerator::VisitDeclStmt(DeclStmt *declStmt) {
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
  assert(loopDecl && "loop decl must be a var decl");
  const QualType type = loopDecl->getType();
  assert(type->isBuiltinType() && "var decl must have a builtin type");
  const llvm::StringRef varName = loopDecl->getName();
  Expr *init = const_cast<Expr *>(loopDecl->getInit());
  assert(init && "var decl is not initialized");
  initDecls.push_back({{type, varName}, init});
}

static void checkForLoopExpressions(ForStmt *forStmt, ASTContext &astContext,
                                    Expr *condLHS,
                                    ForLoopInitDecls &initDecls) {
  bool isCondLHSInInitDecls = false;
  auto checkInitDeclsForExpr = [&initDecls](llvm::StringRef condVar) -> bool {
    for (const auto &it : initDecls) {
      if (it.first.second == condVar) {
        return true;
      }
    }
    return false;
  };

  if (auto *const castExpr = dyn_cast<ImplicitCastExpr>(condLHS)) {
    if (auto *const expr = dyn_cast<DeclRefExpr>(castExpr->getSubExpr())) {
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
}

static void getLoopBounds(ForStmt *forStmt, ForLoopInitDecls &initDecls,
                          Expr *&lowerBound, Expr *&upperBound,
                          Expr *&condLHS) {

  for (const auto &it : initDecls) {
    lowerBound = it.second;
  }

  const auto cond = forStmt->getCond();
  assert(cond && "for loop doesn't have a conditional stmt");

  if (auto *const binOp = dyn_cast<BinaryOperator>(cond)) {
    if (binOp->getOpcode() == BinaryOperator::Opcode::BO_LT ||
        binOp->getOpcode() == BinaryOperator::Opcode::BO_LE) {
      upperBound = binOp->getRHS();
      condLHS = binOp->getLHS();
    }
  }

  assert(condLHS && upperBound && "invalid loop condition");
}

mlir::Value MLIRCodeGenerator::forLoopHandler(ForStmt *theForStmt,
                                              bool isParallel) {
  llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> forScope(
      scopedHashTable);

  auto *const forLoopInitStmt = theForStmt->getInit();
  auto *const forLoopInitDeclStmt = dyn_cast<DeclStmt>(forLoopInitStmt);

  assert(forLoopInitDeclStmt && "init stmt is not a decl stmt");

  ForLoopInitDecls initDecls;
  assert(forLoopInitDeclStmt->isSingleDecl() && "loop must have a single decl");

  auto *const varDecl = dyn_cast<VarDecl>(forLoopInitDeclStmt->getSingleDecl());
  handleLoopDecls(varDecl, initDecls);

  Expr *lowerBound = nullptr, *upperBound = nullptr, *condLHS = nullptr;

  getLoopBounds(const_cast<ForStmt *>(theForStmt), initDecls, lowerBound,
                upperBound, condLHS);

  checkForLoopExpressions(const_cast<ForStmt *>(theForStmt), astContext,
                          condLHS, initDecls);

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

        mlir::Region *loopBody = nullptr;
        mlir::ValueRange inductionVars;
        auto forLoopOp = opBuilder.create<mlir::scf::ForOp>(
            UNKNOWN_LOC, mlirLB, mlirUB, step, llvm::None);
        loopBody = &forLoopOp.getLoopBody();
        inductionVars = forLoopOp.getInductionVar();

        for (const auto &val : inductionVars) {
          declare(varDecl->getName(), val);
        }

        /// generating the body of the for loop
        opBuilder.setInsertionPointToStart(&loopBody->back());

        auto *const body = theForStmt->getBody();
        if (auto *const compStmt = dyn_cast<CompoundStmt>(body)) {
          for (auto *const it : compStmt->body()) {
            this->Visit(it);
          }
        } else
          this->Visit(const_cast<Stmt *>(body));
      }
    }
  }
  return nullptr;
}

mlir::Value MLIRCodeGenerator::VisitForStmt(ForStmt *forStmt) {
  return forLoopHandler(forStmt, false);
}

mlir::Value
MLIRCodeGenerator::VisitImplicitCastExpr(ImplicitCastExpr *implicitCastExpr) {
  return Visit(implicitCastExpr->getSubExpr());
}

mlir::Value MLIRCodeGenerator::VisitArraySubscriptExpr(
    ArraySubscriptExpr *arraySubscriptExpr) {

  auto name = getArrayName(arraySubscriptExpr);
  mlir::Value val = scopedHashTable.lookup(name);
  assert(val && "array must be present in the hash table");

  auto getIndex = [this](ArraySubscriptExpr *arrSub) -> mlir::Value {
    auto index = Visit(arrSub->getIdx());
    assert(index && " index shouldn't be null ");
    return index;
  };

  llvm::SmallVector<mlir::Value> indices;
  clang::Expr *base = arraySubscriptExpr->getBase();
  indices.push_back(getIndex(arraySubscriptExpr));

  // An array subscript can be multi-dimensional
  // in that case we need to iteratively get the base of the array subscript to
  // get to its identifier
  while (true) {
    if (auto *arrSub = dyn_cast<ArraySubscriptExpr>(base)) {
      base = arrSub->getBase();
      indices.push_back(getIndex(arrSub));
    } else if (auto *implCast = dyn_cast<ImplicitCastExpr>(base)) {
      base = implCast->getSubExpr();
    } else {
      break;
    }
  }

  return opBuilder.create<mlir::memref::LoadOp>(UNKNOWN_LOC, val, indices);
}

/// This function is a generic (all binop operations) handler for int and float
/// binops, F1 is a callback for operation involving ints, F2 callback for
/// operations with floats
/// the 'requires' clause makes sure F1 and F2 are functions/functors/lambdas
/// (invocable)
template <typename F1, typename F2>
requires std::is_invocable<F1>::value &&
    std::is_invocable<F2>::value static mlir::Value
    genericBinOpHandler(mlir::Value &lhs, mlir::Value &rhs, F1 &&ifInt,
                        F2 &&ifFloat) {
  auto lhsT = lhs.getType();
  auto rhsT = rhs.getType();
  if ((lhsT.isSignedInteger() || lhsT.isUnsignedInteger()) &&
      (rhsT.isSignedInteger() || rhsT.isUnsignedInteger())) {
    return ifInt();
  } else {
    return ifFloat();
  }
}

mlir::Value MLIRCodeGenerator::handleAssignment(mlir::Value &lhs,
                                                mlir::Value &rhs) {
  opBuilder.create<mlir::memref::StoreOp>(UNKNOWN_LOC, lhs, rhs);
  return nullptr;
}

mlir::Value MLIRCodeGenerator::handleAssignment(Expr *lhs, Expr *rhs) {
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

mlir::Value MLIRCodeGenerator::handleSubtraction(mlir::Value &lhs,
                                                 mlir::Value &rhs) {
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

mlir::Value MLIRCodeGenerator::handleMultiplication(mlir::Value &lhs,
                                                    mlir::Value &rhs) {
  return genericBinOpHandler(
      lhs, rhs,
      [this, &lhs, &rhs]() -> mlir::Value {
        return opBuilder.create<mlir::MulIOp>(UNKNOWN_LOC, lhs.getType(), lhs,
                                              rhs);
      },
      [this, &lhs, &rhs]() -> mlir::Value {
        return opBuilder.create<mlir::MulFOp>(UNKNOWN_LOC, lhs.getType(), lhs,
                                              rhs);
      });
}

mlir::Value MLIRCodeGenerator::handleDivision(mlir::Value &lhs,
                                              mlir::Value &rhs) {
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

mlir::Value MLIRCodeGenerator::handleAddition(mlir::Value &lhs,
                                              mlir::Value &rhs) {

  return genericBinOpHandler(
      lhs, rhs,
      [this, &lhs, &rhs]() -> mlir::Value {
        return opBuilder.create<mlir::AddIOp>(UNKNOWN_LOC, lhs.getType(), lhs,
                                              rhs);
      },
      [this, &lhs, &rhs]() -> mlir::Value {
        return opBuilder.create<mlir::AddFOp>(UNKNOWN_LOC, lhs.getType(), lhs,
                                              rhs);
      });
}

mlir::Value MLIRCodeGenerator::VisitBinaryOperator(BinaryOperator *binOp) {
  if (binOp->isAssignmentOp()) {
    handleAssignment(binOp->getLHS(), binOp->getRHS());
    return nullptr;
  }

  auto lhs = Visit(binOp->getLHS());
  auto rhs = Visit(binOp->getRHS());
  assert(lhs && rhs);
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

mlir::Value MLIRCodeGenerator::VisitDeclRefExpr(DeclRefExpr *declRef) {
  auto declName = declRef->getDecl()->getName();
  auto val = scopedHashTable.lookup(declName);
  assert(val && "declRefExpr should be in the symbol table at this point, "
                "but it's not");
  return val;
}

mlir::Value
MLIRCodeGenerator::VisitIntegerLiteral(IntegerLiteral *integerLiteral) {
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

const std::vector<mlir::Type> &ForLoopArgs::getArgTypes() const {
  return argTypes;
}

const std::vector<StringRef> &ForLoopArgs::getArgNames() const {
  return argNames;
}

std::unique_ptr<mlir::ModuleOp>
MLIRCodeGenerator::runOpt(mlir::ModuleOp *module,
                          llvm::SmallVector<StringRef> &optArgs) {
  auto *context = module->getContext();
  int tempStoredFileFD;
  SmallString<32> tempStoredFile;
  SmallString<256> MLIROptimizedFile;
  llvm::sys::fs::createTemporaryFile("tmp-mlir-file", "mlir", tempStoredFileFD,
                                     tempStoredFile);
  int MLIROptimizedFileFD;
  const auto *sources = &OptionsParser->getSourcePathList();
  const auto name = sources->at(0) + "_for_loops_opt.mlir";

  // Remove the temporary file if it already exists
  if (llvm::sys::fs::exists(name)) {
    llvm::FileRemover fileRemover(name.c_str());
  }
  llvm::sys::fs::createUniqueFile(name, MLIROptimizedFileFD, MLIROptimizedFile);

  std::error_code EC;
  llvm::raw_fd_ostream rawStream(tempStoredFile, EC, llvm::sys::fs::F_None);
  if (rawStream.has_error()) {
    return nullptr;
  }

  rawStream << *module;
  rawStream.flush();
  rawStream.close();

  llvm::FileRemover tempStoredFileRemover(tempStoredFile.c_str());
  Optional<StringRef> Redirects[] = {StringRef(MLIROptimizedFile),
                                     StringRef(MLIROptimizedFile),
                                     StringRef(MLIROptimizedFile)};

  optArgs.insert(optArgs.begin(), "mlir-opt");
  optArgs.push_back(tempStoredFile);
  auto pathOrErr = llvm::sys::findProgramByName("mlir-opt");
  if (!pathOrErr)
    return nullptr;

  const std::string &path = *pathOrErr;
  int RunResult = llvm::sys::ExecuteAndWait(path, optArgs, None, Redirects);
  if (RunResult != 0)
    return nullptr;

  auto OutputBuf = llvm::MemoryBuffer::getFile(MLIROptimizedFile.c_str());
  if (!OutputBuf)
    return nullptr;

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(OutputBuf.get()), llvm::SMLoc());

  auto optimizedModule = mlir::parseSourceFile(sourceMgr, context);

  // clean-up
  if (!SaveOptimizedMLIRFile.getValue()) {
    llvm::FileRemover tempResultFileRemover(MLIROptimizedFile.c_str());
  }

  return std::make_unique<mlir::ModuleOp>(optimizedModule.release());
}

// std::unique_ptr<mlir::ModuleOp>
// runOpt(mlir::ModuleOp *module, mlir::MLIRContext *context,
//       llvm::SmallVector<StringRef> &optArgs) {
//  int tempStoredFileFD;
//  SmallString<32> tempStoredFile;
//  SmallString<256> MLIROptimizedFile;
//  llvm::sys::fs::createTemporaryFile("tmp-mlir-file", "mlir",
//  tempStoredFileFD,
//                                     tempStoredFile);
//  int MLIROptimizedFileFD;
//  const auto *sources = &OptionsParser->getSourcePathList();
//  const auto name = sources->at(0) + "_for_loops_opt.mlir";
//
//  // Remove the temporary file if it already exists
//  if (llvm::sys::fs::exists(name)) {
//    llvm::FileRemover fileRemover(name.c_str());
//  }
//  llvm::sys::fs::createUniqueFile(name, MLIROptimizedFileFD,
//  MLIROptimizedFile);
//
//  std::error_code EC;
//  llvm::raw_fd_ostream rawStream(tempStoredFile, EC, llvm::sys::fs::F_None);
//  if (rawStream.has_error()) {
//    return nullptr;
//  }
//
//  rawStream << *module;
//  rawStream.flush();
//  rawStream.close();
//
//  llvm::FileRemover tempStoredFileRemover(tempStoredFile.c_str());
//  Optional<StringRef> Redirects[] = {StringRef(MLIROptimizedFile),
//                                     StringRef(MLIROptimizedFile),
//                                     StringRef(MLIROptimizedFile)};
//
//  optArgs.insert(optArgs.begin(), "mlir-opt");
//  optArgs.push_back(tempStoredFile);
//  auto pathOrErr = llvm::sys::findProgramByName("mlir-opt");
//  if (!pathOrErr)
//    return nullptr;
//
//  const std::string &path = *pathOrErr;
//  int RunResult = llvm::sys::ExecuteAndWait(path, optArgs, None, Redirects);
//  if (RunResult != 0)
//    return nullptr;
//
//  auto OutputBuf = llvm::MemoryBuffer::getFile(MLIROptimizedFile.c_str());
//  if (!OutputBuf)
//    return nullptr;
//
//  llvm::SourceMgr sourceMgr;
//  sourceMgr.AddNewSourceBuffer(std::move(OutputBuf.get()), llvm::SMLoc());
//
//  auto optimizedModule = mlir::parseSourceFile(sourceMgr, context);
//
//  // clean-up
//  if (!SaveOptimizedMLIRFile.getValue()) {
//    llvm::FileRemover tempResultFileRemover(MLIROptimizedFile.c_str());
//  }
//
//  return std::make_unique<mlir::ModuleOp>(optimizedModule.release());
//}

bool MLIRCodeGenerator::lowerToMLIR() {
  /// First creating a function enclosing the loop,
  /// this function will take as argument the DeclRefExprs found inside the loop
  utils::Decls loopInputs;
  auto forS = const_cast<ForStmt *>(forStmt);
  utils::findLoopInputs(forS, astContext, loopInputs);

  /// creating the function
  for (auto pair : loopInputs) {
    auto type = typeGen.getMemRefType(pair.second);
    assert(type && "type must not be null");
    if (type.getRank() == 0)
      loopArgs.append(type.getElementType(), pair.first);
    else
      loopArgs.append(type, pair.first);
  }

  auto funcType = opBuilder.getFunctionType(loopArgs.getArgTypes(), llvm::None);
  llvm::ScopedHashTableScope<StringRef, mlir::Value> scope(scopedHashTable);

  auto funcName =
      createFunctionName(const_cast<ForStmt *>(forStmt), sourceManager);

  auto funcOp = opBuilder.create<mlir::FuncOp>(UNKNOWN_LOC, funcName, funcType);
  auto &entryBlock = *funcOp.addEntryBlock();
  for (auto it : llvm::zip(loopArgs.getArgNames(), entryBlock.getArguments())) {
    declare(std::get<0>(it), std::get<1>(it));
  }
  opBuilder.setInsertionPointToEnd(&entryBlock);
  (void)Visit(forS);
  moduleOp.push_back(funcOp);
  opBuilder.setInsertionPointToEnd(&entryBlock);
  opBuilder.create<mlir::ReturnOp>(UNKNOWN_LOC);
  if (mlir::failed(mlir::verify(moduleOp))) {
    moduleOp.emitError("module verification error");
#ifdef DEBUG
    moduleOp->dump();
#endif
    return false;
  }

  //  // Converting top level for loop to parallel loop
  //  mlir::PassManager pm(mlirContext);
  //  mlir::OpPassManager &nestedModulePM = pm.nest<mlir::FuncOp>();
  //  nestedModulePM.addPass(std::make_unique<clang::tuner::ParallelizingPass>());
  //  if (mlir::failed(pm.run(moduleOp))) {
  //    moduleOp->emitError("failed to parallelize module");
  //#ifdef DEBUG
  //    moduleOp->dump();
  //#endif
  //    return false;
  //  }

  if (SaveInitialMLIRFile.getValue()) {
    SmallString<256> loopMLIRFile;
    const auto *sources = &OptionsParser->getSourcePathList();
    if (!writeModuleToFile(sources->at(0) + "_for_loops.mlir", loopMLIRFile,
                           moduleOp)) {
      llvm::errs() << "failed to write the generated mlir module to file\n";
    }
  }
  return true;
}

bool MLIRCodeGenerator::runParallelizingPass(mlir::ModuleOp &moduleOp,
                                             mlir::MLIRContext *mlirContext) {
  // Converting top level for loop to parallel loop
  mlir::PassManager pm(mlirContext);
  mlir::OpPassManager &nestedModulePM = pm.nest<mlir::FuncOp>();
  nestedModulePM.addPass(std::make_unique<clang::tuner::ParallelizingPass>());
  if (mlir::failed(pm.run(moduleOp))) {
    moduleOp->emitError("failed to parallelize module");
#ifdef DEBUG
    moduleOp->dump();
#endif
    return false;
  }

  if (SaveInitialMLIRFile.getValue()) {
    SmallString<256> loopMLIRFile;
    const auto *sources = &OptionsParser->getSourcePathList();
    if (!writeModuleToFile(sources->at(0) + "_parallel_for_loops.mlir",
                           loopMLIRFile, moduleOp)) {
      llvm::errs() << "failed to write the generated mlir module to file\n";
    }
  }
  return true;
}

bool MLIRCodeGenerator::lowerToLLVMDialect(mlir::ModuleOp &moduleOp,
                                           mlir::MLIRContext *mlirContext) {

  mlir::registerAllPasses();
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  mlir::PassManager mlirPM(mlirContext);

  // converts SCF ops to Standard ops
  mlirPM.addPass(mlir::createLowerToCFGPass());

  // converts standard to LLVM
  mlirPM.addPass(mlir::createLowerToLLVMPass());

  if (mlir::failed(mlirPM.run(moduleOp))) {
    moduleOp->emitError("failed to lower module");
#ifdef DEBUG
    moduleOp->dump();
#endif
    return false;
  }

  if (SaveLLVMDialectMLIRFile.getValue()) {
    SmallString<256> filename;
    const auto *sources = &OptionsParser->getSourcePathList();
    if (!writeModuleToFile(sources->at(0) + "_llvm_dialect.mlir", filename,
                           moduleOp)) {
      llvm::errs() << "failed to write the generated mlir (llvm dialect) "
                      "module to file\n";
      return false;
    }
  }

  return true;
}

std::unique_ptr<mlir::ModuleOp>
MLIRCodeGenerator::performLoweringAndOptimizationPipeline(
    llvm::SmallVector<StringRef> &optArgs) {

  //  if (!lowerToMLIR())
  //    return nullptr;
  //
  //  auto optimized = runOpt(&moduleOp, mlirContext, optArgs);
  //  if (!optimized)
  //    return nullptr;
  //
  //  if (!lowerToLLVMDialect(*optimized))
  //    return nullptr;
  //
  //  return optimized;
}

static llvm::Module
createLLVMModule(Lockable<llvm::LLVMContext, std::mutex> &lockableLLVMContext) {
  Locked locked(lockableLLVMContext);
  auto &cntx = locked.getObject();
  return llvm::Module("llvm_mod", cntx);
}

MLIRCodeGenerator::MLIRCodeGenerator(
    ForStmt *forStmt, ASTContext &context, mlir::ModuleOp &moduleOp,
    Lockable<llvm::LLVMContext, std::mutex> &lockableLLVMContext,
    mlir::OpBuilder &opBuilder, SourceManager &sourceManager,
    DiagnosticsEngine &diags)
    : sourceManager(sourceManager), lockableLLVMContext(lockableLLVMContext),
      llvmModule(createLLVMModule(lockableLLVMContext)),
      mlirContext(moduleOp->getContext()), moduleOp(moduleOp),
      opBuilder(opBuilder), astContext(context), forStmt(forStmt),
      CGModule(context, {}, {}, {}, llvmModule, diags),
      typeGen(moduleOp, CGModule.getTypes()) {}

} // namespace clang::tuner