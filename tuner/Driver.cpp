//
// Created by Parsa Bagheri on 7/3/21.
//

#include "Driver.h"
#include "CommandLineOpts.h"
#include "IOUtils.h"
#include "AttrForLoopFinder.h"

#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/IPO/StripSymbols.h"

#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Tooling/Tooling.h"

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include <future>

namespace clang::tuner {

static std::string GetExecutablePath(const char *Argv0, void *MainAddr) {
  return llvm::sys::fs::getMainExecutable(Argv0, MainAddr);
}

//===---------------------------------------------------------------------===//
// State query functions
//===---------------------------------------------------------------------===//

static inline bool isFinished(Driver::Status status) {
  return status != Driver::Unfinished;
}

Driver::Status Driver::getMLIRCodeGenStatus() {
  auto lock = MLIRCodeGenStatus.lock();
  return lock.getObject();
}

Driver::Status Driver::getMLIRCompilationStatus() {
  auto lock = MLIRCompilationStatus.lock();
  return lock.getObject();
}

Driver::Status Driver::getRefactoringStatus() {
  auto lock = RefactoringStatus.lock();
  return lock.getObject();
}

Driver::Status Driver::getClangCompilationStatus() {
  auto lock = ClangCompilationStatus.lock();
  return lock.getObject();
}

Driver::Status Driver::getLLVMLinkingStatus() {
  auto lock = LLVMLinkingStatus.lock();
  return lock.getObject();
}

bool Driver::isMLIRCodeGenFinished() {
  return isFinished(getMLIRCodeGenStatus());
}

bool Driver::isMLIRCompilationFinished() {
  return isFinished(getMLIRCompilationStatus());
}

bool Driver::isRefactoringFinished() {
  return isFinished(getRefactoringStatus());
}

bool Driver::isClangCompilationFinished() {
  return isFinished(getClangCompilationStatus());
}

bool Driver::isLLVMLinkingFinished() {
  return isFinished(getLLVMLinkingStatus());
}

void Driver::setMLIRCodeGenStatus(Status status) {
  auto lock = MLIRCodeGenStatus.lock();
  lock.getObject() = status;
}

void Driver::setMLIRCompilationStatus(Status status) {
  auto lock = MLIRCompilationStatus.lock();
  lock.getObject() = status;
}

void Driver::setRefactoringStatus(Status status) {
  auto lock = RefactoringStatus.lock();
  lock.getObject() = status;
}

void Driver::setClangCompilationStatus(Status status) {
  auto lock = ClangCompilationStatus.lock();
  lock.getObject() = status;
}

void Driver::setLLVMLinkingStatus(Status status) {
  auto lock = LLVMLinkingStatus.lock();
  lock.getObject() = status;
}

void Driver::runStripSymbolsPass(llvm::Module &module) {
  llvm::ModuleAnalysisManager llvmAM;
  llvm::StripSymbolsPass llvmPass;
  llvmPass.run(module, llvmAM);
}

// ===--------------------------------------------------------------------=== //
// Pipeline functions
// ===--------------------------------------------------------------------=== //

static auto translateMLIRModule(
    Lockable<llvm::LLVMContext, std::mutex> &lockableLLVMContext,
    mlir::ModuleOp &moduleOp) {
  Locked<llvm::LLVMContext, std::mutex> lock(lockableLLVMContext);
  auto &llvmContext = lock.getObject();
  return mlir::translateModuleToLLVMIR(moduleOp, llvmContext);
}

std::unique_ptr<llvm::Module>
Driver::convertToLLVMIR(FuturePtr<mlir::ModuleOp> futureModuleOp) {

  if (!futureModuleOp.valid())
    return nullptr;

  auto moduleOp = futureModuleOp.get();
  mlir::registerLLVMDialectTranslation(*moduleOp->getContext());

  auto newLLVMModule = translateMLIRModule(lockableLLVMContext, *moduleOp);

  if (!newLLVMModule) {
    llvm::errs() << "failed to emit LLVM IR\n";
    return nullptr;
  }

  if (SaveLLVMModule.getValue()) {
    SmallString<256> fileName;
    const auto &sources = OptionsParser->getSourcePathList();
    if (!writeModuleToFile<llvm::Module>(sources.at(0) + "_for_loops.ll",
                                         fileName, *newLLVMModule))
      llvm::errs() << "failed to write the generated llvm "
                      "module to file\n";
  }

  return newLLVMModule;
}

/// Creating a compilation instance from the args
static std::tuple<std::unique_ptr<driver::Compilation>,
                  std::unique_ptr<CodeGenAction>>
PerformClangCompilationHelper(
    clang::driver::Driver &TheDriver, std::vector<const char *> &Args,
    DiagnosticsEngine &Diags, const char *Argv0, void *MainAddr,
    Lockable<llvm::LLVMContext, std::mutex> &lockableLLVMContext) {

  std::unique_ptr<driver::Compilation> compilation(
      TheDriver.BuildCompilation(Args));

  if (!compilation)
    return {nullptr, nullptr};

  const driver::JobList &Jobs = compilation->getJobs();
  if (Jobs.size() != 1 || !isa<driver::Command>(*Jobs.begin())) {
    SmallString<256> Msg;
    llvm::raw_svector_ostream OS(Msg);
    Jobs.Print(OS, "; ", true);
    Diags.Report(diag::err_fe_expected_compiler_job) << OS.str();
    return {nullptr, nullptr};
  }

  const driver::Command &Cmd = cast<driver::Command>(*Jobs.begin());
  if (llvm::StringRef(Cmd.getCreator().getName()) != "clang") {
    Diags.Report(diag::err_fe_expected_clang_command);
    return {nullptr, nullptr};
  }

  // Initialize a compiler invocation object from clang (-cc1) arguments.
  const llvm::opt::ArgStringList &CCArgs = Cmd.getArguments();
  std::unique_ptr<CompilerInvocation> compilerInvocation(
      new CompilerInvocation);
  CompilerInvocation::CreateFromArgs(*compilerInvocation, CCArgs, Diags);

  // FIXME: This is copied from cc1_main.cpp; simplify and eliminate.
  // Create a compiler instance to handle the actual work.
  CompilerInstance Clang;
  Clang.setInvocation(std::move(compilerInvocation));

  // Create the compilers actual diagnostics engine.
  Clang.createDiagnostics();
  if (!Clang.hasDiagnostics())
    return {nullptr, nullptr};

  if (Clang.getHeaderSearchOpts().UseBuiltinIncludes &&
      Clang.getHeaderSearchOpts().ResourceDir.empty())
    Clang.getHeaderSearchOpts().ResourceDir =
        CompilerInvocation::GetResourcesPath(Argv0, MainAddr);

  // Create and execute the frontend to generate an LLVM bitcode module.
  Locked<llvm::LLVMContext, std::mutex> locked(lockableLLVMContext);
  auto *llvmContext = &locked.getObject();
  std::unique_ptr<CodeGenAction> llvmIRAction(
      new EmitLLVMOnlyAction(llvmContext));
  if (!Clang.ExecuteAction(*llvmIRAction)) {
    return {nullptr, nullptr};
  }

  return {std::move(compilation), std::move(llvmIRAction)};
}

std::future<bool>
Driver::PerformMLIRCodeGeneration(MLIRQueueProducer &mlirModuleProducer) {

  auto futureStatus =
      std::async(std::launch::async, [this, &mlirModuleProducer] {
        auto frontendAction = clang::tuner::newMLIRCodeGenFrontendActionFactory(
            MLIRCodeGenStatus, lockableLLVMContext, mlirModuleProducer, diags,
            [this, &mlirModuleProducer] {
              mlirModuleProducer.close();
              setMLIRCodeGenStatus(Success);
            });

        clang::tooling::ClangTool Tool(OptionsParser->getCompilations(),
                                       OptionsParser->getSourcePathList());

        if (Tool.run(frontendAction.get()) != 0) {
          setMLIRCodeGenStatus(Fail);
          return false;

        } else {
          // Not setting the status here, instead set it in the visitor object;
          // because clang tools are executed on another process, and it doesn't
          // come here unless that process is terminated

          // setMLIRCodeGenStatus(Success);
          return true;
        }
      });

  return futureStatus;
}

std::future<bool>
Driver::PerformMLIRCompilation(MLIRQueueConsumer &mlirConsumer,
                               LLVMQueueProducer &llvmProducer) {

  auto futureStatus =
      std::async(std::launch::async, [this, &mlirConsumer, &llvmProducer] {
        bool status = true;
        while (true) {

          MLIRPair pair;
          if (!mlirConsumer.lockAndDequeue(pair)) {

            // wait until more mlir modules available in the queue for
            // compilation or mlir code generation is finished
            mlirConsumer.waitUntilAvailableOr(
                [this] { return isMLIRCodeGenFinished(); });

            if (isMLIRCodeGenFinished()) {
              if (getMLIRCodeGenStatus() == Success) {
                setMLIRCompilationStatus(Success);
              } else {
                status = false;
                setMLIRCompilationStatus(Fail);
              }
              break;
            } else
              continue; // not finished; check again
          }

          auto &[futureMLIRMod, mlirContext] = pair;

          assert(futureMLIRMod.valid());

          if (auto llvmModule = convertToLLVMIR(std::move(futureMLIRMod))) {

            if (!SaveDebugSymbols) {
              runStripSymbolsPass(*llvmModule);
            }

            std::promise<std::unique_ptr<llvm::Module>> llvmModulePromise;
            llvmModulePromise.set_value(std::move(llvmModule));
            auto futureLLVMModule = llvmModulePromise.get_future();
            llvmProducer.lockAndEnqueue(std::move(futureLLVMModule));
          } else {
            status = false;
            setMLIRCompilationStatus(Fail);
            break;
          }
        }

        return status;
      });

  return futureStatus;
}

std::future<std::unique_ptr<llvm::Module>>
Driver::PerformModuleLinking(FuturePtr<llvm::Module> futureModule,
                             LLVMQueueConsumer &llvmConsumer) {
  assert(futureModule.valid());
  auto futureLinkedModule =
      std::async(std::launch::async,
                 [this, futureModule = std::move(futureModule),
                  &llvmConsumer]() mutable -> std::unique_ptr<llvm::Module> {
                   auto module = futureModule.get();
                   llvm::DataLayout dataLayout(module.get());
                   llvm::Linker linker(*module);

                   while (true) {
                     FuturePtr<llvm::Module> futureLLVMModule;
                     if (!llvmConsumer.lockAndDequeue(futureLLVMModule)) {

                       // wait until more modules available in the queue for
                       // linking or mlir compilation is finished
                       llvmConsumer.waitUntilAvailableOr(
                           [this] { return isMLIRCompilationFinished(); });

                       if (isMLIRCompilationFinished()) {
                         if (getMLIRCompilationStatus() == Success) {
                           setLLVMLinkingStatus(Success);
                         } else {
                           setLLVMLinkingStatus(Fail);
                           return nullptr;
                         }
                         break;
                       } else
                         continue;
                     }

                     auto toBeLinked = futureLLVMModule.get();
                     toBeLinked->setDataLayout(dataLayout);
                     toBeLinked->setTargetTriple(module->getTargetTriple());
                     linker.linkInModule(std::move(toBeLinked));
                   }

                   return module;
                 });

  return futureLinkedModule;
}

static SmallVector<std::string>
PerformRefactoring(const std::vector<std::string> &sources,
                   DiagnosticsEngine &Diags) {
  SmallVector<std::string> refactoredFiles;

  // Refactoring the original files
  for (const auto &source : sources) {
    clang::Rewriter rewriter;
    auto frontendAction =
        clang::tuner::newRewriterFrontendActionFactory(rewriter, Diags);
    clang::tooling::ClangTool Tool(OptionsParser->getCompilations(), {source});
    if (Tool.run(frontendAction.get()) != 0) {
      return {};
    }

    const auto *RewriteBuf =
        rewriter.getRewriteBufferFor(rewriter.getSourceMgr().getMainFileID());

    if (!RewriteBuf) {
      return {};
    }

    SmallString<256> refactoredFileCpp;
    int refactoredFileFD;
    llvm::sys::fs::createUniqueFile(source + "_refactored.cpp",
                                    refactoredFileFD, refactoredFileCpp);

    std::ofstream out(refactoredFileCpp.c_str());
    out << std::string(RewriteBuf->begin(), RewriteBuf->end());
    out.close();

    refactoredFiles.push_back(refactoredFileCpp.c_str());
  }

  return refactoredFiles;
}

/// Creating the argv to be passed to clang
static const llvm::SmallVector<std::string>
CreateClangArgs(const std::vector<std::string> &sources, const char *Argv0,
                const char *refactoredFileCpp) {
  llvm::SmallVector<std::string> CLArgs;

  // Adding compilation database's flags to args
  auto &compilationDatabase = OptionsParser->getCompilations();
  for (const auto &it : compilationDatabase.getAllCompileCommands()) {
    // we might need to switch directories here
    for (auto &elm : it.CommandLine) {
      bool isSource = false;
      for (const auto &source : sources) {
        if (elm == source) {
          isSource = true;
        }
      }
      if (!isSource)
        CLArgs.push_back(elm.c_str());
    }
    CLArgs.push_back(it.Output.c_str());
  }

  if (CLArgs.empty())
    CLArgs.push_back(Argv0);

  for (const auto &it : sources) {
    CLArgs.push_back(it.c_str());
  }

  CLArgs.push_back("-fsyntax-only");

  return CLArgs;
}

/// Creates a const char * vector from a string vector
static std::vector<const char *>
getCStrVec(const llvm::SmallVector<std::string> &strVec) {
  std::vector<const char *> cStrVec(strVec.size());
  for (int i = 0; i < strVec.size(); i++) {
    cStrVec[i] = strVec[i].c_str();
  }
  return cStrVec;
}

std::future<std::unique_ptr<llvm::Module>> Driver::PerformClangCompilation() {
  auto futureClangCompilation =
      std::async(std::launch::async, [this]() -> std::unique_ptr<llvm::Module> {
        const auto &sources = OptionsParser->getSourcePathList();
        auto RefactoredFiles = PerformRefactoring(sources, diags);

        if (RefactoredFiles.empty()) {
          llvm::errs() << "Warning: Refactorer didn't produce any files\n";
        }

        setRefactoringStatus(Success);
        auto tmpArgs =
            CreateClangArgs(sources, argv0, RefactoredFiles[0].c_str());
        auto Args = getCStrVec(tmpArgs);

        auto [compilation, llvmIRAction] = PerformClangCompilationHelper(
            clangDriver, Args, diags, argv0, mainAddr, lockableLLVMContext);

        if (!compilation || !llvmIRAction) {
          setClangCompilationStatus(Fail);
          return nullptr;
        }

        auto module = llvmIRAction->takeModule();
        if (!module) {
          llvm::errs() << "clang compilation failed\n";
          setClangCompilationStatus(Fail);
          return nullptr;
        }

        if (!SaveDebugSymbols) {
          runStripSymbolsPass(*module);
        }

        setClangCompilationStatus(Success);
        return module;
      });

  return futureClangCompilation;
}

/// Creates a const char * vector from a string vector
llvm::SmallVector<const char *>
getCStrVec(llvm::SmallVector<std::string> &strVec) {
  llvm::SmallVector<const char *> cStrVec(strVec.size());
  for (int i = 0, end = strVec.size(); i < end; i++) {
    cStrVec[i] = strVec[i].c_str();
  }

  return cStrVec;
}

/// Creating the clang driver
static clang::driver::Driver
CreateClangDriver(DiagnosticsEngine &Diags, const char *Argv0, void *MainAddr) {
  const std::string TripleStr = llvm::sys::getProcessTriple();
  llvm::Triple T(TripleStr);
  //  ExitOnErr.setBanner("clang tuner");
  std::string Path = GetExecutablePath(Argv0, MainAddr);

  clang::driver::Driver TheDriver(Path, T.str(), Diags);
  TheDriver.setTitle("clang tuner");
  TheDriver.setCheckInputsExist(false);

  return TheDriver;
}

static DiagnosticsEngine CreateDiagnosticsEngine() {
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter *DiagClient =
      new TextDiagnosticPrinter(llvm::errs(), DiagOpts.get());
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  return DiagnosticsEngine(DiagID, DiagOpts.get(), DiagClient);
}

Driver::Driver(int argc, const char **argv, void *mainAddr)
    : mainAddr(mainAddr), argv0(argv[0]), diags(CreateDiagnosticsEngine()),
      clangDriver(CreateClangDriver(diags, argv[0], mainAddr)),
      lockableLLVMContext(), MLIRCodeGenStatus(Unfinished),
      MLIRCompilationStatus(Unfinished), RefactoringStatus(Unfinished),
      ClangCompilationStatus(Unfinished), LLVMLinkingStatus(Unfinished) {

  llvm::InitLLVM X(argc, argv);
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  // Initialize targets for clang module support.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();
}

Driver::~Driver() {
#ifdef DEBUG

#define toString(TOKEN) #TOKEN

  auto StatusToString = [](Status status) {
    switch (status) {
    case Status::Fail:
      return toString(Status::Fail);
    case Status::Unfinished:
      return toString(Status::Unfinished);
    case Status::Success:
      return toString(Status::Success);
    }
  };

  llvm::errs() << "=== Status of Driver object upon destruction ===\n";
  llvm::errs() << toString(getRefactoringStatus()) << ": "
               << StatusToString(getRefactoringStatus()) << "\n";
  llvm::errs() << toString(getClangCompilationStatus()) << ": "
               << StatusToString(getClangCompilationStatus()) << "\n";
  llvm::errs() << toString(getMLIRCodeGenStatus()) << ": "
               << StatusToString(getMLIRCodeGenStatus()) << "\n";
  llvm::errs() << toString(getMLIRCompilationStatus()) << ": "
               << StatusToString(getMLIRCompilationStatus()) << "\n";
  llvm::errs() << toString(getLLVMLinkingStatus()) << ": "
               << StatusToString(getLLVMLinkingStatus()) << "\n";

#endif
}

} // namespace clang::tuner
