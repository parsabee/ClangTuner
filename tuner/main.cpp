//
// Created by parsa on 4/24/21.
//

/// Pipeline of the compiler:
/// i) Generate mlir code for the attributed statements with the
/// MLIRCodeGenerator. (optionally save to file with -m0)
/// ii) Perform additional optimizations with mlir-opt. (optionally save to
/// file with -m1)
/// iii) Lower to llvm ir dialect. (optionally save to file with -m2)
/// iv) Convert llvm dialect to llvm ir module. (optionally save to file with
/// -m3)
/// v) Refactor the original source code(extract the mlir attributed stmts
/// into functions). (optionally save to file with -rs)
/// vi) Compile the refactored code with clang down to llvm bit code.
/// vii) Link the two llvm modules modules with the llvm::Linker
/// viii) Write bitcode to file

#include "AttrForLoopFinder.h"
#include "AttrForLoopRefactorer.h"
#include "Driver.h"
#include "FindTuneAttr.h"

#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/SerializedDiagnosticPrinter.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/IPO/StripSymbols.h"

#include <fstream>
#include <iostream>
#include <tuple>

#include "CommandLineOpts.h"

using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;

std::string GetExecutablePath(const char *Argv0, void *MainAddr) {
  return llvm::sys::fs::getMainExecutable(Argv0, MainAddr);
}

llvm::ExitOnError ExitOnErr;

// Defining the command line options, these options are declared in
// CommandLineOpts.h
llvm::cl::OptionCategory ClangTuneCategory("clang-tune options");

llvm::cl::opt<bool>
    SaveInitialMLIRFile("m0",
                        llvm::cl::desc("Save the initial mlir code to "
                                       "file."),
                        llvm::cl::cat(ClangTuneCategory));

llvm::cl::opt<bool>
    SaveOptimizedMLIRFile("m1",
                          llvm::cl::desc("Save the optimized mlir code to "
                                         "file."),
                          llvm::cl::cat(ClangTuneCategory));

llvm::cl::opt<bool>
    SaveLLVMDialectMLIRFile("m2",
                            llvm::cl::desc("Save the llvm dialect mlir code to "
                                           "file."),
                            llvm::cl::cat(ClangTuneCategory));

llvm::cl::opt<bool>
    SaveLLVMModule("m3",
                   llvm::cl::desc("Save the final llvm module generated "
                                  "by lowering mlir code file."),
                   llvm::cl::cat(ClangTuneCategory));

llvm::cl::opt<bool>
    SaveRefactoredSourceFile("rs",
                             llvm::cl::desc("Save the refactored source file."),
                             llvm::cl::cat(ClangTuneCategory));

llvm::cl::opt<bool> SaveDebugSymbols(
    "dbg",
    llvm::cl::desc("Save source information for creating debug symbols."),
    llvm::cl::cat(ClangTuneCategory));

clang::tooling::CommonOptionsParser *OptionsParser;

// static clang::tooling::CommonOptionsParser &OptionsParser;
// static llvm::Expected<clang::tooling::CommonOptionsParser> ExpectedParser;

/// Creating a compilation instance from the args
std::tuple<std::unique_ptr<Compilation>, std::unique_ptr<CodeGenAction>>
PerformCompilation(Driver &TheDriver, llvm::SmallVector<const char *> &Args,
                   DiagnosticsEngine &Diags, const char *Argv0, void *MainAddr,
                   llvm::LLVMContext *llvmContext) {
  std::unique_ptr<Compilation> compilation(TheDriver.BuildCompilation(Args));
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
  std::unique_ptr<CodeGenAction> llvmIRAction(
      new EmitLLVMOnlyAction(llvmContext));
  if (!Clang.ExecuteAction(*llvmIRAction)) {
    return {nullptr, nullptr};
  }

  return {std::move(compilation), std::move(llvmIRAction)};
}

/// Creates a const char * vector from a string vector
llvm::SmallVector<const char *>
getCStrVec(llvm::SmallVector<std::string> &strVec) {

  llvm::SmallVector<const char *> cStrVec(strVec.size());
  for (int i = 0; i < strVec.size(); i++) {
    cStrVec[i] = strVec[i].c_str();
  }

  return std::move(cStrVec);
}

/// Creating the argv to be passed to clang
llvm::SmallVector<std::string>
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

  return std::move(CLArgs);
}

/// Creating the clang driver
Driver CreateClangDriver(DiagnosticsEngine &Diags, const char *Argv0,
                         void *MainAddr) {
  const std::string TripleStr = llvm::sys::getProcessTriple();
  llvm::Triple T(TripleStr);
  ExitOnErr.setBanner("clang tuner");
  std::string Path = GetExecutablePath(Argv0, MainAddr);

  Driver TheDriver(Path, T.str(), Diags);
  TheDriver.setTitle("clang tuner");
  TheDriver.setCheckInputsExist(false);

  return TheDriver;
}

SmallVector<std::string>
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
      llvm::errs() << "Nothing to tune! compile with clang!\n";
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

int main(int argc, const char **argv) {
  llvm::InitLLVM X(argc, argv);
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  // Initialize targets for clang module support.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  auto ExpectedParser = clang::tooling::CommonOptionsParser::create(
      argc, argv, ClangTuneCategory);

  if (!ExpectedParser) {
    // Fail gracefully for unsupported options.
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }
  OptionsParser = &ExpectedParser.get();

  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter *DiagClient =
      new TextDiagnosticPrinter(llvm::errs(), DiagOpts.get());
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  DiagnosticsEngine Diags(DiagID, DiagOpts.get(), DiagClient);

  const auto &sources = OptionsParser->getSourcePathList();

  auto RefactoredFiles = PerformRefactoring(sources, Diags);

  if (RefactoredFiles.empty()) {
    llvm::errs() << "Error: Refactorer didn't produce any files\n";
    return 1;
  }

  void *MainAddr = (void *)GetExecutablePath;
  auto TheDriver = CreateClangDriver(Diags, argv[0], MainAddr);

  auto CLArgs = CreateClangArgs(sources, argv[0], RefactoredFiles[0].c_str());
  auto Args = getCStrVec(CLArgs);

  llvm::LLVMContext llvmContext;
  auto [compilation, llvmIRAction] = PerformCompilation(
      TheDriver, Args, Diags, argv[0], MainAddr, &llvmContext);

  if (!compilation || !llvmIRAction)
    return 1;

  auto module = llvmIRAction->takeModule();
  if (!module) {
    llvm::errs() << "clang compilation failed\n";
    return 1;
  }

  if (!SaveDebugSymbols) {
    llvm::ModuleAnalysisManager llvmAM;
    llvm::StripSymbolsPass llvmPass;
    llvmPass.run(*module, llvmAM);
  }

  llvm::DataLayout dataLayout(module.get());
  clang::tuner::Modules modules;
  mlir::MLIRContext mlirContext;

  auto frontendAction = clang::tuner::newMLIRCodeGenFrontendActionFactory(
      mlirContext, llvmContext, modules, Diags);
  clang::tooling::ClangTool Tool(OptionsParser->getCompilations(),
                                 OptionsParser->getSourcePathList());
  if (Tool.run(frontendAction.get()) != 0) {
    return 1;
  }

  llvm::Linker linker(*module);
  for (auto &it : modules) {
    auto &toBeLinked = it.getValue();
    toBeLinked->setDataLayout(dataLayout);
    toBeLinked->setTargetTriple(module->getTargetTriple());
    linker.linkInModule(std::move(toBeLinked));
  }

  std::error_code bitCodeWriterError;
  llvm::raw_fd_ostream bitCodeOStream(sources.at(0) + ".ll",
                                      bitCodeWriterError,
                                      llvm::sys::fs::F_None);
  bitCodeOStream << *module;
  bitCodeOStream.flush();

  // clean up the files
  if (!SaveRefactoredSourceFile.getValue()) {
    for (auto &file : RefactoredFiles) {
      llvm::FileRemover refactoredFileRemover(file.c_str());
    }
  }

  return 0;
}