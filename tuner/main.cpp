//
// Created by parsa on 4/24/21.
//

#include "FindAttrStmts.h"
#include "FindTuneAttr.h"
#include "ForLoopRefactorer.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Frontend/ChainedDiagnosticConsumer.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/SerializedDiagnosticPrinter.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#include <clang/Basic/DiagnosticFrontend.h>
#include <fstream>
#include <iostream>

std::string GetExecutablePath(const char *Argv0, void *MainAddr) {
  return llvm::sys::fs::getMainExecutable(Argv0, MainAddr);
}

using namespace clang;
using namespace clang::driver;

llvm::ExitOnError ExitOnErr;

int main(int argc, char **argv) {
  //  if (argc != 2)
  //    return 1;

  // This just needs to be some symbol in the binary; C++ doesn't
  // allow taking the address of ::main however.
  void *MainAddr = (void *)(intptr_t)GetExecutablePath;
  std::string Path = GetExecutablePath(argv[0], MainAddr);
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter *DiagClient =
      new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagClient);

  const std::string TripleStr = llvm::sys::getProcessTriple();
  llvm::Triple T(TripleStr);

  // Use ELF on Windows-32 and MingW for now.
#ifndef CLANG_INTERPRETER_COFF_FORMAT
  if (T.isOSBinFormatCOFF())
    T.setObjectFormat(llvm::Triple::ELF);
#endif

  ExitOnErr.setBanner("clang tuner");

  Driver TheDriver(Path, T.str(), Diags);
  TheDriver.setTitle("clang tuner");
  TheDriver.setCheckInputsExist(false);

  // FIXME: This is a hack to try to force the driver to do something we can
  // recognize. We need to extend the driver library to support this use model
  // (basically, exactly one input, and the operation mode is hard wired).
  std::ifstream file(argv[1]);
  if (!file) {
    llvm::errs() << "no such file " << argv[1] << "\n";
    std::exit(1);
  }

  std::string fileAsString((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
  mlir::MLIRContext context;
  clang::Rewriter rewriter;
  clang::tooling::runToolOnCode(
      std::make_unique<clang::tuner::FindAttrStmts>(context, rewriter, Diags),
      fileAsString);

  const auto *RewriteBuf =
      rewriter.getRewriteBufferFor(rewriter.getSourceMgr().getMainFileID());

  const std::string inputPath = "refactored_file.cpp";
  std::ofstream out(inputPath);
  out << std::string(RewriteBuf->begin(), RewriteBuf->end());
  out.close();

  SmallVector<const char *> Args;
  Args.push_back("clang++");
  Args.push_back(inputPath.c_str());
  Args.push_back("-fsyntax-only");
  std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(Args));
  if (!C)
    return 0;

  // FIXME: This is copied from ASTUnit.cpp; simplify and eliminate.

  // We expect to get back exactly one command job, if we didn't something
  // failed. Extract that job from the compilation.
  const driver::JobList &Jobs = C->getJobs();
  if (Jobs.size() != 1 || !isa<driver::Command>(*Jobs.begin())) {
    SmallString<256> Msg;
    llvm::raw_svector_ostream OS(Msg);
    Jobs.Print(OS, "; ", true);
    Diags.Report(diag::err_fe_expected_compiler_job) << OS.str();
    return 1;
  }

  const driver::Command &Cmd = cast<driver::Command>(*Jobs.begin());
  if (llvm::StringRef(Cmd.getCreator().getName()) != "clang") {
    Diags.Report(diag::err_fe_expected_clang_command);
    return 1;
  }

  // Initialize a compiler invocation object from the clang (-cc1) arguments.
  const llvm::opt::ArgStringList &CCArgs = Cmd.getArguments();
  std::unique_ptr<CompilerInvocation> CI(new CompilerInvocation);
  CompilerInvocation::CreateFromArgs(*CI, CCArgs, Diags);

  // Show the invocation, with -v.
  if (CI->getHeaderSearchOpts().Verbose) {
    llvm::errs() << "clang invocation:\n";
    Jobs.Print(llvm::errs(), "\n", true);
    llvm::errs() << "\n";
  }

  // FIXME: This is copied from cc1_main.cpp; simplify and eliminate.

  // Create a compiler instance to handle the actual work.
  CompilerInstance Clang;
  Clang.setInvocation(std::move(CI));

  // Create the compilers actual diagnostics engine.
  Clang.createDiagnostics();
  if (!Clang.hasDiagnostics())
    return 1;

  // Infer the builtin include path if unspecified.
  if (Clang.getHeaderSearchOpts().UseBuiltinIncludes &&
      Clang.getHeaderSearchOpts().ResourceDir.empty())
    Clang.getHeaderSearchOpts().ResourceDir =
        CompilerInvocation::GetResourcesPath(argv[0], MainAddr);

  // Create and execute the frontend to generate an LLVM bitcode module.
  std::unique_ptr<CodeGenAction> Act(new EmitLLVMOnlyAction());
  if (!Clang.ExecuteAction(*Act))
    return 1;

  auto module = Act->takeModule();
  module->dump();

  //
  //  if (argc == 2) {
  //    std::ifstream file(argv[1]);
  //    if (!file) {
  //      llvm::errs() << "no such file " << argv[1] << "\n";
  //      std::exit(1);
  //    }
  //
  //    clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagID(
  //        new clang::DiagnosticIDs());
  //    clang::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagOptions(
  //        new clang::DiagnosticOptions());
  //    clang::TextDiagnosticPrinter diagClient(llvm::errs(),
  //    diagOptions.get()); clang::DiagnosticsEngine diags(diagID, diagOptions);
  //    diags.setClient(&diagClient);
  //
  //    std::string fileAsString((std::istreambuf_iterator<char>(file)),
  //                             std::istreambuf_iterator<char>());
  //    mlir::MLIRContext context;
  //    clang::Rewriter rewriter;
  //    clang::tooling::runToolOnCode(
  //        std::make_unique<clang::tuner::FindAttrStmts>(context, rewriter,
  //        diags), fileAsString);
  //
  //    const auto *RewriteBuf =
  //        rewriter.getRewriteBufferFor(rewriter.getSourceMgr().getMainFileID());
  //
  //    const std::string inputPath = "./refactored_file.cpp";
  //    std::ofstream out(inputPath);
  //    out << std::string(RewriteBuf->begin(), RewriteBuf->end());
  //    out.close();
  //
  //    // Path to clang (e.g. /usr/local/bin/clang)
  //    auto clangPath = llvm::sys::findProgramByName("clang++");
  //
  //    if (clangPath.getError()) {
  //      return clangPath.getError().value();
  //    }
  //
  //    std::vector<const char *> args;
  //    args.push_back(
  //        std::string("-triple=" +
  //        llvm::sys::getDefaultTargetTriple()).c_str());
  //    //    args.push_back(clangPath.get().c_str());
  //    //    args.push_back(inputPath.c_str());
  //
  //    clang::driver::Driver TheDriver(clangPath.get().c_str(),
  //                                    llvm::sys::getDefaultTargetTriple(),
  //                                    diags);
  //
  //    clang::CompilerInstance compilerInstance;
  //    auto &compilerInvocation = compilerInstance.getInvocation();
  //
  //    clang::CompilerInvocation::CreateFromArgs(compilerInvocation, args,
  //    diags,
  //                                              clangPath.get().c_str());
  //
  //    void *MainAddr = (void*) (intptr_t) sym;
  //    if (compilerInvocation.getHeaderSearchOpts().UseBuiltinIncludes &&
  //        compilerInvocation.getHeaderSearchOpts().ResourceDir.empty())
  //      compilerInvocation.getHeaderSearchOpts().ResourceDir =
  //          clang::CompilerInvocation::GetResourcesPath(argv[0], MainAddr);
  //
  //    auto *languageOptions = compilerInvocation.getLangOpts();
  //    auto &preprocessorOptions = compilerInvocation.getPreprocessorOpts();
  //    auto &targetOptions = compilerInvocation.getTargetOpts();
  //    auto &frontEndOptions = compilerInvocation.getFrontendOpts();
  //#ifdef DEBUG
  //    frontEndOptions.ShowStats = true;
  //#endif
  ////    auto &headerSearchOpts = compilerInvocation.getHeaderSearchOpts();
  //    //    headerSearchOpts.
  //
  //#ifdef DEBUG
  //    headerSearchOptions.Verbose = true;
  //#endif
  //    auto &codeGenOptions = compilerInvocation.getCodeGenOpts();
  //    frontEndOptions.Inputs.clear();
  //    frontEndOptions.Inputs.push_back(clang::FrontendInputFile(
  //        inputPath, clang::InputKind(clang::Language::CXX)));
  //
  //    targetOptions.Triple = llvm::sys::getDefaultTargetTriple();
  //    compilerInstance.createDiagnostics(&diagClient, false);
  //    if (!compilerInstance.hasDiagnostics())
  //      return 1;
  //
  //    // Create and execute the frontend to generate an LLVM bitcode module.
  //    std::unique_ptr<clang::CodeGenAction> Act(new
  //    clang::EmitLLVMOnlyAction());
  //
  //    if (!compilerInstance.ExecuteAction(*Act))
  //      return 1;
  //
  //    auto module = Act->takeModule();
  //    module->dump();
  //    //    // Carry out the actions
  //    //    int Res = 0;
  //    //    llvm::SmallVector<std::pair<int, const clang::driver::Command *>,
  //    4>
  //    //        FailingCommands;
  //    //    if (C)
  //    //      Res = TheDriver.ExecuteCompilation(*C, FailingCommands);
  //    //
  //    //    bool IsCrash = false;
  //    //    for (const auto &P : FailingCommands) {
  //    //      int CommandRes = P.first;
  //    //      const clang::driver::Command *FailingCommand = P.second;
  //    //      if (!Res) {
  //    //        Res = CommandRes;
  //    //      }
  //    //      IsCrash = CommandRes < 0 || CommandRes == 70;
  //    //      IsCrash |= CommandRes > 128;
  //    //      if (IsCrash) {
  //    //        TheDriver.generateCompilationDiagnostics(*C, *FailingCommand);
  //    //        break;
  //    //      }
  //    //    }
  //  } else {
  //    llvm::errs() << "No file\n";
  //  }
}