//
// Created by parsa on 4/24/21.
//

#include "FindAttrStmts.h"
#include "FindTuneAttr.h"

#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  if (argc == 2) {
    std::ifstream file(argv[1]);
    if (!file) {
      std::cerr << "no such file " << argv[1] << "\n";
      std::exit(1);
    }
    std::string fileAsString((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
    mlir::MLIRContext context;
    clang::Rewriter rewriter;
    clang::tooling::runToolOnCode(
        std::make_unique<clang::tuner::FindAttrStmts>(context, rewriter),
        fileAsString);
    const auto *RewriteBuf =
        rewriter.getRewriteBufferFor(rewriter.getSourceMgr().getMainFileID());

    const std::string inputPath = "refactored_file.cpp";
    std::ofstream out(inputPath);
    out << std::string(RewriteBuf->begin(), RewriteBuf->end());
    out.close();

    // Path to the executable
    std::string outputPath = "getinmemory";

    // Path to clang (e.g. /usr/local/bin/clang)
    auto clangPath = llvm::sys::findProgramByName("clang");

    if (clangPath.getError()) {
      return clangPath.getError().value();
    }

    // Arguments to pass to the clang driver:
    //	clang getinmemory.c -lcurl -v
    std::vector<const char *> args;
    args.push_back(clangPath.get().c_str());
    args.push_back(inputPath.c_str());
    //      args.push_back("-l");
    //      args.push_back("curl");

    clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagID(
        new clang::DiagnosticIDs());
    clang::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagOptions(
        new clang::DiagnosticOptions());
    clang::TextDiagnosticPrinter diagClient(llvm::errs(), diagOptions.get());
    clang::DiagnosticsEngine diags(diagID, diagOptions);

    clang::driver::Driver TheDriver(args[0],
                                    llvm::sys::getDefaultTargetTriple(), diags);

    //      TheDriver.CCCIsCXX();

    // Create the set of actions to perform
    std::unique_ptr<clang::driver::Compilation> C(
        TheDriver.BuildCompilation(args));

    std::unique_ptr<clang::CompilerInvocation> CI(
        new clang::CompilerInvocation);
    clang::CompilerInvocation::CreateFromArgs(*CI, args, diags);

    clang::CompilerInstance clangCC;
    clangCC.setInvocation(std::move(CI));

    // Create and execute the frontend to generate an LLVM bitcode module.
    std::unique_ptr<clang::CodeGenAction> Act(new clang::EmitLLVMOnlyAction());
    if (!clangCC.ExecuteAction(*Act))
      return 1;

    // Carry out the actions
    int Res = 0;
    llvm::SmallVector<std::pair<int, const clang::driver::Command *>, 4>
        FailingCommands;
    if (C)
      Res = TheDriver.ExecuteCompilation(*C, FailingCommands);

    bool IsCrash = false;
    for (const auto &P : FailingCommands) {
      int CommandRes = P.first;
      const clang::driver::Command *FailingCommand = P.second;
      if (!Res) {
        Res = CommandRes;
      }
      IsCrash = CommandRes < 0 || CommandRes == 70;
      IsCrash |= CommandRes > 128;
      if (IsCrash) {
        TheDriver.generateCompilationDiagnostics(*C, *FailingCommand);
        break;
      }
    }
  }
}