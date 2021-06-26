//
// Created by parsa on 4/24/21.
//

#include "FindAttrStmts.h"
#include "FindTuneAttr.h"
#include "ForLoopRefactorer.h"
#include "clang/Basic/DiagnosticFrontend.h"
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
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetRegistry.h"
#include <clang/Tooling/CommonOptionsParser.h>
#include <fstream>
#include <iostream>
#include <llvm/Support/TargetSelect.h>

using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;

bool getStdHeaders(SmallVector<std::string> &headers, bool addFlag = true) {
  int InputFD;
  SmallString<32> InputFile, OutputFile;
  llvm::sys::fs::createTemporaryFile("header-finder-input", "", InputFD,
                                     InputFile);
  llvm::sys::fs::createTemporaryFile("header-finder-output", "", OutputFile);
  llvm::FileRemover InputRemover(InputFile.c_str());
  llvm::FileRemover OutputRemover(OutputFile.c_str());
  Optional<StringRef> Redirects[] = {StringRef(""), StringRef("/dev/null"),
                                     StringRef(OutputFile)};
  StringRef Args[] = {"-E", "-v", "-x", "c++", "/dev/null", "-fsyntax-only"};
  auto pathOrErr = llvm::sys::findProgramByName("cc");
  if (!pathOrErr)
    return false;
  const std::string &path = *pathOrErr;
  int RunResult = llvm::sys::ExecuteAndWait(path, Args, None, Redirects);
  if (RunResult != 0)
    return false;

  auto OutputBuf = llvm::MemoryBuffer::getFile(OutputFile.c_str());
  if (!OutputBuf)
    return false;
  StringRef Output = OutputBuf.get()->getBuffer();

  SmallVector<StringRef, 64> Lines;
  Output.split(Lines, "\n");

  bool isSysHeader = false;
  for (auto l : Lines) {
    if (l.startswith("End of search list.")) {
      isSysHeader = false;
    }
    if (isSysHeader) {
      std::string includeHeader(l);
      if (addFlag)
        includeHeader.replace(includeHeader.begin(), includeHeader.begin() + 1,
                              "-I");
      else
        includeHeader.replace(includeHeader.begin(), includeHeader.begin() + 1,
                              "");
      std::size_t found = includeHeader.find(" ");
      if (found != std::string::npos)
        includeHeader.replace(includeHeader.begin() + found,
                              includeHeader.end(), "");

      headers.push_back(std::move(includeHeader));
    }
    if (l.startswith("#include <...> search starts here:")) {
      isSysHeader = true;
    }
  }

  return true;
}

std::string GetExecutablePath(const char *Argv0, void *MainAddr) {
  return llvm::sys::fs::getMainExecutable(Argv0, MainAddr);
}

bool setUpCompiler(CompilerInstance &Clang, Compilation *C,
                   DiagnosticsEngine &Diags, const char *argv0,
                   void *MainAddr) {
  const driver::JobList &Jobs = C->getJobs();
  if (Jobs.size() != 1 || !isa<driver::Command>(*Jobs.begin())) {
    SmallString<256> Msg;
    llvm::raw_svector_ostream OS(Msg);
    Jobs.Print(OS, "; ", true);
    Diags.Report(diag::err_fe_expected_compiler_job) << OS.str();
    return false;
  }

  const driver::Command &Cmd = cast<driver::Command>(*Jobs.begin());
  if (llvm::StringRef(Cmd.getCreator().getName()) != "clang") {
    Diags.Report(diag::err_fe_expected_clang_command);
    return false;
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
  Clang.setInvocation(std::move(CI));

  // Create the compilers actual diagnostics engine.
  Clang.createDiagnostics();
  if (!Clang.hasDiagnostics())
    return false;

  // Infer the builtin include path if unspecified.
  if (Clang.getHeaderSearchOpts().UseBuiltinIncludes &&
      Clang.getHeaderSearchOpts().ResourceDir.empty()) {
    Clang.getHeaderSearchOpts().ResourceDir =
        CompilerInvocation::GetResourcesPath(argv0, MainAddr);
  }

  return true;
}

llvm::ExitOnError ExitOnErr;

static llvm::cl::OptionCategory ClangTuneCategory("clang-tune options");

int main(int argc, const char **argv) {

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
  clang::tooling::CommonOptionsParser &OptionsParser = ExpectedParser.get();
  clang::tooling::ClangTool Tool(OptionsParser.getCompilations(),
                                 OptionsParser.getSourcePathList());

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

    ExitOnErr.setBanner("clang tuner");

    Driver TheDriver(Path, T.str(), Diags);
    TheDriver.setTitle("clang tuner");
    TheDriver.setCheckInputsExist(false);

  // FIXME: This is a hack to try to force the driver to do something we can
  // recognize. We need to extend the driver library to support this use model
  // (basically, exactly one input, and the operation mode is hard wired).
  //  std::ifstream file(argv[1]);
  //  if (!file) {
  //    llvm::errs() << "no such file " << argv[1] << "\n";
  //    std::exit(1);
  //  }
  //
  //  std::string fileAsString((std::istreambuf_iterator<char>(file)),
  //                           std::istreambuf_iterator<char>());

  // TODO: Maybe we can initialize one LLVMContext here and pass that to
  // CodeGen, instead of creating a new LLVMContext for each CodeGen instance

  clang::tuner::Modules modules;
  mlir::MLIRContext context;
  clang::Rewriter rewriter;
  auto fronendAction =
      clang::tuner::newFindAttrStmtsFrontendActionFactory(
          context, rewriter, Diags, modules);
  Tool.run(fronendAction.get());
  //  auto findAttrStmts = std::make_unique<clang::tuner::FindAttrStmts>(
  //      context, rewriter, Diags, modules);

  //  clang::tooling::runToolOnCode(std::move(findAttrStmts),
  //                                fileAsString);

    const auto *RewriteBuf =
        rewriter.getRewriteBufferFor(rewriter.getSourceMgr().getMainFileID());

    const std::string &refactoredFileCpp = "__tmp_refactored_file__.cpp";

    std::ofstream out(refactoredFileCpp.c_str());
    out << std::string(RewriteBuf->begin(), RewriteBuf->end());
    out.close();
    // this object will delete the file when it goes out of scope
//    llvm::FileRemover refactoredFileRemover(refactoredFileCpp.c_str());

    SmallVector<std::string> headers;
    if (!getStdHeaders(headers))
      return 1;

    SmallVector<const char *> Args;
    Args.push_back(argv[0]);
    for (const auto &it : headers) {
      //    Args.push_back("-I");
      Args.push_back(it.c_str());
    }
    Args.push_back(refactoredFileCpp.c_str());
    Args.push_back("-fsyntax-only"); // this is necessary otherwise the next line
                                     // will yell that there is no compiler jobs

    std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(Args));
    if (!C)
      return 1;

    // Create a compiler instance to handle the actual work.
    CompilerInstance Clang;
    if (!setUpCompiler(Clang, C.get(), Diags, argv[0], MainAddr)) {
      llvm::errs() << "compilation failed\n";
      return 1;
    }

    // Create and execute the frontend to generate an LLVM bitcode module.
    std::unique_ptr<CodeGenAction> llvmIRAction(new EmitLLVMOnlyAction());
    if (!Clang.ExecuteAction(*llvmIRAction)) {
      return 1;
    }

    auto module = llvmIRAction->takeModule();
    llvm::DataLayout dataLayout(module.get());
    llvm::Linker linker(*module);

    for (auto &it : modules) {
      auto &toBeLinked = it.getValue().second;
      toBeLinked->setDataLayout(dataLayout);
      linker.linkInModule(std::move(toBeLinked));
    }

    const std::string &bitCodeFileName = "clang-tuner-llvm-ir-output.ll";
  //  llvm::FileRemover bitCodeFileRemover(bitCodeFileName.c_str());
    std::error_code bitCodeWriterError;
    llvm::raw_fd_ostream bitCodeOStream(bitCodeFileName, bitCodeWriterError,
                                        llvm::sys::fs::F_None);
    bitCodeOStream << *module;
  //  WriteBitcodeToFile(*module, bitCodeOStream);
    bitCodeOStream.flush();
}