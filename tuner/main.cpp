//
// Created by parsa on 4/24/21.
//

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
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include <fstream>
#include <iostream>

#include "CommandLineOpts.h"

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

clang::tooling::CommonOptionsParser *OptionsParser;

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

// static clang::tooling::CommonOptionsParser &OptionsParser;
// static llvm::Expected<clang::tooling::CommonOptionsParser> ExpectedParser;

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

  //  clang::tuner::Driver driver;
  //
  //  auto x = driver.collectTranslationUnits()
  //  return 1;

  // This just needs to be some symbol in the binary; C++ doesn't
  // allow taking the address of ::main however.
  void *MainAddr = (void *)GetExecutablePath;
  std::string Path = GetExecutablePath(argv[0], MainAddr);
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter *DiagClient =
      new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagClient);
  clang::Rewriter rewriter;
  clang::tuner::Modules modules;
  mlir::MLIRContext context;

  // Refactoring the original file
  {
    auto fronendAction = clang::tuner::newFindAttrStmtsFrontendActionFactory(
        context, rewriter, Diags, modules,
        clang::tuner::FindAttrStmtsVisitor::VisitorTy::Refactorer);
    clang::tooling::ClangTool Tool(OptionsParser->getCompilations(),
                                   OptionsParser->getSourcePathList());
    if (Tool.run(fronendAction.get()) != 0) {
      return 1;
    }
  }

  const auto *RewriteBuf =
      rewriter.getRewriteBufferFor(rewriter.getSourceMgr().getMainFileID());

  if (!RewriteBuf) {
    llvm::errs() << "Nothing to tune! compile with clang!\n";
    return 1;
  }

  const auto *sources = &OptionsParser->getSourcePathList();

  SmallString<32> refactoredFileCpp;
  int refactoredFileFD;
  llvm::sys::fs::createUniqueFile(sources->at(0) + "_refactored.cpp",
                                  refactoredFileFD, refactoredFileCpp);

  std::ofstream out(refactoredFileCpp.c_str());
  out << std::string(RewriteBuf->begin(), RewriteBuf->end());
  out.close();

  const std::string &bitCodeFileName = sources->at(0) + ".ll";

  const std::string TripleStr = llvm::sys::getProcessTriple();
  llvm::Triple T(TripleStr);

  ExitOnErr.setBanner("clang tuner");

  Driver TheDriver(Path, T.str(), Diags);
  TheDriver.setTitle("clang tuner");
  TheDriver.setCheckInputsExist(false);

  llvm::SmallVector<std::string> CLArgs;
  auto &compilationDatabase = OptionsParser->getCompilations();

  //  CLArgs.push_back(argv[0]);
  for (const auto &it : compilationDatabase.getAllCompileCommands()) {
    // we might need to switch directories here

    for (auto &elm : it.CommandLine) {
      bool isSource = false;
      for (const auto &source : *sources) {
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
    CLArgs.push_back(argv[0]);

  // sources is const we need to modify it to pass in temporary files created.
  std::vector<std::string> *modifiableSources =
      const_cast<std::vector<std::string> *>(sources);

  modifiableSources->pop_back();
  modifiableSources->push_back(refactoredFileCpp.c_str());

  for (const auto &it : OptionsParser->getSourcePathList()) {
    CLArgs.push_back(it.c_str());
  }

  CLArgs.push_back("-fsyntax-only");

  llvm::SmallVector<const char *> Args(CLArgs.size());
  for (int i = 0; i < CLArgs.size(); i++) {
    Args[i] = CLArgs[i].c_str();
  }

  //  for (const auto &it: Args)
  //    llvm::errs() << it << "\n";

  std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(Args));
  if (!C)
    return 1;

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

  if (Clang.getHeaderSearchOpts().UseBuiltinIncludes &&
      Clang.getHeaderSearchOpts().ResourceDir.empty())
    Clang.getHeaderSearchOpts().ResourceDir =
        CompilerInvocation::GetResourcesPath(argv[0], MainAddr);

  // Create and execute the frontend to generate an LLVM bitcode module.
  std::unique_ptr<CodeGenAction> llvmIRAction(new EmitLLVMOnlyAction());
  if (!Clang.ExecuteAction(*llvmIRAction)) {
    return 1;
  }

  auto module = llvmIRAction->takeModule();
  llvm::DataLayout dataLayout(module.get());
  {
    auto fronendAction = clang::tuner::newFindAttrStmtsFrontendActionFactory(
        context, rewriter, Diags, modules,
        clang::tuner::FindAttrStmtsVisitor::VisitorTy::MLIRCodeGenerator);

    clang::tooling::ClangTool Tool(OptionsParser->getCompilations(),
                                   OptionsParser->getSourcePathList());
    if (Tool.run(fronendAction.get()) != 0) {
      return 1;
    }
  }

  llvm::Linker linker(*module);
  for (auto &it : modules) {
    auto &toBeLinked = it.getValue().second;

//    toBeLinked->setModuleFlag
//        (llvm::Module::ModFlagBehavior::ModFlagBehaviorFirstVal, "Debug Info "
//                                                                 "Version", )
    toBeLinked->setDataLayout(dataLayout);
    toBeLinked->setTargetTriple(module->getTargetTriple());
    linker.linkInModule(std::move(toBeLinked));
  }

  llvm::errs() << bitCodeFileName << "\n";
  std::error_code bitCodeWriterError;
  llvm::raw_fd_ostream bitCodeOStream(bitCodeFileName, bitCodeWriterError,
                                      llvm::sys::fs::F_None);
  bitCodeOStream << *module;
  bitCodeOStream.flush();

  // clean up the files
  if (!SaveRefactoredSourceFile.getValue())
    llvm::FileRemover refactoredFileRemover(refactoredFileCpp.c_str());

  return 0;
}