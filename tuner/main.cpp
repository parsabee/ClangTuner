//
// Created by parsa on 4/24/21.
//

#include "FindTuneAttr.h"
#include "FindAttrStmts.h"

#include "clang/Tooling/Tooling.h"
#include <fstream>
#include <iostream>


int main(int argc, char **argv) {
  if (argc == 2) {
    std::ifstream file(argv[1]);
    if (file) {
      std::string fileAsString((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());
      mlir::MLIRContext context;
      clang::tooling::runToolOnCode(std::make_unique<clang::tuner::FindAttrStmts>(context),
          fileAsString);
    } else {
      std::cerr << "no such file " << argv[1] << "\n";
      std::exit(1);
    }
  }
}