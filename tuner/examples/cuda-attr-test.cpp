#define N 256
#include <stdio.h>
#include <stdarg.h>

void f() {
  int a[N];
  int b[N];
  int c[N] = {0};
  for (int i = 0; i < N; i++) {
    a[i] = 1;
    b[i] = 2;
  }

  [[parallel_for::mlir_opt("--parallel-loop-tiling",
                           "--convert-scf-to-openmp",
                           "--convert-openmp-to-llvm")]]
//  [[parallel_for::mlir_opt("--enable-loop-simplifycfg-term-folding")]]
  for(int j = 0; j < N; j+=1) {
    for (int i = 0; i < N; i += 1) {
      c[i] = a[i] + b[i];
    }
  }

  for (int i = 0; i < N; i+= 1) {
    printf("%d\n", c[i]);
  }
}

int main(int argc, char **argv) {
  f();
}
