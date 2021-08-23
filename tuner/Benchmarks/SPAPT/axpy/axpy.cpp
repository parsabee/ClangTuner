#include "../utils.hpp"

/// These parameters can be defined with -D flag to the compiler

#ifndef N
#define N 256
#endif

#ifndef TYPE
#define TYPE float
#endif

/// generic axpy
/// 'a' mult 'x' plus 'y'. The result is stored in 'c'
///=============================================================================

/// FIXME Currently templated declarations are not implemented in the MLIRCodeGenerator,
/// this will be a generic operation in the future
void axpy(TYPE a, TYPE x[N], TYPE y[N], TYPE c[N]) {

  [[parallel_for::mlir_opt("--convert-scf-to-openmp", "--convert-openmp-to-llvm")]]
  for (size_t i = 0; i < N; i++) {
      c[i] = a * x[i] + y[i];
  }
}

template<typename T, size_t n>
bool verify(T a, T x[n], T y[n], T res[n]) {
  T c[n];
  for (size_t i = 0; i < n; i++) {
      c[i] = a * x[i] + y[i];
  }
  for (size_t i = 0; i < n; i++) {
    if (c[i] != res[i]) {
      return false;
    }
  }
  return true;
}

int main() {
  TYPE x[N];
  TYPE y[N];
  TYPE c[N];
  TYPE random = (TYPE)std::rand();
  initializeRandom_1D<TYPE, N>(x);
  initializeRandom_1D<TYPE, N>(y);

  axpy(random, x, y, c);

  if (!verify<TYPE, N>(random, x, y, c))
    return 1;

  return 0;
}