#include <memory>
extern "C" void __forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_22_3(float, float*, float*, std::size_t, std::size_t, std::size_t, float*, float*, std::size_t, std::size_t, std::size_t, float*, float*, std::size_t, std::size_t, std::size_t);
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

  __forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_22_3(a, c, c, 0, 256, 1, x, x, 0, 256, 1, y, y, 0, 256, 1);

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