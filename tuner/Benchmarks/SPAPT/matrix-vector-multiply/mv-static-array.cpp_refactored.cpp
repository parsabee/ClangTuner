#include <memory>
extern "C" void __forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_matrix_vector_multiply_mv_static_array_cpp_30_3(float*, float*, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, float*, float*, std::size_t, std::size_t, std::size_t, float*, float*, std::size_t, std::size_t, std::size_t);
#include "../utils.hpp"

/// These parameters can be defined with -D flag to the compiler

#ifndef N
#define N 256
#endif

#ifndef M
#define M 256
#endif

#ifndef TYPE
#define TYPE float
#endif

/// generic matrix-vector multiply
/// 'a' and 'b' are the operands the result is stored in 'c'
/// it is assumed that 'c' is initialized to all 0s
///=============================================================================

/// FIXME Currently templated declarations are not implemented in the MLIRCodeGenerator,
/// this will be a generic operation in the future
void mat_vec_mult(TYPE a[M][N], TYPE b[N], TYPE c[M]) {
  __forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_matrix_vector_multiply_mv_static_array_cpp_30_3(a, a, 0, 256, 256, 1, 1, b, b, 0, 256, 1, c, c, 0, 256, 1);

}

template<typename T, size_t m, size_t n>
bool verify(T a[m][n], T b[n], T res[m]) {
  T c[m];
  for (size_t i = 0; i < m; i++) {
    c[i] = 0;
  }
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      c[i] += a[i][j] * b[j];
    }
  }
  for (size_t i = 0; i < m; i++) {
    if (c[i] != res[i]) {
      return false;
    }
  }
  return true;
}

int main() {
  TYPE a[M][N];
  TYPE b[N];
  TYPE c[M];

  initializeRandom_2D<TYPE, M, N>(a);
  initializeRandom_1D<TYPE, N>(b);
  initialize_1D<TYPE, M>(c, 0);

  mat_vec_mult(a, b, c);

  if (!verify<TYPE, M, N>(a, b, c))
    return 1;

  return 0;
}