#include <memory>
#include <iostream>
extern "C" void __forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_matrix_vector_multiply_mv_static_array_cpp_30_3(float*, float*, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, float*, float*, std::size_t, std::size_t, std::size_t, float*, float*, std::size_t, std::size_t, std::size_t);
#include "../utils.hpp"

/// These parameters can be defined with -D flag to the compiler

#ifndef N
#define N 4
#endif

#ifndef M
#define M 4
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
  // void *ap = a;
  // void *bp = b;
  // void *cp = c;
  // std::size_t asize = M * N * sizeof(TYPE);
  // std::size_t bsize = N * sizeof(TYPE);
  // std::size_t csize = M * sizeof(TYPE);
  // float *aalign = (float *)std::align(alignof(decltype(a)), sizeof(decltype(a)), ap, asize);
  // float *balign = (float *)std::align(alignof(decltype(b)), sizeof(decltype(b)), bp, bsize);
  // float *calign = (float *)std::align(alignof(decltype(c)), sizeof(decltype(c)), cp, csize);
  __forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_matrix_vector_multiply_mv_static_array_cpp_30_3((TYPE *)a, (TYPE *)a, 0, 4, 4, 1, 1, (TYPE *)b, (TYPE *)b, 0, 4, 1, (TYPE *)c, (TYPE *)c, 0, 4, 1);

}

template<typename T, size_t m, size_t n>
bool verify(T a[m][n], T b[n], T res[m]) {
  std::cerr << "M: " << m << ",N: " << n << "\n";
  T c[m];
  for (size_t i = 0; i < m; i++) {
    c[i] = 0;
  }

  std::cout << "answer\n";
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      c[i] += a[i][j] * b[j];
    }
    std::cout << c[i] << "\n";
  }
  std::cout << "\nres\n";
  for (size_t i = 0; i < m; i++) {
    std::cout << res[i] << "\n";
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

  // initializeRandom_2D<TYPE, M, N>(a);
  // initializeRandom_1D<TYPE, N>(b);
  initialize_2D<TYPE, M, N>(a, 1);
  initialize_1D<TYPE, N>(b, 2);
  initialize_1D<TYPE, M>(c, 0);

  mat_vec_mult(a, b, c);

  if (!verify<TYPE, M, N>(a, b, c))
    return 1;

  return 0;
}