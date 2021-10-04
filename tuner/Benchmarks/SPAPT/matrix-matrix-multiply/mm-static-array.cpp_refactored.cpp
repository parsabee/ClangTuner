#include <memory>
extern "C" void __forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_matrix_matrix_multiply_mm_static_array_cpp_26_3(float*, float*, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, float*, float*, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, float*, float*, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t);
#include "../utils.hpp"

#ifndef N
#define N 256
#endif

#ifndef M
#define M 256
#endif

#ifndef K
#define K 256
#endif

#ifndef TYPE
#define TYPE float
#endif


/// generic matrix-matrix multiply
/// 'a' and 'b' are the operands are results are stored in 'c'
/// it is assumed that 'c' is initialized to 0
/// it is also assumed that a is m * n and b is n * t
void matmult(TYPE a[M][N], TYPE b[N][K], TYPE c[M][K]) {
  __forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_matrix_matrix_multiply_mm_static_array_cpp_26_3(a, a, 0, 256, 256, 1, 1, b, b, 0, 256, 256, 1, 1, c, c, 0, 256, 256, 1, 1);

}

template<typename T, size_t m, size_t n, size_t k>
bool verify(T a[m][n], T b[n][k], T res[m][k]) {
  T c[m][k];
  for (size_t i = 0; i < m; i++) {
    for (size_t t = 0; t < k; t++) {
      c[i][t] = 0;
    }
  }
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      for (size_t t = 0; t < k; t++) {
        c[i][t] += a[i][j] * b[j][t];
      }
    }
  }
  for (size_t i = 0; i < m; i++) {
    for (size_t t = 0; t < k; t++) {
      if (c[i][t] != res[i][t]) {
        return false;
      }
    }
  }
  return true;
}

int main() {
  TYPE a[M][N];
  TYPE b[N][K];
  TYPE c[M][K];

  initializeRandom_2D<float, M, N>(a);
  initializeRandom_2D<float, N, K>(b);
  initialize_2D<float, M, K>(c, 0.0f);

  matmult(a, b, c);

  if (!verify<float, M, N, K>(a, b, c))
    return 1;

  return 0;
}