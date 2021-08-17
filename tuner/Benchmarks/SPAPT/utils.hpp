#ifndef TUNER_BENCHMARKS_SETUPANDTEARDOWN_HPP
#define TUNER_BENCHMARKS_SETUPANDTEARDOWN_HPP

#include <random>

/// Dynamic arrays
///=========================================================================
template <typename T> void initialize_1D(T *array, T initVal, size_t size) {
  for (size_t i = 0; i < size; i++) {
    array[i] = initVal;
  }
}

template <typename T>
void initialize_2D(T **array, T initVal, size_t size1stD, size_t size2ndD) {
  for (size_t i = 0; i < size1stD; i++) {
    for (size_t j = 0; j < size2ndD; j++) {
      array[i][j] = initVal;
    }
  }
}

template <typename T>
void initialize_3D(T ***array, T initVal, size_t size1stD, size_t size2ndD,
                   size_t size3rdD) {
  for (size_t i = 0; i < size1stD; i++) {
    for (size_t j = 0; j < size2ndD; j++) {
      for (size_t t = 0; t < size3rdD; t++) {
        array[i][j][t] = initVal;
      }
    }
  }
}

template <typename T> void initializeRandom_1D(T *array, size_t size) {
  std::srand(
      std::time(nullptr)); // use current time as seed for random generator
  for (size_t i = 0; i < size; i++) {
    array[i] = std::rand();
  }
}

template <typename T>
void initializeRandom_2D(T **array, size_t size1stD, size_t size2ndD) {
  std::srand(
      std::time(nullptr)); // use current time as seed for random generator
  for (size_t i = 0; i < size1stD; i++) {
    for (size_t j = 0; j < size2ndD; j++) {
      array[i][j] = std::rand();
    }
  }
}

template <typename T>
void initializeRandom_3D(T ***array, size_t size1stD, size_t size2ndD,
                         size_t size3rdD) {
  std::srand(
      std::time(nullptr)); // use current time as seed for random generator
  for (size_t i = 0; i < size1stD; i++) {
    for (size_t j = 0; j < size2ndD; j++) {
      for (size_t t = 0; t < size3rdD; t++) {
        array[i][j][t] = std::rand();
      }
    }
  }
}

/// Static arrays
///=========================================================================
template <typename T, size_t size>
void initialize_1D(T array[size], T initVal) {
  for (size_t i = 0; i < size; i++) {
    array[i] = initVal;
  }
}

template <typename T, size_t size1stD, size_t size2ndD>
void initialize_2D(T array[size1stD][size2ndD], T initVal) {
  for (size_t i = 0; i < size1stD; i++) {
    for (size_t j = 0; j < size2ndD; j++) {
      array[i][j] = initVal;
    }
  }
}

template <typename T, size_t size1stD, size_t size2ndD, size_t size3rdD>
void initialize_3D(T array[size1stD][size2ndD][size3rdD], T initVal) {
  for (size_t i = 0; i < size1stD; i++) {
    for (size_t j = 0; j < size2ndD; j++) {
      for (size_t t = 0; t < size3rdD; t++) {
        array[i][j][t] = initVal;
      }
    }
  }
}

template <typename T, size_t size> void initializeRandom_1D(T array[size]) {
  std::srand(
      std::time(nullptr)); // use current time as seed for random generator
  for (size_t i = 0; i < size; i++) {
    array[i] = std::rand();
  }
}

template <typename T, size_t size1stD, size_t size2ndD>
void initializeRandom_2D(T array[size1stD][size2ndD]) {
  std::srand(
      std::time(nullptr)); // use current time as seed for random generator
  for (size_t i = 0; i < size1stD; i++) {
    for (size_t j = 0; j < size2ndD; j++) {
      array[i][j] = std::rand();
    }
  }
}

template <typename T, size_t size1stD, size_t size2ndD, size_t size3rdD>
void initializeRandom_3D(T array[size1stD][size2ndD][size3rdD]) {
  std::srand(
      std::time(nullptr)); // use current time as seed for random generator
  for (size_t i = 0; i < size1stD; i++) {
    for (size_t j = 0; j < size2ndD; j++) {
      for (size_t t = 0; t < size3rdD; t++) {
        array[i][j][t] = std::rand();
      }
    }
  }
}

#endif // TUNER_BENCHMARKS_SETUPANDTEARDOWN_HPP
