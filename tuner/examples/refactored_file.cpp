#include <cstdlib>
#include <cstdio>
#include <memory>
extern "C" {
void forloop(int *, int *, std::size_t, std::size_t , std::size_t,
             int *, int *, std::size_t, std::size_t , std::size_t,
             int *, int *, std::size_t, std::size_t , std::size_t);
}

#define N 256

void f() {
  int a[N];
  int b[N];
  int c[N] = {0};
  for (int i = 0; i < N; i++) {
    a[i] = 1;
    b[i] = 2;
  }

  void *ap = a;
  void *bp = b;
  void *cp = c;
  std::size_t sz = N;
  auto aligna = std::align(alignof(int), sizeof(int), ap, sz);
  auto alignb = std::align(alignof(int), sizeof(int), bp, sz);
  auto alignc = std::align(alignof(int), sizeof(int), cp, sz);
  forloop(a , a, 0, N, 1,
          b , b, 0, N, 1,
          c , c, 0, N, 1);
  printf("%lu\n", sz);
  for (int i = 0; i < N; i++)
  printf("%d\n", c[i]);
}

int main(int argc, char **argv) {
  f();
}
