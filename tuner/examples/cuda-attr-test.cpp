#define N 256
int main(int argc, char **argv) {
  int a[N];
  int b[N];
  int c[N] = {0};
  for (int i = 0; i < N; i++) {
    a[i] = 1;
    b[i] = 2;
  }

  [[clang::block_dim(32, 64)]]
  for (int i = 0; i < N; i += 1) {
    int j = 1;
    c[i] = a[j] + b[i];
  }
}
