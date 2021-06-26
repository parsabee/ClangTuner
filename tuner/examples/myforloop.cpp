void forloop(int a[256], int b[256], int c[256]) {
  for (int i = 0; i < 256; i += 1) {
    c[i] = a[i] + b[i];
  }
}