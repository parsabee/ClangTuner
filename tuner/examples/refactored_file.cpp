extern void forloop(int [256], int [256], int [256]);
#define N 256

void f() {
  int a[N];
  int b[N];
  int c[N] = {0};
  for (int i = 0; i < N; i++) {
    a[i] = 1;
    b[i] = 2;
  }

  forloop(a , b , c );

}

int main(int argc, char **argv) {
  f();
}
