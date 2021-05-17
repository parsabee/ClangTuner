module  {
  func @the_for_loop(%arg0: memref<256xi32>, %arg1: memref<256xi32>, %arg2: memref<256xi32>) {
    %0 = memref.alloca() : i32
    %c0_i32 = constant 0 : i32
    memref.store %c0_i32, %0[] : i32
    %c256 = constant 256 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    scf.for %arg3 = %c0 to %c256 step %c1 {
      %1 = memref.alloca() : i32
      %c1_i32 = constant 1 : i32
      memref.store %c1_i32, %1[] : i32
      %c1_i32_0 = constant 1 : i32
      %2 = memref.load %1[] : i32
    }
  }
}