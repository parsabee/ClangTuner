module  {
  func @forloop(%arg0: memref<256xi32>, %arg1: memref<256xi32>, %arg2: memref<256xi32>) {
    %c256 = constant 256 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    scf.for %arg3 = %c0 to %c256 step %c1 {
      %0 = memref.load %arg0[%arg3] : memref<256xi32>
      %1 = memref.load %arg1[%arg3] : memref<256xi32>
      %2 = addi %0, %1 : i32
      memref.store %2, %arg2[%arg3] : memref<256xi32>
    }
    return
  }
}