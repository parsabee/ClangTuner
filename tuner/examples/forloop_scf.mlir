module  {
  func @forloop(%arg0: memref<256xi32>, %arg1: memref<256xi32>, %arg2: memref<256xi32>) {
    %c256 = constant 256 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
    %1 = cmpi slt, %0, %c256 : index
    cond_br %1, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %2 = memref.load %arg0[%0] : memref<256xi32>
    %3 = memref.load %arg1[%0] : memref<256xi32>
    %4 = addi %2, %3 : i32
    memref.store %4, %arg2[%0] : memref<256xi32>
    %5 = addi %0, %c1 : index
    br ^bb1(%5 : index)
  ^bb3:  // pred: ^bb1
    return
  }
}