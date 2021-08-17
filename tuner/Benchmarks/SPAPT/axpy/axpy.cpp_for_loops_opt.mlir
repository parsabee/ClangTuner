module  {
  func @__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_22_3(%arg0: f32, %arg1: memref<256xf32>, %arg2: memref<256xf32>, %arg3: memref<256xf32>) {
    %c256 = constant 256 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c0_0 = constant 0 : index
    %c1_1 = constant 1 : index
    %0 = muli %c1, %c1_1 : index
    scf.parallel (%arg4) = (%c0) to (%c256) step (%0) {
      scf.parallel (%arg5) = (%c0_0) to (%0) step (%c1) {
        %1 = addi %arg5, %arg4 : index
        %2 = memref.load %arg2[%1] : memref<256xf32>
        %3 = mulf %arg0, %2 : f32
        %4 = memref.load %arg3[%1] : memref<256xf32>
        %5 = addf %3, %4 : f32
        memref.store %5, %arg1[%1] : memref<256xf32>
        scf.yield
      }
      scf.yield
    }
    return
  }
}

