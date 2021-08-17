module  {
  func @__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_22_3(%arg0: f32, %arg1: memref<256xf32>, %arg2: memref<256xf32>, %arg3: memref<256xf32>) {
    %c256 = constant 256 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    scf.parallel (%arg4) = (%c0) to (%c256) step (%c1) {
      %0 = memref.load %arg2[%arg4] : memref<256xf32>
      %1 = mulf %arg0, %0 : f32
      %2 = memref.load %arg3[%arg4] : memref<256xf32>
      %3 = addf %1, %2 : f32
      memref.store %3, %arg1[%arg4] : memref<256xf32>
      scf.yield
    }
    return
  }
}