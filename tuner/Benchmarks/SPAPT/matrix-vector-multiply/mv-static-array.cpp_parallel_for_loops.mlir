module  {
  func @__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_matrix_vector_multiply_mv_static_array_cpp_30_3(%arg0: memref<1024x1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>) {
    %c1024 = constant 1024 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    scf.parallel (%arg3) = (%c0) to (%c1024) step (%c1) {
      %c1024_0 = constant 1024 : index
      %c0_1 = constant 0 : index
      %c1_2 = constant 1 : index
      scf.for %arg4 = %c0_1 to %c1024_0 step %c1_2 {
        %0 = memref.load %arg0[%arg4, %arg3] : memref<1024x1024xf32>
        %1 = memref.load %arg1[%arg4] : memref<1024xf32>
        %2 = mulf %0, %1 : f32
        memref.store %2, %arg2[%arg3] : memref<1024xf32>
      }
      scf.yield
    }
    return
  }
}