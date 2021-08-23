module  {
  llvm.func @__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_22_3(%arg0: f32, %arg1: !llvm.ptr<f32>, %arg2: !llvm.ptr<f32>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !llvm.ptr<f32>, %arg7: !llvm.ptr<f32>, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr<f32>, %arg12: !llvm.ptr<f32>, %arg13: i64, %arg14: i64, %arg15: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg2, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.insertvalue %arg3, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.insertvalue %arg4, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.insertvalue %arg7, %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg8, %8[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg9, %9[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %arg10, %10[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.insertvalue %arg11, %12[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.insertvalue %arg12, %13[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.insertvalue %arg13, %14[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.insertvalue %arg14, %15[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.insertvalue %arg15, %16[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %18 = llvm.mlir.constant(256 : index) : i64
    %19 = llvm.mlir.constant(0 : index) : i64
    %20 = llvm.mlir.constant(1 : index) : i64
    omp.parallel {
      omp.wsloop (%arg16) : i64 = (%19) to (%18) step (%20) {
        %21 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
        %22 = llvm.getelementptr %21[%arg16] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
        %23 = llvm.load %22 : !llvm.ptr<f32>
        %24 = llvm.fmul %arg0, %23  : f32
        %25 = llvm.extractvalue %17[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
        %26 = llvm.getelementptr %25[%arg16] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
        %27 = llvm.load %26 : !llvm.ptr<f32>
        %28 = llvm.fadd %24, %27  : f32
        %29 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
        %30 = llvm.getelementptr %29[%arg16] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
        llvm.store %28, %30 : !llvm.ptr<f32>
        omp.yield
      }
      omp.terminator
    }
    llvm.return
  }
}

