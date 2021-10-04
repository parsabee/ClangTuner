; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare i8* @malloc(i64)

declare void @free(i8*)

define void @__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_22_3(float %0, float* %1, float* %2, i64 %3, i64 %4, i64 %5, float* %6, float* %7, i64 %8, i64 %9, i64 %10, float* %11, float* %12, i64 %13, i64 %14, i64 %15) !dbg !3 {
  %17 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } undef, float* %1, 0, !dbg !7
  %18 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %17, float* %2, 1, !dbg !9
  %19 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %18, i64 %3, 2, !dbg !10
  %20 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %19, i64 %4, 3, 0, !dbg !11
  %21 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %20, i64 %5, 4, 0, !dbg !12
  %22 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } undef, float* %6, 0, !dbg !13
  %23 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %22, float* %7, 1, !dbg !14
  %24 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %23, i64 %8, 2, !dbg !15
  %25 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %24, i64 %9, 3, 0, !dbg !16
  %26 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %25, i64 %10, 4, 0, !dbg !17
  %27 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } undef, float* %11, 0, !dbg !18
  %28 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %27, float* %12, 1, !dbg !19
  %29 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %28, i64 %13, 2, !dbg !20
  %30 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %29, i64 %14, 3, 0, !dbg !21
  %31 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %30, i64 %15, 4, 0, !dbg !22
  br label %32, !dbg !23

32:                                               ; preds = %35, %16
  %33 = phi i64 [ %46, %35 ], [ 0, %16 ]
  %34 = icmp slt i64 %33, 256, !dbg !24
  br i1 %34, label %35, label %47, !dbg !25

35:                                               ; preds = %32
  %36 = extractvalue { float*, float*, i64, [1 x i64], [1 x i64] } %26, 1, !dbg !26
  %37 = getelementptr float, float* %36, i64 %33, !dbg !27
  %38 = load float, float* %37, align 4, !dbg !28
  %39 = fmul float %0, %38, !dbg !29
  %40 = extractvalue { float*, float*, i64, [1 x i64], [1 x i64] } %31, 1, !dbg !30
  %41 = getelementptr float, float* %40, i64 %33, !dbg !31
  %42 = load float, float* %41, align 4, !dbg !32
  %43 = fadd float %39, %42, !dbg !33
  %44 = extractvalue { float*, float*, i64, [1 x i64], [1 x i64] } %21, 1, !dbg !34
  %45 = getelementptr float, float* %44, i64 %33, !dbg !35
  store float %43, float* %45, align 4, !dbg !36
  %46 = add i64 %33, 1, !dbg !37
  br label %32, !dbg !38

47:                                               ; preds = %32
  ret void, !dbg !39
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_22_3", linkageName: "__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_22_3", scope: null, file: !4, line: 2, type: !5, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "<stdin>", directory: "/Users/parsabagheri/Development/llvm-project/tuner/Benchmarks/SPAPT/axpy")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 4, column: 10, scope: !8)
!8 = !DILexicalBlockFile(scope: !3, file: !4, discriminator: 0)
!9 = !DILocation(line: 5, column: 10, scope: !8)
!10 = !DILocation(line: 6, column: 10, scope: !8)
!11 = !DILocation(line: 7, column: 10, scope: !8)
!12 = !DILocation(line: 8, column: 10, scope: !8)
!13 = !DILocation(line: 10, column: 10, scope: !8)
!14 = !DILocation(line: 11, column: 10, scope: !8)
!15 = !DILocation(line: 12, column: 10, scope: !8)
!16 = !DILocation(line: 13, column: 11, scope: !8)
!17 = !DILocation(line: 14, column: 11, scope: !8)
!18 = !DILocation(line: 16, column: 11, scope: !8)
!19 = !DILocation(line: 17, column: 11, scope: !8)
!20 = !DILocation(line: 18, column: 11, scope: !8)
!21 = !DILocation(line: 19, column: 11, scope: !8)
!22 = !DILocation(line: 20, column: 11, scope: !8)
!23 = !DILocation(line: 24, column: 5, scope: !8)
!24 = !DILocation(line: 26, column: 11, scope: !8)
!25 = !DILocation(line: 27, column: 5, scope: !8)
!26 = !DILocation(line: 29, column: 11, scope: !8)
!27 = !DILocation(line: 30, column: 11, scope: !8)
!28 = !DILocation(line: 31, column: 11, scope: !8)
!29 = !DILocation(line: 32, column: 11, scope: !8)
!30 = !DILocation(line: 33, column: 11, scope: !8)
!31 = !DILocation(line: 34, column: 11, scope: !8)
!32 = !DILocation(line: 35, column: 11, scope: !8)
!33 = !DILocation(line: 36, column: 11, scope: !8)
!34 = !DILocation(line: 37, column: 11, scope: !8)
!35 = !DILocation(line: 38, column: 11, scope: !8)
!36 = !DILocation(line: 39, column: 5, scope: !8)
!37 = !DILocation(line: 40, column: 11, scope: !8)
!38 = !DILocation(line: 41, column: 5, scope: !8)
!39 = !DILocation(line: 43, column: 5, scope: !8)
