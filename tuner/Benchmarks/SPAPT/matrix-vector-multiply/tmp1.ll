; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare i8* @malloc(i64)

declare void @free(i8*)

define void @__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_matrix_vector_multiply_mv_static_array_cpp_30_3(float* %0, float* %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, float* %7, float* %8, i64 %9, i64 %10, i64 %11, float* %12, float* %13, i64 %14, i64 %15, i64 %16) !dbg !3 {
  %18 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } undef, float* %0, 0, !dbg !7
  %19 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %18, float* %1, 1, !dbg !9
  %20 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %19, i64 %2, 2, !dbg !10
  %21 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %20, i64 %3, 3, 0, !dbg !11
  %22 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %21, i64 %5, 4, 0, !dbg !12
  %23 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %22, i64 %4, 3, 1, !dbg !13
  %24 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %23, i64 %6, 4, 1, !dbg !14
  %25 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } undef, float* %7, 0, !dbg !15
  %26 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %25, float* %8, 1, !dbg !16
  %27 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %26, i64 %9, 2, !dbg !17
  %28 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %27, i64 %10, 3, 0, !dbg !18
  %29 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %28, i64 %11, 4, 0, !dbg !19
  %30 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } undef, float* %12, 0, !dbg !20
  %31 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %30, float* %13, 1, !dbg !21
  %32 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %31, i64 %14, 2, !dbg !22
  %33 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %32, i64 %15, 3, 0, !dbg !23
  %34 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %33, i64 %16, 4, 0, !dbg !24
  br label %35, !dbg !25

35:                                               ; preds = %55, %17
  %36 = phi i64 [ %56, %55 ], [ 0, %17 ]
  %37 = icmp slt i64 %36, 4, !dbg !26
  br i1 %37, label %38, label %57, !dbg !27

38:                                               ; preds = %35
  br label %39, !dbg !28

39:                                               ; preds = %42, %38
  %40 = phi i64 [ %54, %42 ], [ 0, %38 ]
  %41 = icmp slt i64 %40, 4, !dbg !29
  br i1 %41, label %42, label %55, !dbg !30

42:                                               ; preds = %39
  %43 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %24, 1, !dbg !31
  %44 = mul i64 %40, 4, !dbg !32
  %45 = add i64 %44, %36, !dbg !33
  %46 = getelementptr float, float* %43, i64 %45, !dbg !34
  %47 = load float, float* %46, align 4, !dbg !35
  %48 = extractvalue { float*, float*, i64, [1 x i64], [1 x i64] } %29, 1, !dbg !36
  %49 = getelementptr float, float* %48, i64 %40, !dbg !37
  %50 = load float, float* %49, align 4, !dbg !38
  %51 = fmul float %47, %50, !dbg !39
  %52 = extractvalue { float*, float*, i64, [1 x i64], [1 x i64] } %34, 1, !dbg !40
  %53 = getelementptr float, float* %52, i64 %36, !dbg !41
  store float %51, float* %53, align 4, !dbg !42
  %54 = add i64 %40, 1, !dbg !43
  br label %39, !dbg !44

55:                                               ; preds = %39
  %56 = add i64 %36, 1, !dbg !45
  br label %35, !dbg !46

57:                                               ; preds = %35
  ret void, !dbg !47
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_matrix_vector_multiply_mv_static_array_cpp_30_3", linkageName: "__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_matrix_vector_multiply_mv_static_array_cpp_30_3", scope: null, file: !4, line: 2, type: !5, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "<stdin>", directory: "/Users/parsabagheri/Development/llvm-project/tuner/Benchmarks/SPAPT/matrix-vector-multiply")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 4, column: 10, scope: !8)
!8 = !DILexicalBlockFile(scope: !3, file: !4, discriminator: 0)
!9 = !DILocation(line: 5, column: 10, scope: !8)
!10 = !DILocation(line: 6, column: 10, scope: !8)
!11 = !DILocation(line: 7, column: 10, scope: !8)
!12 = !DILocation(line: 8, column: 10, scope: !8)
!13 = !DILocation(line: 9, column: 10, scope: !8)
!14 = !DILocation(line: 10, column: 10, scope: !8)
!15 = !DILocation(line: 12, column: 10, scope: !8)
!16 = !DILocation(line: 13, column: 11, scope: !8)
!17 = !DILocation(line: 14, column: 11, scope: !8)
!18 = !DILocation(line: 15, column: 11, scope: !8)
!19 = !DILocation(line: 16, column: 11, scope: !8)
!20 = !DILocation(line: 18, column: 11, scope: !8)
!21 = !DILocation(line: 19, column: 11, scope: !8)
!22 = !DILocation(line: 20, column: 11, scope: !8)
!23 = !DILocation(line: 21, column: 11, scope: !8)
!24 = !DILocation(line: 22, column: 11, scope: !8)
!25 = !DILocation(line: 26, column: 5, scope: !8)
!26 = !DILocation(line: 28, column: 11, scope: !8)
!27 = !DILocation(line: 29, column: 5, scope: !8)
!28 = !DILocation(line: 31, column: 5, scope: !8)
!29 = !DILocation(line: 33, column: 11, scope: !8)
!30 = !DILocation(line: 34, column: 5, scope: !8)
!31 = !DILocation(line: 36, column: 11, scope: !8)
!32 = !DILocation(line: 38, column: 11, scope: !8)
!33 = !DILocation(line: 39, column: 11, scope: !8)
!34 = !DILocation(line: 40, column: 11, scope: !8)
!35 = !DILocation(line: 41, column: 11, scope: !8)
!36 = !DILocation(line: 42, column: 11, scope: !8)
!37 = !DILocation(line: 43, column: 11, scope: !8)
!38 = !DILocation(line: 44, column: 11, scope: !8)
!39 = !DILocation(line: 45, column: 11, scope: !8)
!40 = !DILocation(line: 46, column: 11, scope: !8)
!41 = !DILocation(line: 47, column: 11, scope: !8)
!42 = !DILocation(line: 48, column: 5, scope: !8)
!43 = !DILocation(line: 49, column: 11, scope: !8)
!44 = !DILocation(line: 50, column: 5, scope: !8)
!45 = !DILocation(line: 52, column: 11, scope: !8)
!46 = !DILocation(line: 53, column: 5, scope: !8)
!47 = !DILocation(line: 55, column: 5, scope: !8)
