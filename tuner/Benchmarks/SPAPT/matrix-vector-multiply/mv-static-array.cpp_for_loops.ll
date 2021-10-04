; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

%struct.ident_t = type { i32, i32, i32, i32, i8* }

@0 = private unnamed_addr constant [152 x i8] c";LLVMDialectModule;__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_matrix_vector_multiply_mv_static_array_cpp_30_3;26;5;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([152 x i8], [152 x i8]* @0, i32 0, i32 0) }, align 8
@2 = private unnamed_addr constant [152 x i8] c";LLVMDialectModule;__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_matrix_vector_multiply_mv_static_array_cpp_30_3;27;7;;\00", align 1
@3 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([152 x i8], [152 x i8]* @2, i32 0, i32 0) }, align 8
@4 = private unnamed_addr constant %struct.ident_t { i32 0, i32 66, i32 0, i32 0, i8* getelementptr inbounds ([152 x i8], [152 x i8]* @2, i32 0, i32 0) }, align 8

declare i8* @malloc(i64)

declare void @free(i8*)

define void @__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_matrix_vector_multiply_mv_static_array_cpp_30_3(float* %0, float* %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, float* %7, float* %8, i64 %9, i64 %10, i64 %11, float* %12, float* %13, i64 %14, i64 %15, i64 %16) !dbg !3 {
  %.reloaded = alloca { float*, float*, i64, [2 x i64], [2 x i64] }, align 8, !dbg !7
  %.reloaded7 = alloca { float*, float*, i64, [1 x i64], [1 x i64] }, align 8, !dbg !7
  %.reloaded8 = alloca { float*, float*, i64, [1 x i64], [1 x i64] }, align 8, !dbg !7
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
  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(%struct.ident_t* @1), !dbg !25
  store { float*, float*, i64, [2 x i64], [2 x i64] } %24, { float*, float*, i64, [2 x i64], [2 x i64] }* %.reloaded, align 8
  store { float*, float*, i64, [1 x i64], [1 x i64] } %29, { float*, float*, i64, [1 x i64], [1 x i64] }* %.reloaded7, align 8
  store { float*, float*, i64, [1 x i64], [1 x i64] } %34, { float*, float*, i64, [1 x i64], [1 x i64] }* %.reloaded8, align 8
  br label %omp_parallel

omp_parallel:                                     ; preds = %17
  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* @1, i32 3, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, { float*, float*, i64, [2 x i64], [2 x i64] }*, { float*, float*, i64, [1 x i64], [1 x i64] }*, { float*, float*, i64, [1 x i64], [1 x i64] }*)* @__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_matrix_vector_multiply_mv_static_array_cpp_30_3..omp_par to void (i32*, i32*, ...)*), { float*, float*, i64, [2 x i64], [2 x i64] }* %.reloaded, { float*, float*, i64, [1 x i64], [1 x i64] }* %.reloaded7, { float*, float*, i64, [1 x i64], [1 x i64] }* %.reloaded8), !dbg !26
  br label %omp.par.outlined.exit

omp.par.outlined.exit:                            ; preds = %omp_parallel
  br label %omp.par.exit.split

omp.par.exit.split:                               ; preds = %omp.par.outlined.exit
  ret void, !dbg !27
}

; Function Attrs: norecurse nounwind
define internal void @__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_matrix_vector_multiply_mv_static_array_cpp_30_3..omp_par(i32* noalias %tid.addr, i32* noalias %zero.addr, { float*, float*, i64, [2 x i64], [2 x i64] }* %.reloaded, { float*, float*, i64, [1 x i64], [1 x i64] }* %.reloaded7, { float*, float*, i64, [1 x i64], [1 x i64] }* %.reloaded8) #0 !dbg !28 {
omp.par.entry:
  %tid.addr.local = alloca i32, align 4
  %0 = load i32, i32* %tid.addr, align 4
  store i32 %0, i32* %tid.addr.local, align 4
  %tid = load i32, i32* %tid.addr.local, align 4
  %1 = load { float*, float*, i64, [2 x i64], [2 x i64] }, { float*, float*, i64, [2 x i64], [2 x i64] }* %.reloaded, align 8
  %2 = load { float*, float*, i64, [1 x i64], [1 x i64] }, { float*, float*, i64, [1 x i64], [1 x i64] }* %.reloaded7, align 8
  %3 = load { float*, float*, i64, [1 x i64], [1 x i64] }, { float*, float*, i64, [1 x i64], [1 x i64] }* %.reloaded8, align 8
  br label %omp.par.region

omp.par.outlined.exit.exitStub:                   ; preds = %omp.par.pre_finalize
  ret void

omp.par.region:                                   ; preds = %omp.par.entry
  br label %omp.par.region1

omp.par.region1:                                  ; preds = %omp.par.region
  %p.lastiter = alloca i32, align 4
  %p.lowerbound = alloca i64, align 8
  %p.upperbound = alloca i64, align 8
  %p.stride = alloca i64, align 8
  br label %omp_loop.preheader

omp_loop.preheader:                               ; preds = %omp.par.region1
  store i64 0, i64* %p.lowerbound, align 4
  store i64 1023, i64* %p.upperbound, align 4
  store i64 1, i64* %p.stride, align 4
  %omp_global_thread_num5 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @3)
  call void @__kmpc_for_static_init_8u(%struct.ident_t* @3, i32 %omp_global_thread_num5, i32 34, i32* %p.lastiter, i64* %p.lowerbound, i64* %p.upperbound, i64* %p.stride, i64 1, i64 1)
  %4 = load i64, i64* %p.lowerbound, align 4
  %5 = load i64, i64* %p.upperbound, align 4
  %6 = sub i64 %5, %4
  %7 = add i64 %6, 1
  br label %omp_loop.header

omp_loop.header:                                  ; preds = %omp_loop.inc, %omp_loop.preheader
  %omp_loop.iv = phi i64 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
  br label %omp_loop.cond

omp_loop.cond:                                    ; preds = %omp_loop.header
  %omp_loop.cmp = icmp ult i64 %omp_loop.iv, %7
  br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit

omp_loop.exit:                                    ; preds = %omp_loop.cond
  call void @__kmpc_for_static_fini(%struct.ident_t* @3, i32 %omp_global_thread_num5)
  %omp_global_thread_num6 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @3), !dbg !29
  call void @__kmpc_barrier(%struct.ident_t* @4, i32 %omp_global_thread_num6), !dbg !29
  br label %omp_loop.after

omp_loop.after:                                   ; preds = %omp_loop.exit
  br label %omp.par.pre_finalize, !dbg !30

omp.par.pre_finalize:                             ; preds = %omp_loop.after
  br label %omp.par.outlined.exit.exitStub

omp_loop.body:                                    ; preds = %omp_loop.cond
  %8 = add i64 %omp_loop.iv, %4
  %9 = mul i64 %8, 1
  %10 = add i64 %9, 0
  br label %omp.wsloop.region

omp.wsloop.region:                                ; preds = %omp_loop.body
  br label %omp.wsloop.region2, !dbg !31

omp.wsloop.region2:                               ; preds = %omp.wsloop.region3, %omp.wsloop.region
  %11 = phi i64 [ %24, %omp.wsloop.region3 ], [ 0, %omp.wsloop.region ], !dbg !29
  %12 = icmp slt i64 %11, 1024, !dbg !32
  br i1 %12, label %omp.wsloop.region3, label %omp.wsloop.region4, !dbg !33

omp.wsloop.region4:                               ; preds = %omp.wsloop.region2
  br label %omp.wsloop.exit, !dbg !34

omp.wsloop.exit:                                  ; preds = %omp.wsloop.region4
  br label %omp_loop.inc

omp_loop.inc:                                     ; preds = %omp.wsloop.exit
  %omp_loop.next = add nuw i64 %omp_loop.iv, 1
  br label %omp_loop.header

omp.wsloop.region3:                               ; preds = %omp.wsloop.region2
  %13 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %1, 1, !dbg !35
  %14 = mul i64 %11, 1024, !dbg !36
  %15 = add i64 %14, %10, !dbg !37
  %16 = getelementptr float, float* %13, i64 %15, !dbg !38
  %17 = load float, float* %16, align 4, !dbg !39
  %18 = extractvalue { float*, float*, i64, [1 x i64], [1 x i64] } %2, 1, !dbg !40
  %19 = getelementptr float, float* %18, i64 %11, !dbg !41
  %20 = load float, float* %19, align 4, !dbg !42
  %21 = fmul float %17, %20, !dbg !43
  %22 = extractvalue { float*, float*, i64, [1 x i64], [1 x i64] } %3, 1, !dbg !44
  %23 = getelementptr float, float* %22, i64 %10, !dbg !45
  store float %21, float* %23, align 4, !dbg !46
  %24 = add i64 %11, 1, !dbg !47
  br label %omp.wsloop.region2, !dbg !48
}

; Function Attrs: nounwind
declare i32 @__kmpc_global_thread_num(%struct.ident_t*) #1

; Function Attrs: nounwind
declare void @__kmpc_for_static_init_8u(%struct.ident_t*, i32, i32, i32*, i64*, i64*, i64*, i64, i64) #1

; Function Attrs: nounwind
declare void @__kmpc_for_static_fini(%struct.ident_t*, i32) #1

; Function Attrs: convergent nounwind
declare void @__kmpc_barrier(%struct.ident_t*, i32) #2

; Function Attrs: nounwind
declare !callback !49 void @__kmpc_fork_call(%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) #1

attributes #0 = { norecurse nounwind }
attributes #1 = { nounwind }
attributes #2 = { convergent nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_matrix_vector_multiply_mv_static_array_cpp_30_3", linkageName: "__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_matrix_vector_multiply_mv_static_array_cpp_30_3", scope: null, file: !4, line: 2, type: !5, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "Benchmarks/SPAPT/matrix-vector-multiply/mv-static-array.cpp_for_loops_opt.mlir", directory: "/Users/parsabagheri/Development/llvm-project/tuner")
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
!26 = !DILocation(line: 27, column: 7, scope: !8)
!27 = !DILocation(line: 56, column: 5, scope: !8)
!28 = distinct !DISubprogram(name: "__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_matrix_vector_multiply_mv_static_array_cpp_30_3..omp_par", linkageName: "__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_matrix_vector_multiply_mv_static_array_cpp_30_3..omp_par", scope: null, file: !4, type: !5, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!29 = !DILocation(line: 27, column: 7, scope: !28)
!30 = !DILocation(line: 54, column: 7, scope: !28)
!31 = !DILocation(line: 31, column: 9, scope: !28)
!32 = !DILocation(line: 33, column: 15, scope: !28)
!33 = !DILocation(line: 34, column: 9, scope: !28)
!34 = !DILocation(line: 52, column: 9, scope: !28)
!35 = !DILocation(line: 36, column: 15, scope: !28)
!36 = !DILocation(line: 38, column: 15, scope: !28)
!37 = !DILocation(line: 39, column: 15, scope: !28)
!38 = !DILocation(line: 40, column: 15, scope: !28)
!39 = !DILocation(line: 41, column: 15, scope: !28)
!40 = !DILocation(line: 42, column: 15, scope: !28)
!41 = !DILocation(line: 43, column: 15, scope: !28)
!42 = !DILocation(line: 44, column: 15, scope: !28)
!43 = !DILocation(line: 45, column: 15, scope: !28)
!44 = !DILocation(line: 46, column: 15, scope: !28)
!45 = !DILocation(line: 47, column: 15, scope: !28)
!46 = !DILocation(line: 48, column: 9, scope: !28)
!47 = !DILocation(line: 49, column: 15, scope: !28)
!48 = !DILocation(line: 50, column: 9, scope: !28)
!49 = !{!50}
!50 = !{i64 2, i64 -1, i64 -1, i1 true}
