; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%struct.ident_t = type { i32, i32, i32, i32, i8* }

@0 = private unnamed_addr constant [123 x i8] c";LLVMDialectModule;__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_24_3;26;5;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([123 x i8], [123 x i8]* @0, i32 0, i32 0) }, align 8
@2 = private unnamed_addr constant [123 x i8] c";LLVMDialectModule;__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_24_3;28;9;;\00", align 1
@3 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([123 x i8], [123 x i8]* @2, i32 0, i32 0) }, align 8
@4 = private unnamed_addr constant %struct.ident_t { i32 0, i32 66, i32 0, i32 0, i8* getelementptr inbounds ([123 x i8], [123 x i8]* @2, i32 0, i32 0) }, align 8
@5 = private unnamed_addr constant [123 x i8] c";LLVMDialectModule;__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_24_3;27;7;;\00", align 1
@6 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([123 x i8], [123 x i8]* @5, i32 0, i32 0) }, align 8
@7 = private unnamed_addr constant %struct.ident_t { i32 0, i32 66, i32 0, i32 0, i8* getelementptr inbounds ([123 x i8], [123 x i8]* @5, i32 0, i32 0) }, align 8

declare i8* @malloc(i64)

declare void @free(i8*)

define void @__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_24_3(float %0, float* %1, float* %2, i64 %3, i64 %4, i64 %5, float* %6, float* %7, i64 %8, i64 %9, i64 %10, float* %11, float* %12, i64 %13, i64 %14, i64 %15) !dbg !3 {
  %.reloaded = alloca { float*, float*, i64, [1 x i64], [1 x i64] }, align 8, !dbg !7
  %.reloaded22 = alloca float, align 4, !dbg !7
  %.reloaded23 = alloca { float*, float*, i64, [1 x i64], [1 x i64] }, align 8, !dbg !7
  %.reloaded24 = alloca { float*, float*, i64, [1 x i64], [1 x i64] }, align 8, !dbg !7
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
  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(%struct.ident_t* @1), !dbg !23
  store { float*, float*, i64, [1 x i64], [1 x i64] } %26, { float*, float*, i64, [1 x i64], [1 x i64] }* %.reloaded, align 8
  store float %0, float* %.reloaded22, align 4
  store { float*, float*, i64, [1 x i64], [1 x i64] } %31, { float*, float*, i64, [1 x i64], [1 x i64] }* %.reloaded23, align 8
  store { float*, float*, i64, [1 x i64], [1 x i64] } %21, { float*, float*, i64, [1 x i64], [1 x i64] }* %.reloaded24, align 8
  br label %omp_parallel

omp_parallel:                                     ; preds = %16
  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* @1, i32 4, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, { float*, float*, i64, [1 x i64], [1 x i64] }*, float*, { float*, float*, i64, [1 x i64], [1 x i64] }*, { float*, float*, i64, [1 x i64], [1 x i64] }*)* @__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_24_3..omp_par to void (i32*, i32*, ...)*), { float*, float*, i64, [1 x i64], [1 x i64] }* %.reloaded, float* %.reloaded22, { float*, float*, i64, [1 x i64], [1 x i64] }* %.reloaded23, { float*, float*, i64, [1 x i64], [1 x i64] }* %.reloaded24), !dbg !24
  br label %omp.par.outlined.exit

omp.par.outlined.exit:                            ; preds = %omp_parallel
  br label %omp.par.exit.split

omp.par.exit.split:                               ; preds = %omp.par.outlined.exit
  ret void, !dbg !25
}

; Function Attrs: norecurse nounwind
define internal void @__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_24_3..omp_par(i32* noalias %tid.addr, i32* noalias %zero.addr, { float*, float*, i64, [1 x i64], [1 x i64] }* %.reloaded, float* %.reloaded22, { float*, float*, i64, [1 x i64], [1 x i64] }* %.reloaded23, { float*, float*, i64, [1 x i64], [1 x i64] }* %.reloaded24) #0 !dbg !26 {
omp.par.entry:
  %tid.addr.local = alloca i32, align 4
  %0 = load i32, i32* %tid.addr, align 4
  store i32 %0, i32* %tid.addr.local, align 4
  %tid = load i32, i32* %tid.addr.local, align 4
  %1 = load { float*, float*, i64, [1 x i64], [1 x i64] }, { float*, float*, i64, [1 x i64], [1 x i64] }* %.reloaded, align 8
  %2 = load float, float* %.reloaded22, align 4
  %3 = load { float*, float*, i64, [1 x i64], [1 x i64] }, { float*, float*, i64, [1 x i64], [1 x i64] }* %.reloaded23, align 8
  %4 = load { float*, float*, i64, [1 x i64], [1 x i64] }, { float*, float*, i64, [1 x i64], [1 x i64] }* %.reloaded24, align 8
  br label %omp.par.region

omp.par.outlined.exit.exitStub:                   ; preds = %omp.par.pre_finalize
  ret void

omp.par.region:                                   ; preds = %omp.par.entry
  br label %omp.par.region1

omp.par.region1:                                  ; preds = %omp.par.region
  %p.lastiter16 = alloca i32, align 4
  %p.lowerbound17 = alloca i64, align 8
  %p.upperbound18 = alloca i64, align 8
  %p.stride19 = alloca i64, align 8
  br label %omp_loop.preheader

omp_loop.preheader:                               ; preds = %omp.par.region1
  store i64 0, i64* %p.lowerbound17, align 4
  store i64 255, i64* %p.upperbound18, align 4
  store i64 1, i64* %p.stride19, align 4
  %omp_global_thread_num20 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @6)
  call void @__kmpc_for_static_init_8u(%struct.ident_t* @6, i32 %omp_global_thread_num20, i32 34, i32* %p.lastiter16, i64* %p.lowerbound17, i64* %p.upperbound18, i64* %p.stride19, i64 1, i64 1)
  %5 = load i64, i64* %p.lowerbound17, align 4
  %6 = load i64, i64* %p.upperbound18, align 4
  %7 = sub i64 %6, %5
  %8 = add i64 %7, 1
  br label %omp_loop.header

omp_loop.header:                                  ; preds = %omp_loop.inc, %omp_loop.preheader
  %omp_loop.iv = phi i64 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
  br label %omp_loop.cond

omp_loop.cond:                                    ; preds = %omp_loop.header
  %omp_loop.cmp = icmp ult i64 %omp_loop.iv, %8
  br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit

omp_loop.exit:                                    ; preds = %omp_loop.cond
  call void @__kmpc_for_static_fini(%struct.ident_t* @6, i32 %omp_global_thread_num20)
  %omp_global_thread_num21 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @6), !dbg !27
  call void @__kmpc_barrier(%struct.ident_t* @7, i32 %omp_global_thread_num21), !dbg !27
  br label %omp_loop.after

omp_loop.after:                                   ; preds = %omp_loop.exit
  br label %omp.par.pre_finalize, !dbg !28

omp.par.pre_finalize:                             ; preds = %omp_loop.after
  br label %omp.par.outlined.exit.exitStub

omp_loop.body:                                    ; preds = %omp_loop.cond
  %9 = add i64 %omp_loop.iv, %5
  %10 = mul i64 %9, 1
  %11 = add i64 %10, 0
  br label %omp.wsloop.region

omp.wsloop.region:                                ; preds = %omp_loop.body
  %p.lastiter = alloca i32, align 4
  %p.lowerbound = alloca i64, align 8
  %p.upperbound = alloca i64, align 8
  %p.stride = alloca i64, align 8
  br label %omp_loop.preheader2

omp_loop.preheader2:                              ; preds = %omp.wsloop.region
  store i64 0, i64* %p.lowerbound, align 4
  store i64 0, i64* %p.upperbound, align 4
  store i64 1, i64* %p.stride, align 4
  %omp_global_thread_num14 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @3)
  call void @__kmpc_for_static_init_8u(%struct.ident_t* @3, i32 %omp_global_thread_num14, i32 34, i32* %p.lastiter, i64* %p.lowerbound, i64* %p.upperbound, i64* %p.stride, i64 1, i64 1)
  %12 = load i64, i64* %p.lowerbound, align 4
  %13 = load i64, i64* %p.upperbound, align 4
  %14 = sub i64 %13, %12
  %15 = add i64 %14, 1
  br label %omp_loop.header3

omp_loop.header3:                                 ; preds = %omp_loop.inc6, %omp_loop.preheader2
  %omp_loop.iv9 = phi i64 [ 0, %omp_loop.preheader2 ], [ %omp_loop.next11, %omp_loop.inc6 ]
  br label %omp_loop.cond4

omp_loop.cond4:                                   ; preds = %omp_loop.header3
  %omp_loop.cmp10 = icmp ult i64 %omp_loop.iv9, %15
  br i1 %omp_loop.cmp10, label %omp_loop.body5, label %omp_loop.exit7

omp_loop.exit7:                                   ; preds = %omp_loop.cond4
  call void @__kmpc_for_static_fini(%struct.ident_t* @3, i32 %omp_global_thread_num14)
  %omp_global_thread_num15 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @3), !dbg !29
  call void @__kmpc_barrier(%struct.ident_t* @4, i32 %omp_global_thread_num15), !dbg !29
  br label %omp_loop.after8

omp_loop.after8:                                  ; preds = %omp_loop.exit7
  br label %omp.wsloop.exit, !dbg !30

omp.wsloop.exit:                                  ; preds = %omp_loop.after8
  br label %omp_loop.inc

omp_loop.inc:                                     ; preds = %omp.wsloop.exit
  %omp_loop.next = add nuw i64 %omp_loop.iv, 1
  br label %omp_loop.header

omp_loop.body5:                                   ; preds = %omp_loop.cond4
  %16 = add i64 %omp_loop.iv9, %12
  %17 = mul i64 %16, 1
  %18 = add i64 %17, 0
  br label %omp.wsloop.region13

omp.wsloop.region13:                              ; preds = %omp_loop.body5
  %19 = add i64 %18, %11, !dbg !31
  %20 = extractvalue { float*, float*, i64, [1 x i64], [1 x i64] } %1, 1, !dbg !32
  %21 = getelementptr float, float* %20, i64 %19, !dbg !33
  %22 = load float, float* %21, align 4, !dbg !34
  %23 = fmul float %2, %22, !dbg !35
  %24 = extractvalue { float*, float*, i64, [1 x i64], [1 x i64] } %3, 1, !dbg !36
  %25 = getelementptr float, float* %24, i64 %19, !dbg !37
  %26 = load float, float* %25, align 4, !dbg !38
  %27 = fadd float %23, %26, !dbg !39
  %28 = extractvalue { float*, float*, i64, [1 x i64], [1 x i64] } %4, 1, !dbg !40
  %29 = getelementptr float, float* %28, i64 %19, !dbg !41
  store float %27, float* %29, align 4, !dbg !42
  br label %omp.wsloop.exit12, !dbg !43

omp.wsloop.exit12:                                ; preds = %omp.wsloop.region13
  br label %omp_loop.inc6

omp_loop.inc6:                                    ; preds = %omp.wsloop.exit12
  %omp_loop.next11 = add nuw i64 %omp_loop.iv9, 1
  br label %omp_loop.header3
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
declare !callback !44 void @__kmpc_fork_call(%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) #1

attributes #0 = { norecurse nounwind }
attributes #1 = { nounwind }
attributes #2 = { convergent nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_24_3", linkageName: "__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_24_3", scope: null, file: !4, line: 2, type: !5, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "Benchmarks/SPAPT/axpy/axpy.cpp_for_loops_opt.mlir", directory:
"/Users/parsabagheri/Development/llvm-project/tuner")
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
!23 = !DILocation(line: 26, column: 5, scope: !8)
!24 = !DILocation(line: 27, column: 7, scope: !8)
!25 = !DILocation(line: 47, column: 5, scope: !8)
!26 = distinct !DISubprogram(name: "__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_24_3..omp_par", linkageName: "__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_24_3..omp_par", scope: null, file: !4, type: !5, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!27 = !DILocation(line: 27, column: 7, scope: !26)
!28 = !DILocation(line: 45, column: 7, scope: !26)
!29 = !DILocation(line: 28, column: 9, scope: !26)
!30 = !DILocation(line: 43, column: 9, scope: !26)
!31 = !DILocation(line: 29, column: 17, scope: !26)
!32 = !DILocation(line: 30, column: 17, scope: !26)
!33 = !DILocation(line: 31, column: 17, scope: !26)
!34 = !DILocation(line: 32, column: 17, scope: !26)
!35 = !DILocation(line: 33, column: 17, scope: !26)
!36 = !DILocation(line: 34, column: 17, scope: !26)
!37 = !DILocation(line: 35, column: 17, scope: !26)
!38 = !DILocation(line: 36, column: 17, scope: !26)
!39 = !DILocation(line: 37, column: 17, scope: !26)
!40 = !DILocation(line: 38, column: 17, scope: !26)
!41 = !DILocation(line: 39, column: 17, scope: !26)
!42 = !DILocation(line: 40, column: 11, scope: !26)
!43 = !DILocation(line: 41, column: 11, scope: !26)
!44 = !{!45}
!45 = !{i64 2, i64 -1, i64 -1, i1 true}
