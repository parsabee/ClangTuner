; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%0 = type { i32, i32, i32, i32, i8* }

@0 = private unnamed_addr constant [123 x i8] c";LLVMDialectModule;__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_22_3;24;5;;\00", align 1
@1 = private unnamed_addr constant %0 { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([123 x i8], [123 x i8]* @0, i32 0, i32 0) }, align 8
@2 = private unnamed_addr constant [123 x i8] c";LLVMDialectModule;__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_22_3;25;7;;\00", align 1
@3 = private unnamed_addr constant %0 { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([123 x i8], [123 x i8]* @2, i32 0, i32 0) }, align 8
@4 = private unnamed_addr constant %0 { i32 0, i32 66, i32 0, i32 0, i8* getelementptr inbounds ([123 x i8], [123 x i8]* @2, i32 0, i32 0) }, align 8

declare i8* @malloc(i64)

declare void @free(i8*)

define void @__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_22_3(float %0, float* %1, float* %2, i64 %3, i64 %4, i64 %5, float* %6, float* %7, i64 %8, i64 %9, i64 %10, float* %11, float* %12, i64 %13, i64 %14, i64 %15) {
  %17 = alloca { float*, float*, i64, [1 x i64], [1 x i64] }, align 8
  %18 = alloca float, align 4
  %19 = alloca { float*, float*, i64, [1 x i64], [1 x i64] }, align 8
  %20 = alloca { float*, float*, i64, [1 x i64], [1 x i64] }, align 8
  %21 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } undef, float* %1, 0
  %22 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %21, float* %2, 1
  %23 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %22, i64 %3, 2
  %24 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %23, i64 %4, 3, 0
  %25 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %24, i64 %5, 4, 0
  %26 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } undef, float* %6, 0
  %27 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %26, float* %7, 1
  %28 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %27, i64 %8, 2
  %29 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %28, i64 %9, 3, 0
  %30 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %29, i64 %10, 4, 0
  %31 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } undef, float* %11, 0
  %32 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %31, float* %12, 1
  %33 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %32, i64 %13, 2
  %34 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %33, i64 %14, 3, 0
  %35 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %34, i64 %15, 4, 0
  %36 = call i32 @__kmpc_global_thread_num(%0* @1)
  store { float*, float*, i64, [1 x i64], [1 x i64] } %30, { float*, float*, i64, [1 x i64], [1 x i64] }* %17, align 8
  store float %0, float* %18, align 4
  store { float*, float*, i64, [1 x i64], [1 x i64] } %35, { float*, float*, i64, [1 x i64], [1 x i64] }* %19, align 8
  store { float*, float*, i64, [1 x i64], [1 x i64] } %25, { float*, float*, i64, [1 x i64], [1 x i64] }* %20, align 8
  br label %37

37:                                               ; preds = %16
  call void (%0*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%0* @1, i32 4, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, { float*, float*, i64, [1 x i64], [1 x i64] }*, float*, { float*, float*, i64, [1 x i64], [1 x i64] }*, { float*, float*, i64, [1 x i64], [1 x i64] }*)* @5 to void (i32*, i32*, ...)*), { float*, float*, i64, [1 x i64], [1 x i64] }* %17, float* %18, { float*, float*, i64, [1 x i64], [1 x i64] }* %19, { float*, float*, i64, [1 x i64], [1 x i64] }* %20)
  br label %38

38:                                               ; preds = %37
  br label %39

39:                                               ; preds = %38
  ret void
}

; Function Attrs: norecurse nounwind
define internal void @5(i32* noalias %0, i32* noalias %1, { float*, float*, i64, [1 x i64], [1 x i64] }* %2, float* %3, { float*, float*, i64, [1 x i64], [1 x i64] }* %4, { float*, float*, i64, [1 x i64], [1 x i64] }* %5) #0 {
  %7 = alloca i32, align 4
  %8 = load i32, i32* %0, align 4
  store i32 %8, i32* %7, align 4
  %9 = load i32, i32* %7, align 4
  %10 = load { float*, float*, i64, [1 x i64], [1 x i64] }, { float*, float*, i64, [1 x i64], [1 x i64] }* %2, align 8
  %11 = load float, float* %3, align 4
  %12 = load { float*, float*, i64, [1 x i64], [1 x i64] }, { float*, float*, i64, [1 x i64], [1 x i64] }* %4, align 8
  %13 = load { float*, float*, i64, [1 x i64], [1 x i64] }, { float*, float*, i64, [1 x i64], [1 x i64] }* %5, align 8
  br label %15

14:                                               ; preds = %34
  ret void

15:                                               ; preds = %6
  br label %16

16:                                               ; preds = %15
  %17 = alloca i32, align 4
  %18 = alloca i64, align 8
  %19 = alloca i64, align 8
  %20 = alloca i64, align 8
  br label %21

21:                                               ; preds = %16
  store i64 0, i64* %18, align 4
  store i64 255, i64* %19, align 4
  store i64 1, i64* %20, align 4
  %22 = call i32 @__kmpc_global_thread_num(%0* @3)
  call void @__kmpc_for_static_init_8u(%0* @3, i32 %22, i32 34, i32* %17, i64* %18, i64* %19, i64* %20, i64 1, i64 1)
  %23 = load i64, i64* %18, align 4
  %24 = load i64, i64* %19, align 4
  %25 = sub i64 %24, %23
  %26 = add i64 %25, 1
  br label %27

27:                                               ; preds = %51, %21
  %28 = phi i64 [ 0, %21 ], [ %52, %51 ]
  br label %29

29:                                               ; preds = %27
  %30 = icmp ult i64 %28, %26
  br i1 %30, label %35, label %31

31:                                               ; preds = %29
  call void @__kmpc_for_static_fini(%0* @3, i32 %22)
  %32 = call i32 @__kmpc_global_thread_num(%0* @3)
  call void @__kmpc_barrier(%0* @4, i32 %32)
  br label %33

33:                                               ; preds = %31
  br label %34

34:                                               ; preds = %33
  br label %14

35:                                               ; preds = %29
  %36 = add i64 %28, %23
  %37 = mul i64 %36, 1
  %38 = add i64 %37, 0
  br label %39

39:                                               ; preds = %35
  %40 = extractvalue { float*, float*, i64, [1 x i64], [1 x i64] } %10, 1
  %41 = getelementptr float, float* %40, i64 %38
  %42 = load float, float* %41, align 4
  %43 = fmul float %11, %42
  %44 = extractvalue { float*, float*, i64, [1 x i64], [1 x i64] } %12, 1
  %45 = getelementptr float, float* %44, i64 %38
  %46 = load float, float* %45, align 4
  %47 = fadd float %43, %46
  %48 = extractvalue { float*, float*, i64, [1 x i64], [1 x i64] } %13, 1
  %49 = getelementptr float, float* %48, i64 %38
  store float %47, float* %49, align 4
  br label %50

50:                                               ; preds = %39
  br label %51

51:                                               ; preds = %50
  %52 = add nuw i64 %28, 1
  br label %27
}

; Function Attrs: nounwind
declare i32 @__kmpc_global_thread_num(%0*) #1

; Function Attrs: nounwind
declare void @__kmpc_for_static_init_8u(%0*, i32, i32, i32*, i64*, i64*, i64*, i64, i64) #1

; Function Attrs: nounwind
declare void @__kmpc_for_static_fini(%0*, i32) #1

; Function Attrs: convergent nounwind
declare void @__kmpc_barrier(%0*, i32) #2

; Function Attrs: nounwind
declare !callback !1 void @__kmpc_fork_call(%0*, i32, void (i32*, i32*, ...)*, ...) #1

attributes #0 = { norecurse nounwind }
attributes #1 = { nounwind }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{!2}
!2 = !{i64 2, i64 -1, i64 -1, i1 true}
