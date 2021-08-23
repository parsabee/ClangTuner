; ModuleID = '/Users/parsabagheri/Development/llvm-project/tuner/Benchmarks/SPAPT/axpy/axpy.cpp'
source_filename = "/Users/parsabagheri/Development/llvm-project/tuner/Benchmarks/SPAPT/axpy/axpy.cpp"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

%0 = type { i32, i32, i32, i32, i8* }

@0 = private unnamed_addr constant %0 { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([123 x i8], [123 x i8]* @4, i32 0, i32 0) }, align 8
@1 = private unnamed_addr constant %0 { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([123 x i8], [123 x i8]* @3, i32 0, i32 0) }, align 8
@2 = private unnamed_addr constant %0 { i32 0, i32 66, i32 0, i32 0, i8* getelementptr inbounds ([123 x i8], [123 x i8]* @3, i32 0, i32 0) }, align 8
@3 = private unnamed_addr constant [123 x i8] c";LLVMDialectModule;__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_22_3;25;7;;\00", align 1
@4 = private unnamed_addr constant [123 x i8] c";LLVMDialectModule;__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_22_3;24;5;;\00", align 1

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define void @_Z4axpyfPfS_S_(float %0, float* %1, float* %2, float* %3) #0 {
  %5 = alloca float, align 4
  %6 = alloca float*, align 8
  %7 = alloca float*, align 8
  %8 = alloca float*, align 8
  %9 = alloca i64, align 8
  store float %0, float* %5, align 4
  store float* %1, float** %6, align 8
  store float* %2, float** %7, align 8
  store float* %3, float** %8, align 8
  store i64 0, i64* %9, align 8
  br label %10

10:                                               ; preds = %28, %4
  %11 = load i64, i64* %9, align 8
  %12 = icmp ult i64 %11, 256
  br i1 %12, label %13, label %31

13:                                               ; preds = %10
  %14 = load float, float* %5, align 4
  %15 = load float*, float** %6, align 8
  %16 = load i64, i64* %9, align 8
  %17 = getelementptr inbounds float, float* %15, i64 %16
  %18 = load float, float* %17, align 4
  %19 = fmul float %14, %18
  %20 = load float*, float** %7, align 8
  %21 = load i64, i64* %9, align 8
  %22 = getelementptr inbounds float, float* %20, i64 %21
  %23 = load float, float* %22, align 4
  %24 = fadd float %19, %23
  %25 = load float*, float** %8, align 8
  %26 = load i64, i64* %9, align 8
  %27 = getelementptr inbounds float, float* %25, i64 %26
  store float %24, float* %27, align 4
  br label %28

28:                                               ; preds = %13
  %29 = load i64, i64* %9, align 8
  %30 = add i64 %29, 1
  store i64 %30, i64* %9, align 8
  br label %10, !llvm.loop !6

31:                                               ; preds = %10
  ret void
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline norecurse nounwind optnone ssp uwtable mustprogress
define i32 @main() #2 {
  %1 = alloca i32, align 4
  %2 = alloca [256 x float], align 16
  %3 = alloca [256 x float], align 16
  %4 = alloca [256 x float], align 16
  %5 = alloca float, align 4
  store i32 0, i32* %1, align 4
  %6 = call i32 @rand()
  %7 = sitofp i32 %6 to float
  store float %7, float* %5, align 4
  %8 = getelementptr inbounds [256 x float], [256 x float]* %2, i64 0, i64 0
  call void @_Z19initializeRandom_1DIfLm256EEvPT_(float* %8)
  %9 = getelementptr inbounds [256 x float], [256 x float]* %3, i64 0, i64 0
  call void @_Z19initializeRandom_1DIfLm256EEvPT_(float* %9)
  %10 = load float, float* %5, align 4
  %11 = getelementptr inbounds [256 x float], [256 x float]* %2, i64 0, i64 0
  %12 = getelementptr inbounds [256 x float], [256 x float]* %3, i64 0, i64 0
  %13 = getelementptr inbounds [256 x float], [256 x float]* %4, i64 0, i64 0
  call void @_Z4axpyfPfS_S_(float %10, float* %11, float* %12, float* %13)
  %14 = load float, float* %5, align 4
  %15 = getelementptr inbounds [256 x float], [256 x float]* %2, i64 0, i64 0
  %16 = getelementptr inbounds [256 x float], [256 x float]* %3, i64 0, i64 0
  %17 = getelementptr inbounds [256 x float], [256 x float]* %4, i64 0, i64 0
  %18 = call zeroext i1 @_Z6verifyIfLm256EEbT_PS0_S1_S1_(float %14, float* %15, float* %16, float* %17)
  br i1 %18, label %20, label %19

19:                                               ; preds = %0
  store i32 1, i32* %1, align 4
  br label %21

20:                                               ; preds = %0
  store i32 0, i32* %1, align 4
  br label %21

21:                                               ; preds = %20, %19
  %22 = load i32, i32* %1, align 4
  ret i32 %22
}

declare i32 @rand() #3

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr void @_Z19initializeRandom_1DIfLm256EEvPT_(float* %0) #0 {
  %2 = alloca float*, align 8
  %3 = alloca i64, align 8
  store float* %0, float** %2, align 8
  %4 = call i64 @time(i64* null)
  %5 = trunc i64 %4 to i32
  call void @srand(i32 %5)
  store i64 0, i64* %3, align 8
  br label %6

6:                                                ; preds = %15, %1
  %7 = load i64, i64* %3, align 8
  %8 = icmp ult i64 %7, 256
  br i1 %8, label %9, label %18

9:                                                ; preds = %6
  %10 = call i32 @rand()
  %11 = sitofp i32 %10 to float
  %12 = load float*, float** %2, align 8
  %13 = load i64, i64* %3, align 8
  %14 = getelementptr inbounds float, float* %12, i64 %13
  store float %11, float* %14, align 4
  br label %15

15:                                               ; preds = %9
  %16 = load i64, i64* %3, align 8
  %17 = add i64 %16, 1
  store i64 %17, i64* %3, align 8
  br label %6, !llvm.loop !8

18:                                               ; preds = %6
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr zeroext i1 @_Z6verifyIfLm256EEbT_PS0_S1_S1_(float %0, float* %1, float* %2, float* %3) #0 {
  %5 = alloca i1, align 1
  %6 = alloca float, align 4
  %7 = alloca float*, align 8
  %8 = alloca float*, align 8
  %9 = alloca float*, align 8
  %10 = alloca [256 x float], align 16
  %11 = alloca i64, align 8
  %12 = alloca i64, align 8
  store float %0, float* %6, align 4
  store float* %1, float** %7, align 8
  store float* %2, float** %8, align 8
  store float* %3, float** %9, align 8
  store i64 0, i64* %11, align 8
  br label %13

13:                                               ; preds = %30, %4
  %14 = load i64, i64* %11, align 8
  %15 = icmp ult i64 %14, 256
  br i1 %15, label %16, label %33

16:                                               ; preds = %13
  %17 = load float, float* %6, align 4
  %18 = load float*, float** %7, align 8
  %19 = load i64, i64* %11, align 8
  %20 = getelementptr inbounds float, float* %18, i64 %19
  %21 = load float, float* %20, align 4
  %22 = fmul float %17, %21
  %23 = load float*, float** %8, align 8
  %24 = load i64, i64* %11, align 8
  %25 = getelementptr inbounds float, float* %23, i64 %24
  %26 = load float, float* %25, align 4
  %27 = fadd float %22, %26
  %28 = load i64, i64* %11, align 8
  %29 = getelementptr inbounds [256 x float], [256 x float]* %10, i64 0, i64 %28
  store float %27, float* %29, align 4
  br label %30

30:                                               ; preds = %16
  %31 = load i64, i64* %11, align 8
  %32 = add i64 %31, 1
  store i64 %32, i64* %11, align 8
  br label %13, !llvm.loop !9

33:                                               ; preds = %13
  store i64 0, i64* %12, align 8
  br label %34

34:                                               ; preds = %48, %33
  %35 = load i64, i64* %12, align 8
  %36 = icmp ult i64 %35, 256
  br i1 %36, label %37, label %51

37:                                               ; preds = %34
  %38 = load i64, i64* %12, align 8
  %39 = getelementptr inbounds [256 x float], [256 x float]* %10, i64 0, i64 %38
  %40 = load float, float* %39, align 4
  %41 = load float*, float** %9, align 8
  %42 = load i64, i64* %12, align 8
  %43 = getelementptr inbounds float, float* %41, i64 %42
  %44 = load float, float* %43, align 4
  %45 = fcmp une float %40, %44
  br i1 %45, label %46, label %47

46:                                               ; preds = %37
  store i1 false, i1* %5, align 1
  br label %52

47:                                               ; preds = %37
  br label %48

48:                                               ; preds = %47
  %49 = load i64, i64* %12, align 8
  %50 = add i64 %49, 1
  store i64 %50, i64* %12, align 8
  br label %34, !llvm.loop !10

51:                                               ; preds = %34
  store i1 true, i1* %5, align 1
  br label %52

52:                                               ; preds = %51, %46
  %53 = load i1, i1* %5, align 1
  ret i1 %53
}

declare void @srand(i32) #3

declare i64 @time(i64*) #3

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
  %36 = call i32 @__kmpc_global_thread_num(%0* @0)
  store { float*, float*, i64, [1 x i64], [1 x i64] } %30, { float*, float*, i64, [1 x i64], [1 x i64] }* %17, align 8
  store float %0, float* %18, align 4
  store { float*, float*, i64, [1 x i64], [1 x i64] } %35, { float*, float*, i64, [1 x i64], [1 x i64] }* %19, align 8
  store { float*, float*, i64, [1 x i64], [1 x i64] } %25, { float*, float*, i64, [1 x i64], [1 x i64] }* %20, align 8
  br label %37

37:                                               ; preds = %16
  call void (%0*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%0* @0, i32 4, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, { float*, float*, i64, [1 x i64], [1 x i64] }*, float*, { float*, float*, i64, [1 x i64], [1 x i64] }*, { float*, float*, i64, [1 x i64], [1 x i64] }*)* @5 to void (i32*, i32*, ...)*), { float*, float*, i64, [1 x i64], [1 x i64] }* %17, float* %18, { float*, float*, i64, [1 x i64], [1 x i64] }* %19, { float*, float*, i64, [1 x i64], [1 x i64] }* %20)
  br label %38

38:                                               ; preds = %37
  br label %39

39:                                               ; preds = %38
  ret void
}

; Function Attrs: nounwind
declare i32 @__kmpc_global_thread_num(%0*) #4

; Function Attrs: norecurse nounwind
define internal void @5(i32* noalias %0, i32* noalias %1, { float*, float*, i64, [1 x i64], [1 x i64] }* %2, float* %3, { float*, float*, i64, [1 x i64], [1 x i64] }* %4, { float*, float*, i64, [1 x i64], [1 x i64] }* %5) #5 {
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
  %22 = call i32 @__kmpc_global_thread_num(%0* @1)
  call void @__kmpc_for_static_init_8u(%0* @1, i32 %22, i32 34, i32* %17, i64* %18, i64* %19, i64* %20, i64 1, i64 1)
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
  call void @__kmpc_for_static_fini(%0* @1, i32 %22)
  %32 = call i32 @__kmpc_global_thread_num(%0* @1)
  call void @__kmpc_barrier(%0* @2, i32 %32)
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
declare !callback !11 void @__kmpc_fork_call(%0*, i32, void (i32*, i32*, ...)*, ...) #4

; Function Attrs: nounwind
declare void @__kmpc_for_static_init_8u(%0*, i32, i32, i32*, i64*, i64*, i64*, i64, i64) #4

; Function Attrs: nounwind
declare void @__kmpc_for_static_fini(%0*, i32) #4

; Function Attrs: convergent nounwind
declare void @__kmpc_barrier(%0*, i32) #6

attributes #0 = { noinline nounwind optnone ssp uwtable mustprogress "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { noinline norecurse nounwind optnone ssp uwtable mustprogress "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #3 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #4 = { nounwind }
attributes #5 = { norecurse nounwind }
attributes #6 = { convergent nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.linker.options = !{}
!llvm.ident = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 15]}
!1 = !{i32 7, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"PIC Level", i32 2}
!5 = !{!"clang version 13.0.0 (https://github.com/parsabee/llvm-project.git f12c995b2adcdb8f69500a45366a12c2aa5f0db6)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !7}
!10 = distinct !{!10, !7}
!11 = !{!12}
!12 = !{i64 2, i64 -1, i64 -1, i1 true}
