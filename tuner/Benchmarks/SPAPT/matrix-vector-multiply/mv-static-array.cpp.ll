; ModuleID = '/Users/parsabagheri/Development/llvm-project/tuner/Benchmarks/SPAPT/matrix-vector-multiply/mv-static-array.cpp'
source_filename = "/Users/parsabagheri/Development/llvm-project/tuner/Benchmarks/SPAPT/matrix-vector-multiply/mv-static-array.cpp"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

%0 = type { i32, i32, i32, i32, i8* }

@0 = private unnamed_addr constant %0 { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([152 x i8], [152 x i8]* @4, i32 0, i32 0) }, align 8
@1 = private unnamed_addr constant %0 { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([152 x i8], [152 x i8]* @3, i32 0, i32 0) }, align 8
@2 = private unnamed_addr constant %0 { i32 0, i32 66, i32 0, i32 0, i8* getelementptr inbounds ([152 x i8], [152 x i8]* @3, i32 0, i32 0) }, align 8
@3 = private unnamed_addr constant [152 x i8] c";LLVMDialectModule;__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_matrix_vector_multiply_mv_static_array_cpp_30_3;27;7;;\00", align 1
@4 = private unnamed_addr constant [152 x i8] c";LLVMDialectModule;__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_matrix_vector_multiply_mv_static_array_cpp_30_3;26;5;;\00", align 1

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define void @_Z12mat_vec_multPA256_fPfS1_([256 x float]* %0, float* %1, float* %2) #0 {
  %4 = alloca [256 x float]*, align 8
  %5 = alloca float*, align 8
  %6 = alloca float*, align 8
  %7 = alloca i64, align 8
  %8 = alloca i64, align 8
  store [256 x float]* %0, [256 x float]** %4, align 8
  store float* %1, float** %5, align 8
  store float* %2, float** %6, align 8
  store i64 0, i64* %7, align 8
  br label %9

9:                                                ; preds = %37, %3
  %10 = load i64, i64* %7, align 8
  %11 = icmp ult i64 %10, 256
  br i1 %11, label %12, label %40

12:                                               ; preds = %9
  store i64 0, i64* %8, align 8
  br label %13

13:                                               ; preds = %33, %12
  %14 = load i64, i64* %8, align 8
  %15 = icmp ult i64 %14, 256
  br i1 %15, label %16, label %36

16:                                               ; preds = %13
  %17 = load [256 x float]*, [256 x float]** %4, align 8
  %18 = load i64, i64* %7, align 8
  %19 = getelementptr inbounds [256 x float], [256 x float]* %17, i64 %18
  %20 = load i64, i64* %8, align 8
  %21 = getelementptr inbounds [256 x float], [256 x float]* %19, i64 0, i64 %20
  %22 = load float, float* %21, align 4
  %23 = load float*, float** %5, align 8
  %24 = load i64, i64* %8, align 8
  %25 = getelementptr inbounds float, float* %23, i64 %24
  %26 = load float, float* %25, align 4
  %27 = fmul float %22, %26
  %28 = load float*, float** %6, align 8
  %29 = load i64, i64* %7, align 8
  %30 = getelementptr inbounds float, float* %28, i64 %29
  %31 = load float, float* %30, align 4
  %32 = fadd float %31, %27
  store float %32, float* %30, align 4
  br label %33

33:                                               ; preds = %16
  %34 = load i64, i64* %8, align 8
  %35 = add i64 %34, 1
  store i64 %35, i64* %8, align 8
  br label %13, !llvm.loop !6

36:                                               ; preds = %13
  br label %37

37:                                               ; preds = %36
  %38 = load i64, i64* %7, align 8
  %39 = add i64 %38, 1
  store i64 %39, i64* %7, align 8
  br label %9, !llvm.loop !8

40:                                               ; preds = %9
  ret void
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline norecurse nounwind optnone ssp uwtable mustprogress
define i32 @main() #2 {
  %1 = alloca i32, align 4
  %2 = alloca [256 x [256 x float]], align 16
  %3 = alloca [256 x float], align 16
  %4 = alloca [256 x float], align 16
  store i32 0, i32* %1, align 4
  %5 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* %2, i64 0, i64 0
  call void @_Z19initializeRandom_2DIfLm256ELm256EEvPAT1__T_([256 x float]* %5)
  %6 = getelementptr inbounds [256 x float], [256 x float]* %3, i64 0, i64 0
  call void @_Z19initializeRandom_1DIfLm256EEvPT_(float* %6)
  %7 = getelementptr inbounds [256 x float], [256 x float]* %4, i64 0, i64 0
  call void @_Z13initialize_1DIfLm256EEvPT_S0_(float* %7, float 0.000000e+00)
  %8 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* %2, i64 0, i64 0
  %9 = getelementptr inbounds [256 x float], [256 x float]* %3, i64 0, i64 0
  %10 = getelementptr inbounds [256 x float], [256 x float]* %4, i64 0, i64 0
  call void @_Z12mat_vec_multPA256_fPfS1_([256 x float]* %8, float* %9, float* %10)
  %11 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* %2, i64 0, i64 0
  %12 = getelementptr inbounds [256 x float], [256 x float]* %3, i64 0, i64 0
  %13 = getelementptr inbounds [256 x float], [256 x float]* %4, i64 0, i64 0
  %14 = call zeroext i1 @_Z6verifyIfLm256ELm256EEbPAT1__T_PS0_S3_([256 x float]* %11, float* %12, float* %13)
  br i1 %14, label %16, label %15

15:                                               ; preds = %0
  store i32 1, i32* %1, align 4
  br label %17

16:                                               ; preds = %0
  store i32 0, i32* %1, align 4
  br label %17

17:                                               ; preds = %16, %15
  %18 = load i32, i32* %1, align 4
  ret i32 %18
}

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr void @_Z19initializeRandom_2DIfLm256ELm256EEvPAT1__T_([256 x float]* %0) #0 {
  %2 = alloca [256 x float]*, align 8
  %3 = alloca i64, align 8
  %4 = alloca i64, align 8
  store [256 x float]* %0, [256 x float]** %2, align 8
  %5 = call i64 @time(i64* null)
  %6 = trunc i64 %5 to i32
  call void @srand(i32 %6)
  store i64 0, i64* %3, align 8
  br label %7

7:                                                ; preds = %26, %1
  %8 = load i64, i64* %3, align 8
  %9 = icmp ult i64 %8, 256
  br i1 %9, label %10, label %29

10:                                               ; preds = %7
  store i64 0, i64* %4, align 8
  br label %11

11:                                               ; preds = %22, %10
  %12 = load i64, i64* %4, align 8
  %13 = icmp ult i64 %12, 256
  br i1 %13, label %14, label %25

14:                                               ; preds = %11
  %15 = call i32 @rand()
  %16 = sitofp i32 %15 to float
  %17 = load [256 x float]*, [256 x float]** %2, align 8
  %18 = load i64, i64* %3, align 8
  %19 = getelementptr inbounds [256 x float], [256 x float]* %17, i64 %18
  %20 = load i64, i64* %4, align 8
  %21 = getelementptr inbounds [256 x float], [256 x float]* %19, i64 0, i64 %20
  store float %16, float* %21, align 4
  br label %22

22:                                               ; preds = %14
  %23 = load i64, i64* %4, align 8
  %24 = add i64 %23, 1
  store i64 %24, i64* %4, align 8
  br label %11, !llvm.loop !9

25:                                               ; preds = %11
  br label %26

26:                                               ; preds = %25
  %27 = load i64, i64* %3, align 8
  %28 = add i64 %27, 1
  store i64 %28, i64* %3, align 8
  br label %7, !llvm.loop !10

29:                                               ; preds = %7
  ret void
}

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
  br label %6, !llvm.loop !11

18:                                               ; preds = %6
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr void @_Z13initialize_1DIfLm256EEvPT_S0_(float* %0, float %1) #0 {
  %3 = alloca float*, align 8
  %4 = alloca float, align 4
  %5 = alloca i64, align 8
  store float* %0, float** %3, align 8
  store float %1, float* %4, align 4
  store i64 0, i64* %5, align 8
  br label %6

6:                                                ; preds = %14, %2
  %7 = load i64, i64* %5, align 8
  %8 = icmp ult i64 %7, 256
  br i1 %8, label %9, label %17

9:                                                ; preds = %6
  %10 = load float, float* %4, align 4
  %11 = load float*, float** %3, align 8
  %12 = load i64, i64* %5, align 8
  %13 = getelementptr inbounds float, float* %11, i64 %12
  store float %10, float* %13, align 4
  br label %14

14:                                               ; preds = %9
  %15 = load i64, i64* %5, align 8
  %16 = add i64 %15, 1
  store i64 %16, i64* %5, align 8
  br label %6, !llvm.loop !12

17:                                               ; preds = %6
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr zeroext i1 @_Z6verifyIfLm256ELm256EEbPAT1__T_PS0_S3_([256 x float]* %0, float* %1, float* %2) #0 {
  %4 = alloca i1, align 1
  %5 = alloca [256 x float]*, align 8
  %6 = alloca float*, align 8
  %7 = alloca float*, align 8
  %8 = alloca [256 x float], align 16
  %9 = alloca i64, align 8
  %10 = alloca i64, align 8
  %11 = alloca i64, align 8
  %12 = alloca i64, align 8
  store [256 x float]* %0, [256 x float]** %5, align 8
  store float* %1, float** %6, align 8
  store float* %2, float** %7, align 8
  store i64 0, i64* %9, align 8
  br label %13

13:                                               ; preds = %19, %3
  %14 = load i64, i64* %9, align 8
  %15 = icmp ult i64 %14, 256
  br i1 %15, label %16, label %22

16:                                               ; preds = %13
  %17 = load i64, i64* %9, align 8
  %18 = getelementptr inbounds [256 x float], [256 x float]* %8, i64 0, i64 %17
  store float 0.000000e+00, float* %18, align 4
  br label %19

19:                                               ; preds = %16
  %20 = load i64, i64* %9, align 8
  %21 = add i64 %20, 1
  store i64 %21, i64* %9, align 8
  br label %13, !llvm.loop !13

22:                                               ; preds = %13
  store i64 0, i64* %10, align 8
  br label %23

23:                                               ; preds = %50, %22
  %24 = load i64, i64* %10, align 8
  %25 = icmp ult i64 %24, 256
  br i1 %25, label %26, label %53

26:                                               ; preds = %23
  store i64 0, i64* %11, align 8
  br label %27

27:                                               ; preds = %46, %26
  %28 = load i64, i64* %11, align 8
  %29 = icmp ult i64 %28, 256
  br i1 %29, label %30, label %49

30:                                               ; preds = %27
  %31 = load [256 x float]*, [256 x float]** %5, align 8
  %32 = load i64, i64* %10, align 8
  %33 = getelementptr inbounds [256 x float], [256 x float]* %31, i64 %32
  %34 = load i64, i64* %11, align 8
  %35 = getelementptr inbounds [256 x float], [256 x float]* %33, i64 0, i64 %34
  %36 = load float, float* %35, align 4
  %37 = load float*, float** %6, align 8
  %38 = load i64, i64* %11, align 8
  %39 = getelementptr inbounds float, float* %37, i64 %38
  %40 = load float, float* %39, align 4
  %41 = fmul float %36, %40
  %42 = load i64, i64* %10, align 8
  %43 = getelementptr inbounds [256 x float], [256 x float]* %8, i64 0, i64 %42
  %44 = load float, float* %43, align 4
  %45 = fadd float %44, %41
  store float %45, float* %43, align 4
  br label %46

46:                                               ; preds = %30
  %47 = load i64, i64* %11, align 8
  %48 = add i64 %47, 1
  store i64 %48, i64* %11, align 8
  br label %27, !llvm.loop !14

49:                                               ; preds = %27
  br label %50

50:                                               ; preds = %49
  %51 = load i64, i64* %10, align 8
  %52 = add i64 %51, 1
  store i64 %52, i64* %10, align 8
  br label %23, !llvm.loop !15

53:                                               ; preds = %23
  store i64 0, i64* %12, align 8
  br label %54

54:                                               ; preds = %68, %53
  %55 = load i64, i64* %12, align 8
  %56 = icmp ult i64 %55, 256
  br i1 %56, label %57, label %71

57:                                               ; preds = %54
  %58 = load i64, i64* %12, align 8
  %59 = getelementptr inbounds [256 x float], [256 x float]* %8, i64 0, i64 %58
  %60 = load float, float* %59, align 4
  %61 = load float*, float** %7, align 8
  %62 = load i64, i64* %12, align 8
  %63 = getelementptr inbounds float, float* %61, i64 %62
  %64 = load float, float* %63, align 4
  %65 = fcmp une float %60, %64
  br i1 %65, label %66, label %67

66:                                               ; preds = %57
  store i1 false, i1* %4, align 1
  br label %72

67:                                               ; preds = %57
  br label %68

68:                                               ; preds = %67
  %69 = load i64, i64* %12, align 8
  %70 = add i64 %69, 1
  store i64 %70, i64* %12, align 8
  br label %54, !llvm.loop !16

71:                                               ; preds = %54
  store i1 true, i1* %4, align 1
  br label %72

72:                                               ; preds = %71, %66
  %73 = load i1, i1* %4, align 1
  ret i1 %73
}

declare void @srand(i32) #3

declare i64 @time(i64*) #3

declare i32 @rand() #3

define void @__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_matrix_vector_multiply_mv_static_array_cpp_30_3(float* %0, float* %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, float* %7, float* %8, i64 %9, i64 %10, i64 %11, float* %12, float* %13, i64 %14, i64 %15, i64 %16) {
  %18 = alloca { float*, float*, i64, [2 x i64], [2 x i64] }, align 8
  %19 = alloca { float*, float*, i64, [1 x i64], [1 x i64] }, align 8
  %20 = alloca { float*, float*, i64, [1 x i64], [1 x i64] }, align 8
  %21 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } undef, float* %0, 0
  %22 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %21, float* %1, 1
  %23 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %22, i64 %2, 2
  %24 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %23, i64 %3, 3, 0
  %25 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %24, i64 %5, 4, 0
  %26 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %25, i64 %4, 3, 1
  %27 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %26, i64 %6, 4, 1
  %28 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } undef, float* %7, 0
  %29 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %28, float* %8, 1
  %30 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %29, i64 %9, 2
  %31 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %30, i64 %10, 3, 0
  %32 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %31, i64 %11, 4, 0
  %33 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } undef, float* %12, 0
  %34 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %33, float* %13, 1
  %35 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %34, i64 %14, 2
  %36 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %35, i64 %15, 3, 0
  %37 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %36, i64 %16, 4, 0
  %38 = call i32 @__kmpc_global_thread_num(%0* @0)
  store { float*, float*, i64, [2 x i64], [2 x i64] } %27, { float*, float*, i64, [2 x i64], [2 x i64] }* %18, align 8
  store { float*, float*, i64, [1 x i64], [1 x i64] } %32, { float*, float*, i64, [1 x i64], [1 x i64] }* %19, align 8
  store { float*, float*, i64, [1 x i64], [1 x i64] } %37, { float*, float*, i64, [1 x i64], [1 x i64] }* %20, align 8
  br label %39

39:                                               ; preds = %17
  call void (%0*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%0* @0, i32 3, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, { float*, float*, i64, [2 x i64], [2 x i64] }*, { float*, float*, i64, [1 x i64], [1 x i64] }*, { float*, float*, i64, [1 x i64], [1 x i64] }*)* @5 to void (i32*, i32*, ...)*), { float*, float*, i64, [2 x i64], [2 x i64] }* %18, { float*, float*, i64, [1 x i64], [1 x i64] }* %19, { float*, float*, i64, [1 x i64], [1 x i64] }* %20)
  br label %40

40:                                               ; preds = %39
  br label %41

41:                                               ; preds = %40
  ret void
}

; Function Attrs: nounwind
declare i32 @__kmpc_global_thread_num(%0*) #4

; Function Attrs: norecurse nounwind
define internal void @5(i32* noalias %0, i32* noalias %1, { float*, float*, i64, [2 x i64], [2 x i64] }* %2, { float*, float*, i64, [1 x i64], [1 x i64] }* %3, { float*, float*, i64, [1 x i64], [1 x i64] }* %4) #5 {
  %6 = alloca i32, align 4
  %7 = load i32, i32* %0, align 4
  store i32 %7, i32* %6, align 4
  %8 = load i32, i32* %6, align 4
  %9 = load { float*, float*, i64, [2 x i64], [2 x i64] }, { float*, float*, i64, [2 x i64], [2 x i64] }* %2, align 8
  %10 = load { float*, float*, i64, [1 x i64], [1 x i64] }, { float*, float*, i64, [1 x i64], [1 x i64] }* %3, align 8
  %11 = load { float*, float*, i64, [1 x i64], [1 x i64] }, { float*, float*, i64, [1 x i64], [1 x i64] }* %4, align 8
  br label %13

12:                                               ; preds = %32
  ret void

13:                                               ; preds = %5
  br label %14

14:                                               ; preds = %13
  %15 = alloca i32, align 4
  %16 = alloca i64, align 8
  %17 = alloca i64, align 8
  %18 = alloca i64, align 8
  br label %19

19:                                               ; preds = %14
  store i64 0, i64* %16, align 4
  store i64 255, i64* %17, align 4
  store i64 1, i64* %18, align 4
  %20 = call i32 @__kmpc_global_thread_num(%0* @1)
  call void @__kmpc_for_static_init_8u(%0* @1, i32 %20, i32 34, i32* %15, i64* %16, i64* %17, i64* %18, i64 1, i64 1)
  %21 = load i64, i64* %16, align 4
  %22 = load i64, i64* %17, align 4
  %23 = sub i64 %22, %21
  %24 = add i64 %23, 1
  br label %25

25:                                               ; preds = %43, %19
  %26 = phi i64 [ 0, %19 ], [ %44, %43 ]
  br label %27

27:                                               ; preds = %25
  %28 = icmp ult i64 %26, %24
  br i1 %28, label %33, label %29

29:                                               ; preds = %27
  call void @__kmpc_for_static_fini(%0* @1, i32 %20)
  %30 = call i32 @__kmpc_global_thread_num(%0* @1)
  call void @__kmpc_barrier(%0* @2, i32 %30)
  br label %31

31:                                               ; preds = %29
  br label %32

32:                                               ; preds = %31
  br label %12

33:                                               ; preds = %27
  %34 = add i64 %26, %21
  %35 = mul i64 %34, 1
  %36 = add i64 %35, 0
  br label %37

37:                                               ; preds = %33
  br label %38

38:                                               ; preds = %45, %37
  %39 = phi i64 [ %57, %45 ], [ 0, %37 ]
  %40 = icmp slt i64 %39, 256
  br i1 %40, label %45, label %41

41:                                               ; preds = %38
  br label %42

42:                                               ; preds = %41
  br label %43

43:                                               ; preds = %42
  %44 = add nuw i64 %26, 1
  br label %25

45:                                               ; preds = %38
  %46 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %9, 1
  %47 = mul i64 %39, 256
  %48 = add i64 %47, %36
  %49 = getelementptr float, float* %46, i64 %48
  %50 = load float, float* %49, align 4
  %51 = extractvalue { float*, float*, i64, [1 x i64], [1 x i64] } %10, 1
  %52 = getelementptr float, float* %51, i64 %39
  %53 = load float, float* %52, align 4
  %54 = fmul float %50, %53
  %55 = extractvalue { float*, float*, i64, [1 x i64], [1 x i64] } %11, 1
  %56 = getelementptr float, float* %55, i64 %36
  store float %54, float* %56, align 4
  %57 = add i64 %39, 1
  br label %38
}

; Function Attrs: nounwind
declare !callback !17 void @__kmpc_fork_call(%0*, i32, void (i32*, i32*, ...)*, ...) #4

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
!5 = !{!"clang version 13.0.0 (https://github.com/parsabee/llvm-project.git 71acce2142bff5fbe1745038ad2ff5ae63e93a45)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !7}
!10 = distinct !{!10, !7}
!11 = distinct !{!11, !7}
!12 = distinct !{!12, !7}
!13 = distinct !{!13, !7}
!14 = distinct !{!14, !7}
!15 = distinct !{!15, !7}
!16 = distinct !{!16, !7}
!17 = !{!18}
!18 = !{i64 2, i64 -1, i64 -1, i1 true}
