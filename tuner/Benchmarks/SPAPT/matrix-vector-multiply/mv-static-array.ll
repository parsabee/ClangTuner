; ModuleID = 'mv-static-array.cpp'
source_filename = "mv-static-array.cpp"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

%"class.std::__1::basic_ostream" = type { i32 (...)**, %"class.std::__1::basic_ios.base" }
%"class.std::__1::basic_ios.base" = type <{ %"class.std::__1::ios_base", %"class.std::__1::basic_ostream"*, i32 }>
%"class.std::__1::ios_base" = type { i32 (...)**, i32, i64, i64, i32, i32, i8*, i8*, void (i32, %"class.std::__1::ios_base"*, i32)**, i32*, i64, i64, i64*, i64, i64, i8**, i64, i64 }
%"class.std::__1::locale::id" = type <{ %"struct.std::__1::once_flag", i32, [4 x i8] }>
%"struct.std::__1::once_flag" = type { i64 }
%"class.std::__1::basic_ostream<char>::sentry" = type { i8, %"class.std::__1::basic_ostream"* }
%"class.std::__1::ostreambuf_iterator" = type { %"class.std::__1::basic_streambuf"* }
%"class.std::__1::basic_streambuf" = type { i32 (...)**, %"class.std::__1::locale", i8*, i8*, i8*, i8*, i8*, i8* }
%"class.std::__1::locale" = type { %"class.std::__1::locale::__imp"* }
%"class.std::__1::locale::__imp" = type opaque
%"class.std::__1::basic_ios" = type <{ %"class.std::__1::ios_base", %"class.std::__1::basic_ostream"*, i32, [4 x i8] }>
%"class.std::__1::basic_string" = type { %"class.std::__1::__compressed_pair" }
%"class.std::__1::__compressed_pair" = type { %"struct.std::__1::__compressed_pair_elem" }
%"struct.std::__1::__compressed_pair_elem" = type { %"struct.std::__1::basic_string<char>::__rep" }
%"struct.std::__1::basic_string<char>::__rep" = type { %union.anon }
%union.anon = type { %"struct.std::__1::basic_string<char>::__long" }
%"struct.std::__1::basic_string<char>::__long" = type { i64, i64, i8* }
%"struct.std::__1::__default_init_tag" = type { i8 }
%"class.std::__1::__basic_string_common" = type { i8 }
%"struct.std::__1::__compressed_pair_elem.1" = type { i8 }
%"class.std::__1::allocator" = type { i8 }
%"struct.std::__1::basic_string<char>::__short" = type { %union.anon.0, [23 x i8] }
%union.anon.0 = type { i8 }
%"struct.std::__1::iterator" = type { i8 }
%"class.std::__1::ctype" = type <{ %"class.std::__1::locale::facet", i32*, i8, [7 x i8] }>
%"class.std::__1::locale::facet" = type { %"class.std::__1::__shared_count" }
%"class.std::__1::__shared_count" = type { i32 (...)**, i64 }

@_ZNSt3__14coutE = external global %"class.std::__1::basic_ostream", align 8
@.str = private unnamed_addr constant [3 x i8] c", \00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@_ZNSt3__15ctypeIcE2idE = external global %"class.std::__1::locale::id", align 8

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define dso_local void @_Z12mat_vec_multPA1024_fPfS1_([1024 x float]* %0, float* %1, float* %2) #0 {
  %4 = alloca [1024 x float]*, align 8
  %5 = alloca float*, align 8
  %6 = alloca float*, align 8
  %7 = alloca float*, align 8
  %8 = alloca float*, align 8
  %9 = alloca float*, align 8
  %10 = alloca i64, align 8
  %11 = alloca i64, align 8
  store [1024 x float]* %0, [1024 x float]** %4, align 8
  store float* %1, float** %5, align 8
  store float* %2, float** %6, align 8
  %12 = load [1024 x float]*, [1024 x float]** %4, align 8
  %13 = bitcast [1024 x float]* %12 to float*
  store float* %13, float** %7, align 8
  %14 = load float*, float** %5, align 8
  store float* %14, float** %8, align 8
  %15 = load float*, float** %6, align 8
  store float* %15, float** %9, align 8
  store i64 0, i64* %10, align 8
  br label %16

16:                                               ; preds = %45, %3
  %17 = load i64, i64* %10, align 8
  %18 = icmp ult i64 %17, 1024
  br i1 %18, label %19, label %48

19:                                               ; preds = %16
  store i64 0, i64* %11, align 8
  br label %20

20:                                               ; preds = %41, %19
  %21 = load i64, i64* %11, align 8
  %22 = icmp ult i64 %21, 1024
  br i1 %22, label %23, label %44

23:                                               ; preds = %20
  %24 = load float*, float** %7, align 8
  %25 = load i64, i64* %10, align 8
  %26 = mul i64 %25, 1024
  %27 = load i64, i64* %11, align 8
  %28 = add i64 %26, %27
  %29 = getelementptr inbounds float, float* %24, i64 %28
  %30 = load float, float* %29, align 4
  %31 = load float*, float** %8, align 8
  %32 = load i64, i64* %11, align 8
  %33 = getelementptr inbounds float, float* %31, i64 %32
  %34 = load float, float* %33, align 4
  %35 = fmul float %30, %34
  %36 = load float*, float** %9, align 8
  %37 = load i64, i64* %10, align 8
  %38 = getelementptr inbounds float, float* %36, i64 %37
  %39 = load float, float* %38, align 4
  %40 = fadd float %39, %35
  store float %40, float* %38, align 4
  br label %41

41:                                               ; preds = %23
  %42 = load i64, i64* %11, align 8
  %43 = add i64 %42, 1
  store i64 %43, i64* %11, align 8
  br label %20, !llvm.loop !3

44:                                               ; preds = %20
  br label %45

45:                                               ; preds = %44
  %46 = load i64, i64* %10, align 8
  %47 = add i64 %46, 1
  store i64 %47, i64* %10, align 8
  br label %16, !llvm.loop !5

48:                                               ; preds = %16
  ret void
}

; Function Attrs: noinline norecurse optnone ssp uwtable mustprogress
define dso_local i32 @main() #1 {
  %1 = alloca i32, align 4
  %2 = alloca [1024 x [1024 x float]], align 16
  %3 = alloca [1024 x float], align 16
  %4 = alloca [1024 x float], align 16
  store i32 0, i32* %1, align 4
  %5 = getelementptr inbounds [1024 x [1024 x float]], [1024 x [1024 x float]]* %2, i64 0, i64 0
  call void @_Z19initializeRandom_2DIfLm1024ELm1024EEvPAT1__T_([1024 x float]* %5)
  %6 = getelementptr inbounds [1024 x float], [1024 x float]* %3, i64 0, i64 0
  call void @_Z19initializeRandom_1DIfLm1024EEvPT_(float* %6)
  %7 = getelementptr inbounds [1024 x float], [1024 x float]* %4, i64 0, i64 0
  call void @_Z13initialize_1DIfLm1024EEvPT_S0_(float* %7, float 0.000000e+00)
  %8 = getelementptr inbounds [1024 x [1024 x float]], [1024 x [1024 x float]]* %2, i64 0, i64 0
  %9 = getelementptr inbounds [1024 x float], [1024 x float]* %3, i64 0, i64 0
  %10 = getelementptr inbounds [1024 x float], [1024 x float]* %4, i64 0, i64 0
  call void @_Z12mat_vec_multPA1024_fPfS1_([1024 x float]* %8, float* %9, float* %10)
  %11 = getelementptr inbounds [1024 x [1024 x float]], [1024 x [1024 x float]]* %2, i64 0, i64 0
  %12 = getelementptr inbounds [1024 x float], [1024 x float]* %3, i64 0, i64 0
  %13 = getelementptr inbounds [1024 x float], [1024 x float]* %4, i64 0, i64 0
  %14 = call zeroext i1 @_Z6verifyIfLm1024ELm1024EEbPAT1__T_PS0_S3_([1024 x float]* %11, float* %12, float* %13)
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

; Function Attrs: noinline optnone ssp uwtable mustprogress
define linkonce_odr void @_Z19initializeRandom_2DIfLm1024ELm1024EEvPAT1__T_([1024 x float]* %0) #2 {
  %2 = alloca [1024 x float]*, align 8
  %3 = alloca i64, align 8
  %4 = alloca i64, align 8
  store [1024 x float]* %0, [1024 x float]** %2, align 8
  %5 = call i64 @time(i64* null)
  %6 = trunc i64 %5 to i32
  call void @srand(i32 %6)
  store i64 0, i64* %3, align 8
  br label %7

7:                                                ; preds = %26, %1
  %8 = load i64, i64* %3, align 8
  %9 = icmp ult i64 %8, 1024
  br i1 %9, label %10, label %29

10:                                               ; preds = %7
  store i64 0, i64* %4, align 8
  br label %11

11:                                               ; preds = %22, %10
  %12 = load i64, i64* %4, align 8
  %13 = icmp ult i64 %12, 1024
  br i1 %13, label %14, label %25

14:                                               ; preds = %11
  %15 = call i32 @rand()
  %16 = sitofp i32 %15 to float
  %17 = load [1024 x float]*, [1024 x float]** %2, align 8
  %18 = load i64, i64* %3, align 8
  %19 = getelementptr inbounds [1024 x float], [1024 x float]* %17, i64 %18
  %20 = load i64, i64* %4, align 8
  %21 = getelementptr inbounds [1024 x float], [1024 x float]* %19, i64 0, i64 %20
  store float %16, float* %21, align 4
  br label %22

22:                                               ; preds = %14
  %23 = load i64, i64* %4, align 8
  %24 = add i64 %23, 1
  store i64 %24, i64* %4, align 8
  br label %11, !llvm.loop !6

25:                                               ; preds = %11
  br label %26

26:                                               ; preds = %25
  %27 = load i64, i64* %3, align 8
  %28 = add i64 %27, 1
  store i64 %28, i64* %3, align 8
  br label %7, !llvm.loop !7

29:                                               ; preds = %7
  ret void
}

; Function Attrs: noinline optnone ssp uwtable mustprogress
define linkonce_odr void @_Z19initializeRandom_1DIfLm1024EEvPT_(float* %0) #2 {
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
  %8 = icmp ult i64 %7, 1024
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
define linkonce_odr void @_Z13initialize_1DIfLm1024EEvPT_S0_(float* %0, float %1) #0 {
  %3 = alloca float*, align 8
  %4 = alloca float, align 4
  %5 = alloca i64, align 8
  store float* %0, float** %3, align 8
  store float %1, float* %4, align 4
  store i64 0, i64* %5, align 8
  br label %6

6:                                                ; preds = %14, %2
  %7 = load i64, i64* %5, align 8
  %8 = icmp ult i64 %7, 1024
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
  br label %6, !llvm.loop !9

17:                                               ; preds = %6
  ret void
}

; Function Attrs: noinline optnone ssp uwtable mustprogress
define linkonce_odr zeroext i1 @_Z6verifyIfLm1024ELm1024EEbPAT1__T_PS0_S3_([1024 x float]* %0, float* %1, float* %2) #2 {
  %4 = alloca i1, align 1
  %5 = alloca [1024 x float]*, align 8
  %6 = alloca float*, align 8
  %7 = alloca float*, align 8
  %8 = alloca [1024 x float], align 16
  %9 = alloca i64, align 8
  %10 = alloca i64, align 8
  %11 = alloca i64, align 8
  %12 = alloca i64, align 8
  store [1024 x float]* %0, [1024 x float]** %5, align 8
  store float* %1, float** %6, align 8
  store float* %2, float** %7, align 8
  store i64 0, i64* %9, align 8
  br label %13

13:                                               ; preds = %19, %3
  %14 = load i64, i64* %9, align 8
  %15 = icmp ult i64 %14, 1024
  br i1 %15, label %16, label %22

16:                                               ; preds = %13
  %17 = load i64, i64* %9, align 8
  %18 = getelementptr inbounds [1024 x float], [1024 x float]* %8, i64 0, i64 %17
  store float 0.000000e+00, float* %18, align 4
  br label %19

19:                                               ; preds = %16
  %20 = load i64, i64* %9, align 8
  %21 = add i64 %20, 1
  store i64 %21, i64* %9, align 8
  br label %13, !llvm.loop !10

22:                                               ; preds = %13
  store i64 0, i64* %10, align 8
  br label %23

23:                                               ; preds = %50, %22
  %24 = load i64, i64* %10, align 8
  %25 = icmp ult i64 %24, 1024
  br i1 %25, label %26, label %53

26:                                               ; preds = %23
  store i64 0, i64* %11, align 8
  br label %27

27:                                               ; preds = %46, %26
  %28 = load i64, i64* %11, align 8
  %29 = icmp ult i64 %28, 1024
  br i1 %29, label %30, label %49

30:                                               ; preds = %27
  %31 = load [1024 x float]*, [1024 x float]** %5, align 8
  %32 = load i64, i64* %10, align 8
  %33 = getelementptr inbounds [1024 x float], [1024 x float]* %31, i64 %32
  %34 = load i64, i64* %11, align 8
  %35 = getelementptr inbounds [1024 x float], [1024 x float]* %33, i64 0, i64 %34
  %36 = load float, float* %35, align 4
  %37 = load float*, float** %6, align 8
  %38 = load i64, i64* %11, align 8
  %39 = getelementptr inbounds float, float* %37, i64 %38
  %40 = load float, float* %39, align 4
  %41 = fmul float %36, %40
  %42 = load i64, i64* %10, align 8
  %43 = getelementptr inbounds [1024 x float], [1024 x float]* %8, i64 0, i64 %42
  %44 = load float, float* %43, align 4
  %45 = fadd float %44, %41
  store float %45, float* %43, align 4
  br label %46

46:                                               ; preds = %30
  %47 = load i64, i64* %11, align 8
  %48 = add i64 %47, 1
  store i64 %48, i64* %11, align 8
  br label %27, !llvm.loop !11

49:                                               ; preds = %27
  br label %50

50:                                               ; preds = %49
  %51 = load i64, i64* %10, align 8
  %52 = add i64 %51, 1
  store i64 %52, i64* %10, align 8
  br label %23, !llvm.loop !12

53:                                               ; preds = %23
  store i64 0, i64* %12, align 8
  br label %54

54:                                               ; preds = %79, %53
  %55 = load i64, i64* %12, align 8
  %56 = icmp ult i64 %55, 1024
  br i1 %56, label %57, label %82

57:                                               ; preds = %54
  %58 = load i64, i64* %12, align 8
  %59 = getelementptr inbounds [1024 x float], [1024 x float]* %8, i64 0, i64 %58
  %60 = load float, float* %59, align 4
  %61 = call nonnull align 8 dereferenceable(8) %"class.std::__1::basic_ostream"* @_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEElsEf(%"class.std::__1::basic_ostream"* nonnull dereferenceable(8) @_ZNSt3__14coutE, float %60)
  %62 = call nonnull align 8 dereferenceable(8) %"class.std::__1::basic_ostream"* @_ZNSt3__1lsINS_11char_traitsIcEEEERNS_13basic_ostreamIcT_EES6_PKc(%"class.std::__1::basic_ostream"* nonnull align 8 dereferenceable(8) %61, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0))
  %63 = load float*, float** %7, align 8
  %64 = load i64, i64* %12, align 8
  %65 = getelementptr inbounds float, float* %63, i64 %64
  %66 = load float, float* %65, align 4
  %67 = call nonnull align 8 dereferenceable(8) %"class.std::__1::basic_ostream"* @_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEElsEf(%"class.std::__1::basic_ostream"* nonnull dereferenceable(8) %62, float %66)
  %68 = call nonnull align 8 dereferenceable(8) %"class.std::__1::basic_ostream"* @_ZNSt3__1lsINS_11char_traitsIcEEEERNS_13basic_ostreamIcT_EES6_PKc(%"class.std::__1::basic_ostream"* nonnull align 8 dereferenceable(8) %67, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0))
  %69 = load i64, i64* %12, align 8
  %70 = getelementptr inbounds [1024 x float], [1024 x float]* %8, i64 0, i64 %69
  %71 = load float, float* %70, align 4
  %72 = load float*, float** %7, align 8
  %73 = load i64, i64* %12, align 8
  %74 = getelementptr inbounds float, float* %72, i64 %73
  %75 = load float, float* %74, align 4
  %76 = fcmp une float %71, %75
  br i1 %76, label %77, label %78

77:                                               ; preds = %57
  store i1 false, i1* %4, align 1
  br label %83

78:                                               ; preds = %57
  br label %79

79:                                               ; preds = %78
  %80 = load i64, i64* %12, align 8
  %81 = add i64 %80, 1
  store i64 %81, i64* %12, align 8
  br label %54, !llvm.loop !13

82:                                               ; preds = %54
  store i1 true, i1* %4, align 1
  br label %83

83:                                               ; preds = %82, %77
  %84 = load i1, i1* %4, align 1
  ret i1 %84
}

declare void @srand(i32) #3

declare i64 @time(i64*) #3

declare i32 @rand() #3

; Function Attrs: noinline optnone ssp uwtable mustprogress
define linkonce_odr nonnull align 8 dereferenceable(8) %"class.std::__1::basic_ostream"* @_ZNSt3__1lsINS_11char_traitsIcEEEERNS_13basic_ostreamIcT_EES6_PKc(%"class.std::__1::basic_ostream"* nonnull align 8 dereferenceable(8) %0, i8* %1) #2 {
  %3 = alloca %"class.std::__1::basic_ostream"*, align 8
  %4 = alloca i8*, align 8
  store %"class.std::__1::basic_ostream"* %0, %"class.std::__1::basic_ostream"** %3, align 8
  store i8* %1, i8** %4, align 8
  %5 = load %"class.std::__1::basic_ostream"*, %"class.std::__1::basic_ostream"** %3, align 8
  %6 = load i8*, i8** %4, align 8
  %7 = load i8*, i8** %4, align 8
  %8 = call i64 @_ZNSt3__111char_traitsIcE6lengthEPKc(i8* %7) #9
  %9 = call nonnull align 8 dereferenceable(8) %"class.std::__1::basic_ostream"* @_ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m(%"class.std::__1::basic_ostream"* nonnull align 8 dereferenceable(8) %5, i8* %6, i64 %8)
  ret %"class.std::__1::basic_ostream"* %9
}

declare nonnull align 8 dereferenceable(8) %"class.std::__1::basic_ostream"* @_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEElsEf(%"class.std::__1::basic_ostream"* nonnull dereferenceable(8), float) #3

; Function Attrs: noinline optnone ssp uwtable mustprogress
define linkonce_odr nonnull align 8 dereferenceable(8) %"class.std::__1::basic_ostream"* @_ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m(%"class.std::__1::basic_ostream"* nonnull align 8 dereferenceable(8) %0, i8* %1, i64 %2) #2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %4 = alloca %"class.std::__1::basic_ostream"*, align 8
  %5 = alloca i8*, align 8
  %6 = alloca i64, align 8
  %7 = alloca %"class.std::__1::basic_ostream<char>::sentry", align 8
  %8 = alloca i8*, align 8
  %9 = alloca i32, align 4
  %10 = alloca %"class.std::__1::ostreambuf_iterator", align 8
  %11 = alloca %"class.std::__1::ostreambuf_iterator", align 8
  store %"class.std::__1::basic_ostream"* %0, %"class.std::__1::basic_ostream"** %4, align 8
  store i8* %1, i8** %5, align 8
  store i64 %2, i64* %6, align 8
  %12 = load %"class.std::__1::basic_ostream"*, %"class.std::__1::basic_ostream"** %4, align 8
  invoke void @_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryC1ERS3_(%"class.std::__1::basic_ostream<char>::sentry"* nonnull dereferenceable(16) %7, %"class.std::__1::basic_ostream"* nonnull align 8 dereferenceable(8) %12)
          to label %13 unwind label %80

13:                                               ; preds = %3
  %14 = invoke zeroext i1 @_ZNKSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentrycvbEv(%"class.std::__1::basic_ostream<char>::sentry"* nonnull dereferenceable(16) %7)
          to label %15 unwind label %84

15:                                               ; preds = %13
  br i1 %14, label %16, label %104

16:                                               ; preds = %15
  %17 = load %"class.std::__1::basic_ostream"*, %"class.std::__1::basic_ostream"** %4, align 8
  call void @_ZNSt3__119ostreambuf_iteratorIcNS_11char_traitsIcEEEC1ERNS_13basic_ostreamIcS2_EE(%"class.std::__1::ostreambuf_iterator"* nonnull dereferenceable(8) %11, %"class.std::__1::basic_ostream"* nonnull align 8 dereferenceable(8) %17) #9
  %18 = load i8*, i8** %5, align 8
  %19 = load %"class.std::__1::basic_ostream"*, %"class.std::__1::basic_ostream"** %4, align 8
  %20 = bitcast %"class.std::__1::basic_ostream"* %19 to i8**
  %21 = load i8*, i8** %20, align 8
  %22 = getelementptr i8, i8* %21, i64 -24
  %23 = bitcast i8* %22 to i64*
  %24 = load i64, i64* %23, align 8
  %25 = bitcast %"class.std::__1::basic_ostream"* %19 to i8*
  %26 = getelementptr inbounds i8, i8* %25, i64 %24
  %27 = bitcast i8* %26 to %"class.std::__1::ios_base"*
  %28 = invoke i32 @_ZNKSt3__18ios_base5flagsEv(%"class.std::__1::ios_base"* nonnull dereferenceable(136) %27)
          to label %29 unwind label %84

29:                                               ; preds = %16
  %30 = and i32 %28, 176
  %31 = icmp eq i32 %30, 32
  br i1 %31, label %32, label %36

32:                                               ; preds = %29
  %33 = load i8*, i8** %5, align 8
  %34 = load i64, i64* %6, align 8
  %35 = getelementptr inbounds i8, i8* %33, i64 %34
  br label %38

36:                                               ; preds = %29
  %37 = load i8*, i8** %5, align 8
  br label %38

38:                                               ; preds = %36, %32
  %39 = phi i8* [ %35, %32 ], [ %37, %36 ]
  %40 = load i8*, i8** %5, align 8
  %41 = load i64, i64* %6, align 8
  %42 = getelementptr inbounds i8, i8* %40, i64 %41
  %43 = load %"class.std::__1::basic_ostream"*, %"class.std::__1::basic_ostream"** %4, align 8
  %44 = bitcast %"class.std::__1::basic_ostream"* %43 to i8**
  %45 = load i8*, i8** %44, align 8
  %46 = getelementptr i8, i8* %45, i64 -24
  %47 = bitcast i8* %46 to i64*
  %48 = load i64, i64* %47, align 8
  %49 = bitcast %"class.std::__1::basic_ostream"* %43 to i8*
  %50 = getelementptr inbounds i8, i8* %49, i64 %48
  %51 = bitcast i8* %50 to %"class.std::__1::ios_base"*
  %52 = load %"class.std::__1::basic_ostream"*, %"class.std::__1::basic_ostream"** %4, align 8
  %53 = bitcast %"class.std::__1::basic_ostream"* %52 to i8**
  %54 = load i8*, i8** %53, align 8
  %55 = getelementptr i8, i8* %54, i64 -24
  %56 = bitcast i8* %55 to i64*
  %57 = load i64, i64* %56, align 8
  %58 = bitcast %"class.std::__1::basic_ostream"* %52 to i8*
  %59 = getelementptr inbounds i8, i8* %58, i64 %57
  %60 = bitcast i8* %59 to %"class.std::__1::basic_ios"*
  %61 = invoke signext i8 @_ZNKSt3__19basic_iosIcNS_11char_traitsIcEEE4fillEv(%"class.std::__1::basic_ios"* nonnull dereferenceable(148) %60)
          to label %62 unwind label %84

62:                                               ; preds = %38
  %63 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", %"class.std::__1::ostreambuf_iterator"* %11, i32 0, i32 0
  %64 = load %"class.std::__1::basic_streambuf"*, %"class.std::__1::basic_streambuf"** %63, align 8
  %65 = invoke %"class.std::__1::basic_streambuf"* @_ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_(%"class.std::__1::basic_streambuf"* %64, i8* %18, i8* %39, i8* %42, %"class.std::__1::ios_base"* nonnull align 8 dereferenceable(136) %51, i8 signext %61)
          to label %66 unwind label %84

66:                                               ; preds = %62
  %67 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", %"class.std::__1::ostreambuf_iterator"* %10, i32 0, i32 0
  store %"class.std::__1::basic_streambuf"* %65, %"class.std::__1::basic_streambuf"** %67, align 8
  %68 = call zeroext i1 @_ZNKSt3__119ostreambuf_iteratorIcNS_11char_traitsIcEEE6failedEv(%"class.std::__1::ostreambuf_iterator"* nonnull dereferenceable(8) %10) #9
  br i1 %68, label %69, label %103

69:                                               ; preds = %66
  %70 = load %"class.std::__1::basic_ostream"*, %"class.std::__1::basic_ostream"** %4, align 8
  %71 = bitcast %"class.std::__1::basic_ostream"* %70 to i8**
  %72 = load i8*, i8** %71, align 8
  %73 = getelementptr i8, i8* %72, i64 -24
  %74 = bitcast i8* %73 to i64*
  %75 = load i64, i64* %74, align 8
  %76 = bitcast %"class.std::__1::basic_ostream"* %70 to i8*
  %77 = getelementptr inbounds i8, i8* %76, i64 %75
  %78 = bitcast i8* %77 to %"class.std::__1::basic_ios"*
  invoke void @_ZNSt3__19basic_iosIcNS_11char_traitsIcEEE8setstateEj(%"class.std::__1::basic_ios"* nonnull dereferenceable(148) %78, i32 5)
          to label %79 unwind label %84

79:                                               ; preds = %69
  br label %103

80:                                               ; preds = %3
  %81 = landingpad { i8*, i32 }
          catch i8* null
  %82 = extractvalue { i8*, i32 } %81, 0
  store i8* %82, i8** %8, align 8
  %83 = extractvalue { i8*, i32 } %81, 1
  store i32 %83, i32* %9, align 4
  br label %88

84:                                               ; preds = %69, %62, %38, %16, %13
  %85 = landingpad { i8*, i32 }
          catch i8* null
  %86 = extractvalue { i8*, i32 } %85, 0
  store i8* %86, i8** %8, align 8
  %87 = extractvalue { i8*, i32 } %85, 1
  store i32 %87, i32* %9, align 4
  call void @_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryD1Ev(%"class.std::__1::basic_ostream<char>::sentry"* nonnull dereferenceable(16) %7) #9
  br label %88

88:                                               ; preds = %84, %80
  %89 = load i8*, i8** %8, align 8
  %90 = call i8* @__cxa_begin_catch(i8* %89) #9
  %91 = load %"class.std::__1::basic_ostream"*, %"class.std::__1::basic_ostream"** %4, align 8
  %92 = bitcast %"class.std::__1::basic_ostream"* %91 to i8**
  %93 = load i8*, i8** %92, align 8
  %94 = getelementptr i8, i8* %93, i64 -24
  %95 = bitcast i8* %94 to i64*
  %96 = load i64, i64* %95, align 8
  %97 = bitcast %"class.std::__1::basic_ostream"* %91 to i8*
  %98 = getelementptr inbounds i8, i8* %97, i64 %96
  %99 = bitcast i8* %98 to %"class.std::__1::ios_base"*
  invoke void @_ZNSt3__18ios_base33__set_badbit_and_consider_rethrowEv(%"class.std::__1::ios_base"* nonnull dereferenceable(136) %99)
          to label %100 unwind label %105

100:                                              ; preds = %88
  call void @__cxa_end_catch()
  br label %101

101:                                              ; preds = %100, %104
  %102 = load %"class.std::__1::basic_ostream"*, %"class.std::__1::basic_ostream"** %4, align 8
  ret %"class.std::__1::basic_ostream"* %102

103:                                              ; preds = %79, %66
  br label %104

104:                                              ; preds = %103, %15
  call void @_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryD1Ev(%"class.std::__1::basic_ostream<char>::sentry"* nonnull dereferenceable(16) %7) #9
  br label %101

105:                                              ; preds = %88
  %106 = landingpad { i8*, i32 }
          cleanup
  %107 = extractvalue { i8*, i32 } %106, 0
  store i8* %107, i8** %8, align 8
  %108 = extractvalue { i8*, i32 } %106, 1
  store i32 %108, i32* %9, align 4
  invoke void @__cxa_end_catch()
          to label %109 unwind label %115

109:                                              ; preds = %105
  br label %110

110:                                              ; preds = %109
  %111 = load i8*, i8** %8, align 8
  %112 = load i32, i32* %9, align 4
  %113 = insertvalue { i8*, i32 } undef, i8* %111, 0
  %114 = insertvalue { i8*, i32 } %113, i32 %112, 1
  resume { i8*, i32 } %114

115:                                              ; preds = %105
  %116 = landingpad { i8*, i32 }
          catch i8* null
  %117 = extractvalue { i8*, i32 } %116, 0
  call void @__clang_call_terminate(i8* %117) #10
  unreachable
}

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr i64 @_ZNSt3__111char_traitsIcE6lengthEPKc(i8* %0) #0 align 2 {
  %2 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  %3 = load i8*, i8** %2, align 8
  %4 = call i64 @strlen(i8* %3) #9
  ret i64 %4
}

declare void @_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryC1ERS3_(%"class.std::__1::basic_ostream<char>::sentry"* nonnull dereferenceable(16), %"class.std::__1::basic_ostream"* nonnull align 8 dereferenceable(8)) unnamed_addr #3

declare i32 @__gxx_personality_v0(...)

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr hidden zeroext i1 @_ZNKSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentrycvbEv(%"class.std::__1::basic_ostream<char>::sentry"* nonnull dereferenceable(16) %0) #0 align 2 {
  %2 = alloca %"class.std::__1::basic_ostream<char>::sentry"*, align 8
  store %"class.std::__1::basic_ostream<char>::sentry"* %0, %"class.std::__1::basic_ostream<char>::sentry"** %2, align 8
  %3 = load %"class.std::__1::basic_ostream<char>::sentry"*, %"class.std::__1::basic_ostream<char>::sentry"** %2, align 8
  %4 = getelementptr inbounds %"class.std::__1::basic_ostream<char>::sentry", %"class.std::__1::basic_ostream<char>::sentry"* %3, i32 0, i32 0
  %5 = load i8, i8* %4, align 8
  %6 = trunc i8 %5 to i1
  ret i1 %6
}

; Function Attrs: noinline optnone ssp uwtable mustprogress
define linkonce_odr hidden %"class.std::__1::basic_streambuf"* @_ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_(%"class.std::__1::basic_streambuf"* %0, i8* %1, i8* %2, i8* %3, %"class.std::__1::ios_base"* nonnull align 8 dereferenceable(136) %4, i8 signext %5) #2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %7 = alloca %"class.std::__1::ostreambuf_iterator", align 8
  %8 = alloca %"class.std::__1::ostreambuf_iterator", align 8
  %9 = alloca i8*, align 8
  %10 = alloca i8*, align 8
  %11 = alloca i8*, align 8
  %12 = alloca %"class.std::__1::ios_base"*, align 8
  %13 = alloca i8, align 1
  %14 = alloca i64, align 8
  %15 = alloca i64, align 8
  %16 = alloca i64, align 8
  %17 = alloca %"class.std::__1::basic_string", align 8
  %18 = alloca i8*, align 8
  %19 = alloca i32, align 4
  %20 = alloca i32, align 4
  %21 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", %"class.std::__1::ostreambuf_iterator"* %8, i32 0, i32 0
  store %"class.std::__1::basic_streambuf"* %0, %"class.std::__1::basic_streambuf"** %21, align 8
  store i8* %1, i8** %9, align 8
  store i8* %2, i8** %10, align 8
  store i8* %3, i8** %11, align 8
  store %"class.std::__1::ios_base"* %4, %"class.std::__1::ios_base"** %12, align 8
  store i8 %5, i8* %13, align 1
  %22 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", %"class.std::__1::ostreambuf_iterator"* %8, i32 0, i32 0
  %23 = load %"class.std::__1::basic_streambuf"*, %"class.std::__1::basic_streambuf"** %22, align 8
  %24 = icmp eq %"class.std::__1::basic_streambuf"* %23, null
  br i1 %24, label %25, label %28

25:                                               ; preds = %6
  %26 = bitcast %"class.std::__1::ostreambuf_iterator"* %7 to i8*
  %27 = bitcast %"class.std::__1::ostreambuf_iterator"* %8 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %26, i8* align 8 %27, i64 8, i1 false)
  br label %117

28:                                               ; preds = %6
  %29 = load i8*, i8** %11, align 8
  %30 = load i8*, i8** %9, align 8
  %31 = ptrtoint i8* %29 to i64
  %32 = ptrtoint i8* %30 to i64
  %33 = sub i64 %31, %32
  store i64 %33, i64* %14, align 8
  %34 = load %"class.std::__1::ios_base"*, %"class.std::__1::ios_base"** %12, align 8
  %35 = call i64 @_ZNKSt3__18ios_base5widthEv(%"class.std::__1::ios_base"* nonnull dereferenceable(136) %34)
  store i64 %35, i64* %15, align 8
  %36 = load i64, i64* %15, align 8
  %37 = load i64, i64* %14, align 8
  %38 = icmp sgt i64 %36, %37
  br i1 %38, label %39, label %43

39:                                               ; preds = %28
  %40 = load i64, i64* %14, align 8
  %41 = load i64, i64* %15, align 8
  %42 = sub nsw i64 %41, %40
  store i64 %42, i64* %15, align 8
  br label %44

43:                                               ; preds = %28
  store i64 0, i64* %15, align 8
  br label %44

44:                                               ; preds = %43, %39
  %45 = load i8*, i8** %10, align 8
  %46 = load i8*, i8** %9, align 8
  %47 = ptrtoint i8* %45 to i64
  %48 = ptrtoint i8* %46 to i64
  %49 = sub i64 %47, %48
  store i64 %49, i64* %16, align 8
  %50 = load i64, i64* %16, align 8
  %51 = icmp sgt i64 %50, 0
  br i1 %51, label %52, label %65

52:                                               ; preds = %44
  %53 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", %"class.std::__1::ostreambuf_iterator"* %8, i32 0, i32 0
  %54 = load %"class.std::__1::basic_streambuf"*, %"class.std::__1::basic_streambuf"** %53, align 8
  %55 = load i8*, i8** %9, align 8
  %56 = load i64, i64* %16, align 8
  %57 = call i64 @_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE5sputnEPKcl(%"class.std::__1::basic_streambuf"* nonnull dereferenceable(64) %54, i8* %55, i64 %56)
  %58 = load i64, i64* %16, align 8
  %59 = icmp ne i64 %57, %58
  br i1 %59, label %60, label %64

60:                                               ; preds = %52
  %61 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", %"class.std::__1::ostreambuf_iterator"* %8, i32 0, i32 0
  store %"class.std::__1::basic_streambuf"* null, %"class.std::__1::basic_streambuf"** %61, align 8
  %62 = bitcast %"class.std::__1::ostreambuf_iterator"* %7 to i8*
  %63 = bitcast %"class.std::__1::ostreambuf_iterator"* %8 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %62, i8* align 8 %63, i64 8, i1 false)
  br label %117

64:                                               ; preds = %52
  br label %65

65:                                               ; preds = %64, %44
  %66 = load i64, i64* %15, align 8
  %67 = icmp sgt i64 %66, 0
  br i1 %67, label %68, label %91

68:                                               ; preds = %65
  %69 = load i64, i64* %15, align 8
  %70 = load i8, i8* %13, align 1
  call void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEC1Emc(%"class.std::__1::basic_string"* nonnull dereferenceable(24) %17, i64 %69, i8 signext %70)
  %71 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", %"class.std::__1::ostreambuf_iterator"* %8, i32 0, i32 0
  %72 = load %"class.std::__1::basic_streambuf"*, %"class.std::__1::basic_streambuf"** %71, align 8
  %73 = call i8* @_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE4dataEv(%"class.std::__1::basic_string"* nonnull dereferenceable(24) %17) #9
  %74 = load i64, i64* %15, align 8
  %75 = invoke i64 @_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE5sputnEPKcl(%"class.std::__1::basic_streambuf"* nonnull dereferenceable(64) %72, i8* %73, i64 %74)
          to label %76 unwind label %83

76:                                               ; preds = %68
  %77 = load i64, i64* %15, align 8
  %78 = icmp ne i64 %75, %77
  br i1 %78, label %79, label %87

79:                                               ; preds = %76
  %80 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", %"class.std::__1::ostreambuf_iterator"* %8, i32 0, i32 0
  store %"class.std::__1::basic_streambuf"* null, %"class.std::__1::basic_streambuf"** %80, align 8
  %81 = bitcast %"class.std::__1::ostreambuf_iterator"* %7 to i8*
  %82 = bitcast %"class.std::__1::ostreambuf_iterator"* %8 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %81, i8* align 8 %82, i64 8, i1 false)
  store i32 1, i32* %20, align 4
  br label %88

83:                                               ; preds = %68
  %84 = landingpad { i8*, i32 }
          cleanup
  %85 = extractvalue { i8*, i32 } %84, 0
  store i8* %85, i8** %18, align 8
  %86 = extractvalue { i8*, i32 } %84, 1
  store i32 %86, i32* %19, align 4
  call void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED1Ev(%"class.std::__1::basic_string"* nonnull dereferenceable(24) %17) #9
  br label %120

87:                                               ; preds = %76
  store i32 0, i32* %20, align 4
  br label %88

88:                                               ; preds = %87, %79
  call void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED1Ev(%"class.std::__1::basic_string"* nonnull dereferenceable(24) %17) #9
  %89 = load i32, i32* %20, align 4
  switch i32 %89, label %125 [
    i32 0, label %90
    i32 1, label %117
  ]

90:                                               ; preds = %88
  br label %91

91:                                               ; preds = %90, %65
  %92 = load i8*, i8** %11, align 8
  %93 = load i8*, i8** %10, align 8
  %94 = ptrtoint i8* %92 to i64
  %95 = ptrtoint i8* %93 to i64
  %96 = sub i64 %94, %95
  store i64 %96, i64* %16, align 8
  %97 = load i64, i64* %16, align 8
  %98 = icmp sgt i64 %97, 0
  br i1 %98, label %99, label %112

99:                                               ; preds = %91
  %100 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", %"class.std::__1::ostreambuf_iterator"* %8, i32 0, i32 0
  %101 = load %"class.std::__1::basic_streambuf"*, %"class.std::__1::basic_streambuf"** %100, align 8
  %102 = load i8*, i8** %10, align 8
  %103 = load i64, i64* %16, align 8
  %104 = call i64 @_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE5sputnEPKcl(%"class.std::__1::basic_streambuf"* nonnull dereferenceable(64) %101, i8* %102, i64 %103)
  %105 = load i64, i64* %16, align 8
  %106 = icmp ne i64 %104, %105
  br i1 %106, label %107, label %111

107:                                              ; preds = %99
  %108 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", %"class.std::__1::ostreambuf_iterator"* %8, i32 0, i32 0
  store %"class.std::__1::basic_streambuf"* null, %"class.std::__1::basic_streambuf"** %108, align 8
  %109 = bitcast %"class.std::__1::ostreambuf_iterator"* %7 to i8*
  %110 = bitcast %"class.std::__1::ostreambuf_iterator"* %8 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %109, i8* align 8 %110, i64 8, i1 false)
  br label %117

111:                                              ; preds = %99
  br label %112

112:                                              ; preds = %111, %91
  %113 = load %"class.std::__1::ios_base"*, %"class.std::__1::ios_base"** %12, align 8
  %114 = call i64 @_ZNSt3__18ios_base5widthEl(%"class.std::__1::ios_base"* nonnull dereferenceable(136) %113, i64 0)
  %115 = bitcast %"class.std::__1::ostreambuf_iterator"* %7 to i8*
  %116 = bitcast %"class.std::__1::ostreambuf_iterator"* %8 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %115, i8* align 8 %116, i64 8, i1 false)
  br label %117

117:                                              ; preds = %112, %107, %88, %60, %25
  %118 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", %"class.std::__1::ostreambuf_iterator"* %7, i32 0, i32 0
  %119 = load %"class.std::__1::basic_streambuf"*, %"class.std::__1::basic_streambuf"** %118, align 8
  ret %"class.std::__1::basic_streambuf"* %119

120:                                              ; preds = %83
  %121 = load i8*, i8** %18, align 8
  %122 = load i32, i32* %19, align 4
  %123 = insertvalue { i8*, i32 } undef, i8* %121, 0
  %124 = insertvalue { i8*, i32 } %123, i32 %122, 1
  resume { i8*, i32 } %124

125:                                              ; preds = %88
  unreachable
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden void @_ZNSt3__119ostreambuf_iteratorIcNS_11char_traitsIcEEEC1ERNS_13basic_ostreamIcS2_EE(%"class.std::__1::ostreambuf_iterator"* nonnull dereferenceable(8) %0, %"class.std::__1::basic_ostream"* nonnull align 8 dereferenceable(8) %1) unnamed_addr #4 align 2 {
  %3 = alloca %"class.std::__1::ostreambuf_iterator"*, align 8
  %4 = alloca %"class.std::__1::basic_ostream"*, align 8
  store %"class.std::__1::ostreambuf_iterator"* %0, %"class.std::__1::ostreambuf_iterator"** %3, align 8
  store %"class.std::__1::basic_ostream"* %1, %"class.std::__1::basic_ostream"** %4, align 8
  %5 = load %"class.std::__1::ostreambuf_iterator"*, %"class.std::__1::ostreambuf_iterator"** %3, align 8
  %6 = load %"class.std::__1::basic_ostream"*, %"class.std::__1::basic_ostream"** %4, align 8
  call void @_ZNSt3__119ostreambuf_iteratorIcNS_11char_traitsIcEEEC2ERNS_13basic_ostreamIcS2_EE(%"class.std::__1::ostreambuf_iterator"* nonnull dereferenceable(8) %5, %"class.std::__1::basic_ostream"* nonnull align 8 dereferenceable(8) %6) #9
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr hidden i32 @_ZNKSt3__18ios_base5flagsEv(%"class.std::__1::ios_base"* nonnull dereferenceable(136) %0) #0 align 2 {
  %2 = alloca %"class.std::__1::ios_base"*, align 8
  store %"class.std::__1::ios_base"* %0, %"class.std::__1::ios_base"** %2, align 8
  %3 = load %"class.std::__1::ios_base"*, %"class.std::__1::ios_base"** %2, align 8
  %4 = getelementptr inbounds %"class.std::__1::ios_base", %"class.std::__1::ios_base"* %3, i32 0, i32 1
  %5 = load i32, i32* %4, align 8
  ret i32 %5
}

; Function Attrs: noinline optnone ssp uwtable mustprogress
define linkonce_odr hidden signext i8 @_ZNKSt3__19basic_iosIcNS_11char_traitsIcEEE4fillEv(%"class.std::__1::basic_ios"* nonnull dereferenceable(148) %0) #2 align 2 {
  %2 = alloca %"class.std::__1::basic_ios"*, align 8
  store %"class.std::__1::basic_ios"* %0, %"class.std::__1::basic_ios"** %2, align 8
  %3 = load %"class.std::__1::basic_ios"*, %"class.std::__1::basic_ios"** %2, align 8
  %4 = call i32 @_ZNSt3__111char_traitsIcE3eofEv() #9
  %5 = getelementptr inbounds %"class.std::__1::basic_ios", %"class.std::__1::basic_ios"* %3, i32 0, i32 2
  %6 = load i32, i32* %5, align 8
  %7 = call zeroext i1 @_ZNSt3__111char_traitsIcE11eq_int_typeEii(i32 %4, i32 %6) #9
  br i1 %7, label %8, label %12

8:                                                ; preds = %1
  %9 = call signext i8 @_ZNKSt3__19basic_iosIcNS_11char_traitsIcEEE5widenEc(%"class.std::__1::basic_ios"* nonnull dereferenceable(148) %3, i8 signext 32)
  %10 = sext i8 %9 to i32
  %11 = getelementptr inbounds %"class.std::__1::basic_ios", %"class.std::__1::basic_ios"* %3, i32 0, i32 2
  store i32 %10, i32* %11, align 8
  br label %12

12:                                               ; preds = %8, %1
  %13 = getelementptr inbounds %"class.std::__1::basic_ios", %"class.std::__1::basic_ios"* %3, i32 0, i32 2
  %14 = load i32, i32* %13, align 8
  %15 = trunc i32 %14 to i8
  ret i8 %15
}

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr hidden zeroext i1 @_ZNKSt3__119ostreambuf_iteratorIcNS_11char_traitsIcEEE6failedEv(%"class.std::__1::ostreambuf_iterator"* nonnull dereferenceable(8) %0) #0 align 2 {
  %2 = alloca %"class.std::__1::ostreambuf_iterator"*, align 8
  store %"class.std::__1::ostreambuf_iterator"* %0, %"class.std::__1::ostreambuf_iterator"** %2, align 8
  %3 = load %"class.std::__1::ostreambuf_iterator"*, %"class.std::__1::ostreambuf_iterator"** %2, align 8
  %4 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", %"class.std::__1::ostreambuf_iterator"* %3, i32 0, i32 0
  %5 = load %"class.std::__1::basic_streambuf"*, %"class.std::__1::basic_streambuf"** %4, align 8
  %6 = icmp eq %"class.std::__1::basic_streambuf"* %5, null
  ret i1 %6
}

; Function Attrs: noinline optnone ssp uwtable mustprogress
define linkonce_odr hidden void @_ZNSt3__19basic_iosIcNS_11char_traitsIcEEE8setstateEj(%"class.std::__1::basic_ios"* nonnull dereferenceable(148) %0, i32 %1) #2 align 2 {
  %3 = alloca %"class.std::__1::basic_ios"*, align 8
  %4 = alloca i32, align 4
  store %"class.std::__1::basic_ios"* %0, %"class.std::__1::basic_ios"** %3, align 8
  store i32 %1, i32* %4, align 4
  %5 = load %"class.std::__1::basic_ios"*, %"class.std::__1::basic_ios"** %3, align 8
  %6 = bitcast %"class.std::__1::basic_ios"* %5 to %"class.std::__1::ios_base"*
  %7 = load i32, i32* %4, align 4
  call void @_ZNSt3__18ios_base8setstateEj(%"class.std::__1::ios_base"* nonnull dereferenceable(136) %6, i32 %7)
  ret void
}

; Function Attrs: nounwind
declare void @_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryD1Ev(%"class.std::__1::basic_ostream<char>::sentry"* nonnull dereferenceable(16)) unnamed_addr #5

declare i8* @__cxa_begin_catch(i8*)

declare void @_ZNSt3__18ios_base33__set_badbit_and_consider_rethrowEv(%"class.std::__1::ios_base"* nonnull dereferenceable(136)) #3

declare void @__cxa_end_catch()

; Function Attrs: noinline noreturn nounwind
define linkonce_odr hidden void @__clang_call_terminate(i8* %0) #6 {
  %2 = call i8* @__cxa_begin_catch(i8* %0) #9
  call void @_ZSt9terminatev() #10
  unreachable
}

declare void @_ZSt9terminatev()

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #7

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr hidden i64 @_ZNKSt3__18ios_base5widthEv(%"class.std::__1::ios_base"* nonnull dereferenceable(136) %0) #0 align 2 {
  %2 = alloca %"class.std::__1::ios_base"*, align 8
  store %"class.std::__1::ios_base"* %0, %"class.std::__1::ios_base"** %2, align 8
  %3 = load %"class.std::__1::ios_base"*, %"class.std::__1::ios_base"** %2, align 8
  %4 = getelementptr inbounds %"class.std::__1::ios_base", %"class.std::__1::ios_base"* %3, i32 0, i32 3
  %5 = load i64, i64* %4, align 8
  ret i64 %5
}

; Function Attrs: noinline optnone ssp uwtable mustprogress
define linkonce_odr hidden i64 @_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE5sputnEPKcl(%"class.std::__1::basic_streambuf"* nonnull dereferenceable(64) %0, i8* %1, i64 %2) #2 align 2 {
  %4 = alloca %"class.std::__1::basic_streambuf"*, align 8
  %5 = alloca i8*, align 8
  %6 = alloca i64, align 8
  store %"class.std::__1::basic_streambuf"* %0, %"class.std::__1::basic_streambuf"** %4, align 8
  store i8* %1, i8** %5, align 8
  store i64 %2, i64* %6, align 8
  %7 = load %"class.std::__1::basic_streambuf"*, %"class.std::__1::basic_streambuf"** %4, align 8
  %8 = load i8*, i8** %5, align 8
  %9 = load i64, i64* %6, align 8
  %10 = bitcast %"class.std::__1::basic_streambuf"* %7 to i64 (%"class.std::__1::basic_streambuf"*, i8*, i64)***
  %11 = load i64 (%"class.std::__1::basic_streambuf"*, i8*, i64)**, i64 (%"class.std::__1::basic_streambuf"*, i8*, i64)*** %10, align 8
  %12 = getelementptr inbounds i64 (%"class.std::__1::basic_streambuf"*, i8*, i64)*, i64 (%"class.std::__1::basic_streambuf"*, i8*, i64)** %11, i64 12
  %13 = load i64 (%"class.std::__1::basic_streambuf"*, i8*, i64)*, i64 (%"class.std::__1::basic_streambuf"*, i8*, i64)** %12, align 8
  %14 = call i64 %13(%"class.std::__1::basic_streambuf"* nonnull dereferenceable(64) %7, i8* %8, i64 %9)
  ret i64 %14
}

; Function Attrs: noinline optnone ssp uwtable
define linkonce_odr hidden void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEC1Emc(%"class.std::__1::basic_string"* nonnull dereferenceable(24) %0, i64 %1, i8 signext %2) unnamed_addr #8 align 2 {
  %4 = alloca %"class.std::__1::basic_string"*, align 8
  %5 = alloca i64, align 8
  %6 = alloca i8, align 1
  store %"class.std::__1::basic_string"* %0, %"class.std::__1::basic_string"** %4, align 8
  store i64 %1, i64* %5, align 8
  store i8 %2, i8* %6, align 1
  %7 = load %"class.std::__1::basic_string"*, %"class.std::__1::basic_string"** %4, align 8
  %8 = load i64, i64* %5, align 8
  %9 = load i8, i8* %6, align 1
  call void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEC2Emc(%"class.std::__1::basic_string"* nonnull dereferenceable(24) %7, i64 %8, i8 signext %9)
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr hidden i8* @_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE4dataEv(%"class.std::__1::basic_string"* nonnull dereferenceable(24) %0) #0 align 2 {
  %2 = alloca %"class.std::__1::basic_string"*, align 8
  store %"class.std::__1::basic_string"* %0, %"class.std::__1::basic_string"** %2, align 8
  %3 = load %"class.std::__1::basic_string"*, %"class.std::__1::basic_string"** %2, align 8
  %4 = call i8* @_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE13__get_pointerEv(%"class.std::__1::basic_string"* nonnull dereferenceable(24) %3) #9
  %5 = call i8* @_ZNSt3__112__to_addressIKcEEPT_S3_(i8* %4) #9
  ret i8* %5
}

; Function Attrs: nounwind
declare void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED1Ev(%"class.std::__1::basic_string"* nonnull dereferenceable(24)) unnamed_addr #5

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr hidden i64 @_ZNSt3__18ios_base5widthEl(%"class.std::__1::ios_base"* nonnull dereferenceable(136) %0, i64 %1) #0 align 2 {
  %3 = alloca %"class.std::__1::ios_base"*, align 8
  %4 = alloca i64, align 8
  %5 = alloca i64, align 8
  store %"class.std::__1::ios_base"* %0, %"class.std::__1::ios_base"** %3, align 8
  store i64 %1, i64* %4, align 8
  %6 = load %"class.std::__1::ios_base"*, %"class.std::__1::ios_base"** %3, align 8
  %7 = getelementptr inbounds %"class.std::__1::ios_base", %"class.std::__1::ios_base"* %6, i32 0, i32 3
  %8 = load i64, i64* %7, align 8
  store i64 %8, i64* %5, align 8
  %9 = load i64, i64* %4, align 8
  %10 = getelementptr inbounds %"class.std::__1::ios_base", %"class.std::__1::ios_base"* %6, i32 0, i32 3
  store i64 %9, i64* %10, align 8
  %11 = load i64, i64* %5, align 8
  ret i64 %11
}

; Function Attrs: noinline optnone ssp uwtable
define linkonce_odr hidden void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEC2Emc(%"class.std::__1::basic_string"* nonnull dereferenceable(24) %0, i64 %1, i8 signext %2) unnamed_addr #8 align 2 {
  %4 = alloca %"class.std::__1::basic_string"*, align 8
  %5 = alloca i64, align 8
  %6 = alloca i8, align 1
  %7 = alloca %"struct.std::__1::__default_init_tag", align 1
  %8 = alloca %"struct.std::__1::__default_init_tag", align 1
  store %"class.std::__1::basic_string"* %0, %"class.std::__1::basic_string"** %4, align 8
  store i64 %1, i64* %5, align 8
  store i8 %2, i8* %6, align 1
  %9 = load %"class.std::__1::basic_string"*, %"class.std::__1::basic_string"** %4, align 8
  %10 = bitcast %"class.std::__1::basic_string"* %9 to %"class.std::__1::__basic_string_common"*
  %11 = getelementptr inbounds %"class.std::__1::basic_string", %"class.std::__1::basic_string"* %9, i32 0, i32 0
  call void @_ZNSt3__117__compressed_pairINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repES5_EC1INS_18__default_init_tagESA_EEOT_OT0_(%"class.std::__1::__compressed_pair"* nonnull dereferenceable(24) %11, %"struct.std::__1::__default_init_tag"* nonnull align 1 dereferenceable(1) %7, %"struct.std::__1::__default_init_tag"* nonnull align 1 dereferenceable(1) %8)
  %12 = load i64, i64* %5, align 8
  %13 = load i8, i8* %6, align 1
  call void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6__initEmc(%"class.std::__1::basic_string"* nonnull dereferenceable(24) %9, i64 %12, i8 signext %13)
  ret void
}

; Function Attrs: noinline optnone ssp uwtable
define linkonce_odr void @_ZNSt3__117__compressed_pairINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repES5_EC1INS_18__default_init_tagESA_EEOT_OT0_(%"class.std::__1::__compressed_pair"* nonnull dereferenceable(24) %0, %"struct.std::__1::__default_init_tag"* nonnull align 1 dereferenceable(1) %1, %"struct.std::__1::__default_init_tag"* nonnull align 1 dereferenceable(1) %2) unnamed_addr #8 align 2 {
  %4 = alloca %"class.std::__1::__compressed_pair"*, align 8
  %5 = alloca %"struct.std::__1::__default_init_tag"*, align 8
  %6 = alloca %"struct.std::__1::__default_init_tag"*, align 8
  store %"class.std::__1::__compressed_pair"* %0, %"class.std::__1::__compressed_pair"** %4, align 8
  store %"struct.std::__1::__default_init_tag"* %1, %"struct.std::__1::__default_init_tag"** %5, align 8
  store %"struct.std::__1::__default_init_tag"* %2, %"struct.std::__1::__default_init_tag"** %6, align 8
  %7 = load %"class.std::__1::__compressed_pair"*, %"class.std::__1::__compressed_pair"** %4, align 8
  %8 = load %"struct.std::__1::__default_init_tag"*, %"struct.std::__1::__default_init_tag"** %5, align 8
  %9 = load %"struct.std::__1::__default_init_tag"*, %"struct.std::__1::__default_init_tag"** %6, align 8
  call void @_ZNSt3__117__compressed_pairINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repES5_EC2INS_18__default_init_tagESA_EEOT_OT0_(%"class.std::__1::__compressed_pair"* nonnull dereferenceable(24) %7, %"struct.std::__1::__default_init_tag"* nonnull align 1 dereferenceable(1) %8, %"struct.std::__1::__default_init_tag"* nonnull align 1 dereferenceable(1) %9)
  ret void
}

declare void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6__initEmc(%"class.std::__1::basic_string"* nonnull dereferenceable(24), i64, i8 signext) #3

; Function Attrs: noinline optnone ssp uwtable
define linkonce_odr void @_ZNSt3__117__compressed_pairINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repES5_EC2INS_18__default_init_tagESA_EEOT_OT0_(%"class.std::__1::__compressed_pair"* nonnull dereferenceable(24) %0, %"struct.std::__1::__default_init_tag"* nonnull align 1 dereferenceable(1) %1, %"struct.std::__1::__default_init_tag"* nonnull align 1 dereferenceable(1) %2) unnamed_addr #8 align 2 {
  %4 = alloca %"class.std::__1::__compressed_pair"*, align 8
  %5 = alloca %"struct.std::__1::__default_init_tag"*, align 8
  %6 = alloca %"struct.std::__1::__default_init_tag"*, align 8
  %7 = alloca %"struct.std::__1::__default_init_tag", align 1
  %8 = alloca %"struct.std::__1::__default_init_tag", align 1
  store %"class.std::__1::__compressed_pair"* %0, %"class.std::__1::__compressed_pair"** %4, align 8
  store %"struct.std::__1::__default_init_tag"* %1, %"struct.std::__1::__default_init_tag"** %5, align 8
  store %"struct.std::__1::__default_init_tag"* %2, %"struct.std::__1::__default_init_tag"** %6, align 8
  %9 = load %"class.std::__1::__compressed_pair"*, %"class.std::__1::__compressed_pair"** %4, align 8
  %10 = bitcast %"class.std::__1::__compressed_pair"* %9 to %"struct.std::__1::__compressed_pair_elem"*
  %11 = load %"struct.std::__1::__default_init_tag"*, %"struct.std::__1::__default_init_tag"** %5, align 8
  %12 = call nonnull align 1 dereferenceable(1) %"struct.std::__1::__default_init_tag"* @_ZNSt3__17forwardINS_18__default_init_tagEEEOT_RNS_16remove_referenceIS2_E4typeE(%"struct.std::__1::__default_init_tag"* nonnull align 1 dereferenceable(1) %11) #9
  call void @_ZNSt3__122__compressed_pair_elemINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repELi0ELb0EEC2ENS_18__default_init_tagE(%"struct.std::__1::__compressed_pair_elem"* nonnull dereferenceable(24) %10)
  %13 = bitcast %"class.std::__1::__compressed_pair"* %9 to %"struct.std::__1::__compressed_pair_elem.1"*
  %14 = load %"struct.std::__1::__default_init_tag"*, %"struct.std::__1::__default_init_tag"** %6, align 8
  %15 = call nonnull align 1 dereferenceable(1) %"struct.std::__1::__default_init_tag"* @_ZNSt3__17forwardINS_18__default_init_tagEEEOT_RNS_16remove_referenceIS2_E4typeE(%"struct.std::__1::__default_init_tag"* nonnull align 1 dereferenceable(1) %14) #9
  call void @_ZNSt3__122__compressed_pair_elemINS_9allocatorIcEELi1ELb1EEC2ENS_18__default_init_tagE(%"struct.std::__1::__compressed_pair_elem.1"* nonnull dereferenceable(1) %13)
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr hidden nonnull align 1 dereferenceable(1) %"struct.std::__1::__default_init_tag"* @_ZNSt3__17forwardINS_18__default_init_tagEEEOT_RNS_16remove_referenceIS2_E4typeE(%"struct.std::__1::__default_init_tag"* nonnull align 1 dereferenceable(1) %0) #0 {
  %2 = alloca %"struct.std::__1::__default_init_tag"*, align 8
  store %"struct.std::__1::__default_init_tag"* %0, %"struct.std::__1::__default_init_tag"** %2, align 8
  %3 = load %"struct.std::__1::__default_init_tag"*, %"struct.std::__1::__default_init_tag"** %2, align 8
  ret %"struct.std::__1::__default_init_tag"* %3
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden void @_ZNSt3__122__compressed_pair_elemINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repELi0ELb0EEC2ENS_18__default_init_tagE(%"struct.std::__1::__compressed_pair_elem"* nonnull dereferenceable(24) %0) unnamed_addr #4 align 2 {
  %2 = alloca %"struct.std::__1::__default_init_tag", align 1
  %3 = alloca %"struct.std::__1::__compressed_pair_elem"*, align 8
  store %"struct.std::__1::__compressed_pair_elem"* %0, %"struct.std::__1::__compressed_pair_elem"** %3, align 8
  %4 = load %"struct.std::__1::__compressed_pair_elem"*, %"struct.std::__1::__compressed_pair_elem"** %3, align 8
  %5 = getelementptr inbounds %"struct.std::__1::__compressed_pair_elem", %"struct.std::__1::__compressed_pair_elem"* %4, i32 0, i32 0
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden void @_ZNSt3__122__compressed_pair_elemINS_9allocatorIcEELi1ELb1EEC2ENS_18__default_init_tagE(%"struct.std::__1::__compressed_pair_elem.1"* nonnull dereferenceable(1) %0) unnamed_addr #4 align 2 {
  %2 = alloca %"struct.std::__1::__default_init_tag", align 1
  %3 = alloca %"struct.std::__1::__compressed_pair_elem.1"*, align 8
  store %"struct.std::__1::__compressed_pair_elem.1"* %0, %"struct.std::__1::__compressed_pair_elem.1"** %3, align 8
  %4 = load %"struct.std::__1::__compressed_pair_elem.1"*, %"struct.std::__1::__compressed_pair_elem.1"** %3, align 8
  %5 = bitcast %"struct.std::__1::__compressed_pair_elem.1"* %4 to %"class.std::__1::allocator"*
  call void @_ZNSt3__19allocatorIcEC2Ev(%"class.std::__1::allocator"* nonnull dereferenceable(1) %5) #9
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden void @_ZNSt3__19allocatorIcEC2Ev(%"class.std::__1::allocator"* nonnull dereferenceable(1) %0) unnamed_addr #4 align 2 {
  %2 = alloca %"class.std::__1::allocator"*, align 8
  store %"class.std::__1::allocator"* %0, %"class.std::__1::allocator"** %2, align 8
  %3 = load %"class.std::__1::allocator"*, %"class.std::__1::allocator"** %2, align 8
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr hidden i8* @_ZNSt3__112__to_addressIKcEEPT_S3_(i8* %0) #0 {
  %2 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  %3 = load i8*, i8** %2, align 8
  ret i8* %3
}

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr hidden i8* @_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE13__get_pointerEv(%"class.std::__1::basic_string"* nonnull dereferenceable(24) %0) #0 align 2 {
  %2 = alloca %"class.std::__1::basic_string"*, align 8
  store %"class.std::__1::basic_string"* %0, %"class.std::__1::basic_string"** %2, align 8
  %3 = load %"class.std::__1::basic_string"*, %"class.std::__1::basic_string"** %2, align 8
  %4 = call zeroext i1 @_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE9__is_longEv(%"class.std::__1::basic_string"* nonnull dereferenceable(24) %3) #9
  br i1 %4, label %5, label %7

5:                                                ; preds = %1
  %6 = call i8* @_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE18__get_long_pointerEv(%"class.std::__1::basic_string"* nonnull dereferenceable(24) %3) #9
  br label %9

7:                                                ; preds = %1
  %8 = call i8* @_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE19__get_short_pointerEv(%"class.std::__1::basic_string"* nonnull dereferenceable(24) %3) #9
  br label %9

9:                                                ; preds = %7, %5
  %10 = phi i8* [ %6, %5 ], [ %8, %7 ]
  ret i8* %10
}

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr hidden zeroext i1 @_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE9__is_longEv(%"class.std::__1::basic_string"* nonnull dereferenceable(24) %0) #0 align 2 {
  %2 = alloca %"class.std::__1::basic_string"*, align 8
  store %"class.std::__1::basic_string"* %0, %"class.std::__1::basic_string"** %2, align 8
  %3 = load %"class.std::__1::basic_string"*, %"class.std::__1::basic_string"** %2, align 8
  %4 = getelementptr inbounds %"class.std::__1::basic_string", %"class.std::__1::basic_string"* %3, i32 0, i32 0
  %5 = call nonnull align 8 dereferenceable(24) %"struct.std::__1::basic_string<char>::__rep"* @_ZNKSt3__117__compressed_pairINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repES5_E5firstEv(%"class.std::__1::__compressed_pair"* nonnull dereferenceable(24) %4) #9
  %6 = getelementptr inbounds %"struct.std::__1::basic_string<char>::__rep", %"struct.std::__1::basic_string<char>::__rep"* %5, i32 0, i32 0
  %7 = bitcast %union.anon* %6 to %"struct.std::__1::basic_string<char>::__short"*
  %8 = getelementptr inbounds %"struct.std::__1::basic_string<char>::__short", %"struct.std::__1::basic_string<char>::__short"* %7, i32 0, i32 0
  %9 = bitcast %union.anon.0* %8 to i8*
  %10 = load i8, i8* %9, align 8
  %11 = zext i8 %10 to i64
  %12 = and i64 %11, 1
  %13 = icmp ne i64 %12, 0
  ret i1 %13
}

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr hidden i8* @_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE18__get_long_pointerEv(%"class.std::__1::basic_string"* nonnull dereferenceable(24) %0) #0 align 2 {
  %2 = alloca %"class.std::__1::basic_string"*, align 8
  store %"class.std::__1::basic_string"* %0, %"class.std::__1::basic_string"** %2, align 8
  %3 = load %"class.std::__1::basic_string"*, %"class.std::__1::basic_string"** %2, align 8
  %4 = getelementptr inbounds %"class.std::__1::basic_string", %"class.std::__1::basic_string"* %3, i32 0, i32 0
  %5 = call nonnull align 8 dereferenceable(24) %"struct.std::__1::basic_string<char>::__rep"* @_ZNKSt3__117__compressed_pairINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repES5_E5firstEv(%"class.std::__1::__compressed_pair"* nonnull dereferenceable(24) %4) #9
  %6 = getelementptr inbounds %"struct.std::__1::basic_string<char>::__rep", %"struct.std::__1::basic_string<char>::__rep"* %5, i32 0, i32 0
  %7 = bitcast %union.anon* %6 to %"struct.std::__1::basic_string<char>::__long"*
  %8 = getelementptr inbounds %"struct.std::__1::basic_string<char>::__long", %"struct.std::__1::basic_string<char>::__long"* %7, i32 0, i32 2
  %9 = load i8*, i8** %8, align 8
  ret i8* %9
}

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr hidden i8* @_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE19__get_short_pointerEv(%"class.std::__1::basic_string"* nonnull dereferenceable(24) %0) #0 align 2 {
  %2 = alloca %"class.std::__1::basic_string"*, align 8
  store %"class.std::__1::basic_string"* %0, %"class.std::__1::basic_string"** %2, align 8
  %3 = load %"class.std::__1::basic_string"*, %"class.std::__1::basic_string"** %2, align 8
  %4 = getelementptr inbounds %"class.std::__1::basic_string", %"class.std::__1::basic_string"* %3, i32 0, i32 0
  %5 = call nonnull align 8 dereferenceable(24) %"struct.std::__1::basic_string<char>::__rep"* @_ZNKSt3__117__compressed_pairINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repES5_E5firstEv(%"class.std::__1::__compressed_pair"* nonnull dereferenceable(24) %4) #9
  %6 = getelementptr inbounds %"struct.std::__1::basic_string<char>::__rep", %"struct.std::__1::basic_string<char>::__rep"* %5, i32 0, i32 0
  %7 = bitcast %union.anon* %6 to %"struct.std::__1::basic_string<char>::__short"*
  %8 = getelementptr inbounds %"struct.std::__1::basic_string<char>::__short", %"struct.std::__1::basic_string<char>::__short"* %7, i32 0, i32 1
  %9 = getelementptr inbounds [23 x i8], [23 x i8]* %8, i64 0, i64 0
  %10 = call i8* @_ZNSt3__114pointer_traitsIPKcE10pointer_toERS1_(i8* nonnull align 1 dereferenceable(1) %9) #9
  ret i8* %10
}

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr hidden nonnull align 8 dereferenceable(24) %"struct.std::__1::basic_string<char>::__rep"* @_ZNKSt3__117__compressed_pairINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repES5_E5firstEv(%"class.std::__1::__compressed_pair"* nonnull dereferenceable(24) %0) #0 align 2 {
  %2 = alloca %"class.std::__1::__compressed_pair"*, align 8
  store %"class.std::__1::__compressed_pair"* %0, %"class.std::__1::__compressed_pair"** %2, align 8
  %3 = load %"class.std::__1::__compressed_pair"*, %"class.std::__1::__compressed_pair"** %2, align 8
  %4 = bitcast %"class.std::__1::__compressed_pair"* %3 to %"struct.std::__1::__compressed_pair_elem"*
  %5 = call nonnull align 8 dereferenceable(24) %"struct.std::__1::basic_string<char>::__rep"* @_ZNKSt3__122__compressed_pair_elemINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repELi0ELb0EE5__getEv(%"struct.std::__1::__compressed_pair_elem"* nonnull dereferenceable(24) %4) #9
  ret %"struct.std::__1::basic_string<char>::__rep"* %5
}

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr hidden nonnull align 8 dereferenceable(24) %"struct.std::__1::basic_string<char>::__rep"* @_ZNKSt3__122__compressed_pair_elemINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repELi0ELb0EE5__getEv(%"struct.std::__1::__compressed_pair_elem"* nonnull dereferenceable(24) %0) #0 align 2 {
  %2 = alloca %"struct.std::__1::__compressed_pair_elem"*, align 8
  store %"struct.std::__1::__compressed_pair_elem"* %0, %"struct.std::__1::__compressed_pair_elem"** %2, align 8
  %3 = load %"struct.std::__1::__compressed_pair_elem"*, %"struct.std::__1::__compressed_pair_elem"** %2, align 8
  %4 = getelementptr inbounds %"struct.std::__1::__compressed_pair_elem", %"struct.std::__1::__compressed_pair_elem"* %3, i32 0, i32 0
  ret %"struct.std::__1::basic_string<char>::__rep"* %4
}

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr hidden i8* @_ZNSt3__114pointer_traitsIPKcE10pointer_toERS1_(i8* nonnull align 1 dereferenceable(1) %0) #0 align 2 {
  %2 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  %3 = load i8*, i8** %2, align 8
  %4 = call i8* @_ZNSt3__19addressofIKcEEPT_RS2_(i8* nonnull align 1 dereferenceable(1) %3) #9
  ret i8* %4
}

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr hidden i8* @_ZNSt3__19addressofIKcEEPT_RS2_(i8* nonnull align 1 dereferenceable(1) %0) #0 {
  %2 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  %3 = load i8*, i8** %2, align 8
  ret i8* %3
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden void @_ZNSt3__119ostreambuf_iteratorIcNS_11char_traitsIcEEEC2ERNS_13basic_ostreamIcS2_EE(%"class.std::__1::ostreambuf_iterator"* nonnull dereferenceable(8) %0, %"class.std::__1::basic_ostream"* nonnull align 8 dereferenceable(8) %1) unnamed_addr #4 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %3 = alloca %"class.std::__1::ostreambuf_iterator"*, align 8
  %4 = alloca %"class.std::__1::basic_ostream"*, align 8
  store %"class.std::__1::ostreambuf_iterator"* %0, %"class.std::__1::ostreambuf_iterator"** %3, align 8
  store %"class.std::__1::basic_ostream"* %1, %"class.std::__1::basic_ostream"** %4, align 8
  %5 = load %"class.std::__1::ostreambuf_iterator"*, %"class.std::__1::ostreambuf_iterator"** %3, align 8
  %6 = bitcast %"class.std::__1::ostreambuf_iterator"* %5 to %"struct.std::__1::iterator"*
  %7 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", %"class.std::__1::ostreambuf_iterator"* %5, i32 0, i32 0
  %8 = load %"class.std::__1::basic_ostream"*, %"class.std::__1::basic_ostream"** %4, align 8
  %9 = bitcast %"class.std::__1::basic_ostream"* %8 to i8**
  %10 = load i8*, i8** %9, align 8
  %11 = getelementptr i8, i8* %10, i64 -24
  %12 = bitcast i8* %11 to i64*
  %13 = load i64, i64* %12, align 8
  %14 = bitcast %"class.std::__1::basic_ostream"* %8 to i8*
  %15 = getelementptr inbounds i8, i8* %14, i64 %13
  %16 = bitcast i8* %15 to %"class.std::__1::basic_ios"*
  %17 = invoke %"class.std::__1::basic_streambuf"* @_ZNKSt3__19basic_iosIcNS_11char_traitsIcEEE5rdbufEv(%"class.std::__1::basic_ios"* nonnull dereferenceable(148) %16)
          to label %18 unwind label %19

18:                                               ; preds = %2
  store %"class.std::__1::basic_streambuf"* %17, %"class.std::__1::basic_streambuf"** %7, align 8
  ret void

19:                                               ; preds = %2
  %20 = landingpad { i8*, i32 }
          catch i8* null
  %21 = extractvalue { i8*, i32 } %20, 0
  call void @__clang_call_terminate(i8* %21) #10
  unreachable
}

; Function Attrs: noinline optnone ssp uwtable mustprogress
define linkonce_odr hidden %"class.std::__1::basic_streambuf"* @_ZNKSt3__19basic_iosIcNS_11char_traitsIcEEE5rdbufEv(%"class.std::__1::basic_ios"* nonnull dereferenceable(148) %0) #2 align 2 {
  %2 = alloca %"class.std::__1::basic_ios"*, align 8
  store %"class.std::__1::basic_ios"* %0, %"class.std::__1::basic_ios"** %2, align 8
  %3 = load %"class.std::__1::basic_ios"*, %"class.std::__1::basic_ios"** %2, align 8
  %4 = bitcast %"class.std::__1::basic_ios"* %3 to %"class.std::__1::ios_base"*
  %5 = call i8* @_ZNKSt3__18ios_base5rdbufEv(%"class.std::__1::ios_base"* nonnull dereferenceable(136) %4)
  %6 = bitcast i8* %5 to %"class.std::__1::basic_streambuf"*
  ret %"class.std::__1::basic_streambuf"* %6
}

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr hidden i8* @_ZNKSt3__18ios_base5rdbufEv(%"class.std::__1::ios_base"* nonnull dereferenceable(136) %0) #0 align 2 {
  %2 = alloca %"class.std::__1::ios_base"*, align 8
  store %"class.std::__1::ios_base"* %0, %"class.std::__1::ios_base"** %2, align 8
  %3 = load %"class.std::__1::ios_base"*, %"class.std::__1::ios_base"** %2, align 8
  %4 = getelementptr inbounds %"class.std::__1::ios_base", %"class.std::__1::ios_base"* %3, i32 0, i32 6
  %5 = load i8*, i8** %4, align 8
  ret i8* %5
}

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr zeroext i1 @_ZNSt3__111char_traitsIcE11eq_int_typeEii(i32 %0, i32 %1) #0 align 2 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  store i32 %1, i32* %4, align 4
  %5 = load i32, i32* %3, align 4
  %6 = load i32, i32* %4, align 4
  %7 = icmp eq i32 %5, %6
  ret i1 %7
}

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr i32 @_ZNSt3__111char_traitsIcE3eofEv() #0 align 2 {
  ret i32 -1
}

; Function Attrs: noinline optnone ssp uwtable mustprogress
define linkonce_odr hidden signext i8 @_ZNKSt3__19basic_iosIcNS_11char_traitsIcEEE5widenEc(%"class.std::__1::basic_ios"* nonnull dereferenceable(148) %0, i8 signext %1) #2 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %3 = alloca %"class.std::__1::basic_ios"*, align 8
  %4 = alloca i8, align 1
  %5 = alloca %"class.std::__1::locale", align 8
  %6 = alloca i8*, align 8
  %7 = alloca i32, align 4
  store %"class.std::__1::basic_ios"* %0, %"class.std::__1::basic_ios"** %3, align 8
  store i8 %1, i8* %4, align 1
  %8 = load %"class.std::__1::basic_ios"*, %"class.std::__1::basic_ios"** %3, align 8
  %9 = bitcast %"class.std::__1::basic_ios"* %8 to %"class.std::__1::ios_base"*
  call void @_ZNKSt3__18ios_base6getlocEv(%"class.std::__1::locale"* sret(%"class.std::__1::locale") align 8 %5, %"class.std::__1::ios_base"* nonnull dereferenceable(136) %9)
  %10 = invoke nonnull align 8 dereferenceable(25) %"class.std::__1::ctype"* @_ZNSt3__19use_facetINS_5ctypeIcEEEERKT_RKNS_6localeE(%"class.std::__1::locale"* nonnull align 8 dereferenceable(8) %5)
          to label %11 unwind label %15

11:                                               ; preds = %2
  %12 = load i8, i8* %4, align 1
  %13 = invoke signext i8 @_ZNKSt3__15ctypeIcE5widenEc(%"class.std::__1::ctype"* nonnull dereferenceable(25) %10, i8 signext %12)
          to label %14 unwind label %15

14:                                               ; preds = %11
  call void @_ZNSt3__16localeD1Ev(%"class.std::__1::locale"* nonnull dereferenceable(8) %5) #9
  ret i8 %13

15:                                               ; preds = %11, %2
  %16 = landingpad { i8*, i32 }
          cleanup
  %17 = extractvalue { i8*, i32 } %16, 0
  store i8* %17, i8** %6, align 8
  %18 = extractvalue { i8*, i32 } %16, 1
  store i32 %18, i32* %7, align 4
  call void @_ZNSt3__16localeD1Ev(%"class.std::__1::locale"* nonnull dereferenceable(8) %5) #9
  br label %19

19:                                               ; preds = %15
  %20 = load i8*, i8** %6, align 8
  %21 = load i32, i32* %7, align 4
  %22 = insertvalue { i8*, i32 } undef, i8* %20, 0
  %23 = insertvalue { i8*, i32 } %22, i32 %21, 1
  resume { i8*, i32 } %23
}

; Function Attrs: noinline optnone ssp uwtable mustprogress
define linkonce_odr hidden nonnull align 8 dereferenceable(25) %"class.std::__1::ctype"* @_ZNSt3__19use_facetINS_5ctypeIcEEEERKT_RKNS_6localeE(%"class.std::__1::locale"* nonnull align 8 dereferenceable(8) %0) #2 {
  %2 = alloca %"class.std::__1::locale"*, align 8
  store %"class.std::__1::locale"* %0, %"class.std::__1::locale"** %2, align 8
  %3 = load %"class.std::__1::locale"*, %"class.std::__1::locale"** %2, align 8
  %4 = call %"class.std::__1::locale::facet"* @_ZNKSt3__16locale9use_facetERNS0_2idE(%"class.std::__1::locale"* nonnull dereferenceable(8) %3, %"class.std::__1::locale::id"* nonnull align 8 dereferenceable(12) @_ZNSt3__15ctypeIcE2idE)
  %5 = bitcast %"class.std::__1::locale::facet"* %4 to %"class.std::__1::ctype"*
  ret %"class.std::__1::ctype"* %5
}

declare void @_ZNKSt3__18ios_base6getlocEv(%"class.std::__1::locale"* sret(%"class.std::__1::locale") align 8, %"class.std::__1::ios_base"* nonnull dereferenceable(136)) #3

; Function Attrs: noinline optnone ssp uwtable mustprogress
define linkonce_odr hidden signext i8 @_ZNKSt3__15ctypeIcE5widenEc(%"class.std::__1::ctype"* nonnull dereferenceable(25) %0, i8 signext %1) #2 align 2 {
  %3 = alloca %"class.std::__1::ctype"*, align 8
  %4 = alloca i8, align 1
  store %"class.std::__1::ctype"* %0, %"class.std::__1::ctype"** %3, align 8
  store i8 %1, i8* %4, align 1
  %5 = load %"class.std::__1::ctype"*, %"class.std::__1::ctype"** %3, align 8
  %6 = load i8, i8* %4, align 1
  %7 = bitcast %"class.std::__1::ctype"* %5 to i8 (%"class.std::__1::ctype"*, i8)***
  %8 = load i8 (%"class.std::__1::ctype"*, i8)**, i8 (%"class.std::__1::ctype"*, i8)*** %7, align 8
  %9 = getelementptr inbounds i8 (%"class.std::__1::ctype"*, i8)*, i8 (%"class.std::__1::ctype"*, i8)** %8, i64 7
  %10 = load i8 (%"class.std::__1::ctype"*, i8)*, i8 (%"class.std::__1::ctype"*, i8)** %9, align 8
  %11 = call signext i8 %10(%"class.std::__1::ctype"* nonnull dereferenceable(25) %5, i8 signext %6)
  ret i8 %11
}

; Function Attrs: nounwind
declare void @_ZNSt3__16localeD1Ev(%"class.std::__1::locale"* nonnull dereferenceable(8)) unnamed_addr #5

declare %"class.std::__1::locale::facet"* @_ZNKSt3__16locale9use_facetERNS0_2idE(%"class.std::__1::locale"* nonnull dereferenceable(8), %"class.std::__1::locale::id"* nonnull align 8 dereferenceable(12)) #3

; Function Attrs: noinline optnone ssp uwtable mustprogress
define linkonce_odr hidden void @_ZNSt3__18ios_base8setstateEj(%"class.std::__1::ios_base"* nonnull dereferenceable(136) %0, i32 %1) #2 align 2 {
  %3 = alloca %"class.std::__1::ios_base"*, align 8
  %4 = alloca i32, align 4
  store %"class.std::__1::ios_base"* %0, %"class.std::__1::ios_base"** %3, align 8
  store i32 %1, i32* %4, align 4
  %5 = load %"class.std::__1::ios_base"*, %"class.std::__1::ios_base"** %3, align 8
  %6 = getelementptr inbounds %"class.std::__1::ios_base", %"class.std::__1::ios_base"* %5, i32 0, i32 4
  %7 = load i32, i32* %6, align 8
  %8 = load i32, i32* %4, align 4
  %9 = or i32 %7, %8
  call void @_ZNSt3__18ios_base5clearEj(%"class.std::__1::ios_base"* nonnull dereferenceable(136) %5, i32 %9)
  ret void
}

declare void @_ZNSt3__18ios_base5clearEj(%"class.std::__1::ios_base"* nonnull dereferenceable(136), i32) #3

; Function Attrs: nounwind
declare i64 @strlen(i8*) #5

attributes #0 = { noinline nounwind optnone ssp uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline norecurse optnone ssp uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noinline optnone ssp uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { noinline nounwind optnone ssp uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { noinline noreturn nounwind }
attributes #7 = { argmemonly nofree nosync nounwind willreturn }
attributes #8 = { noinline optnone ssp uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { nounwind }
attributes #10 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{!"clang version 12.0.0"}
!3 = distinct !{!3, !4}
!4 = !{!"llvm.loop.mustprogress"}
!5 = distinct !{!5, !4}
!6 = distinct !{!6, !4}
!7 = distinct !{!7, !4}
!8 = distinct !{!8, !4}
!9 = distinct !{!9, !4}
!10 = distinct !{!10, !4}
!11 = distinct !{!11, !4}
!12 = distinct !{!12, !4}
!13 = distinct !{!13, !4}
