; ModuleID = 'axpy.cpp_refactored.cpp'
source_filename = "axpy.cpp_refactored.cpp"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

; Function Attrs: noinline optnone ssp uwtable mustprogress
define dso_local void @_Z4axpyfPfS_S_(float %0, float* %1, float* %2, float* %3) #0 {
  %5 = alloca float, align 4
  %6 = alloca float*, align 8
  %7 = alloca float*, align 8
  %8 = alloca float*, align 8
  store float %0, float* %5, align 4
  store float* %1, float** %6, align 8
  store float* %2, float** %7, align 8
  store float* %3, float** %8, align 8
  %9 = load float, float* %5, align 4
  %10 = load float*, float** %8, align 8
  %11 = load float*, float** %8, align 8
  %12 = load float*, float** %6, align 8
  %13 = load float*, float** %6, align 8
  %14 = load float*, float** %7, align 8
  %15 = load float*, float** %7, align 8
  call void @__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_22_3(float %9, float* %10, float* %11, i64 0, i64 256, i64 1, float* %12, float* %13, i64 0, i64 256, i64 1, float* %14, float* %15, i64 0, i64 256, i64 1)
  ret void
}

declare void @__forloop__Users_parsabagheri_Development_llvm_project_tuner_Benchmarks_SPAPT_axpy_axpy_cpp_22_3(float, float*, float*, i64, i64, i64, float*, float*, i64, i64, i64, float*, float*, i64, i64, i64) #1

; Function Attrs: noinline norecurse optnone ssp uwtable mustprogress
define dso_local i32 @main() #2 {
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

declare i32 @rand() #1

; Function Attrs: noinline optnone ssp uwtable mustprogress
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
  br label %6, !llvm.loop !3

18:                                               ; preds = %6
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define linkonce_odr zeroext i1 @_Z6verifyIfLm256EEbT_PS0_S1_S1_(float %0, float* %1, float* %2, float* %3) #3 {
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
  br label %13, !llvm.loop !5

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
  br label %34, !llvm.loop !6

51:                                               ; preds = %34
  store i1 true, i1* %5, align 1
  br label %52

52:                                               ; preds = %51, %46
  %53 = load i1, i1* %5, align 1
  ret i1 %53
}

declare void @srand(i32) #1

declare i64 @time(i64*) #1

attributes #0 = { noinline optnone ssp uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noinline norecurse optnone ssp uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { noinline nounwind optnone ssp uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{!"clang version 12.0.0"}
!3 = distinct !{!3, !4}
!4 = !{!"llvm.loop.mustprogress"}
!5 = distinct !{!5, !4}
!6 = distinct !{!6, !4}
