; ModuleID = 'refactored_file.cpp'
source_filename = "refactored_file.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_ZSt5alignmmRPvRm = comdat any

; Function Attrs: noinline optnone uwtable mustprogress
define dso_local void @_Z1fv() #0 {
entry:
  %a = alloca [256 x i32], align 16
  %b = alloca [256 x i32], align 16
  %c = alloca [256 x i32], align 16
  %i = alloca i32, align 4
  %ap = alloca i8*, align 8
  %bp = alloca i8*, align 8
  %cp = alloca i8*, align 8
  %sz = alloca i64, align 8
  %aligna = alloca i8*, align 8
  %alignb = alloca i8*, align 8
  %alignc = alloca i8*, align 8
  %0 = bitcast [256 x i32]* %c to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %0, i8 0, i64 1024, i1 false)
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %1 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %1, 256
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %2 = load i32, i32* %i, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds [256 x i32], [256 x i32]* %a, i64 0, i64 %idxprom
  store i32 1, i32* %arrayidx, align 4
  %3 = load i32, i32* %i, align 4
  %idxprom1 = sext i32 %3 to i64
  %arrayidx2 = getelementptr inbounds [256 x i32], [256 x i32]* %b, i64 0, i64 %idxprom1
  store i32 2, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %4 = load i32, i32* %i, align 4
  %inc = add nsw i32 %4, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond, !llvm.loop !2

for.end:                                          ; preds = %for.cond
  %arraydecay = getelementptr inbounds [256 x i32], [256 x i32]* %a, i64 0, i64 0
  %5 = bitcast i32* %arraydecay to i8*
  store i8* %5, i8** %ap, align 8
  %arraydecay3 = getelementptr inbounds [256 x i32], [256 x i32]* %b, i64 0, i64 0
  %6 = bitcast i32* %arraydecay3 to i8*
  store i8* %6, i8** %bp, align 8
  %arraydecay4 = getelementptr inbounds [256 x i32], [256 x i32]* %c, i64 0, i64 0
  %7 = bitcast i32* %arraydecay4 to i8*
  store i8* %7, i8** %cp, align 8
  store i64 256, i64* %sz, align 8
  %call = call i8* @_ZSt5alignmmRPvRm(i64 4, i64 4, i8** nonnull align 8 dereferenceable(8) %ap, i64* nonnull align 8 dereferenceable(8) %sz) #5
  store i8* %call, i8** %aligna, align 8
  %call5 = call i8* @_ZSt5alignmmRPvRm(i64 4, i64 4, i8** nonnull align 8 dereferenceable(8) %bp, i64* nonnull align 8 dereferenceable(8) %sz) #5
  store i8* %call5, i8** %alignb, align 8
  %call6 = call i8* @_ZSt5alignmmRPvRm(i64 4, i64 4, i8** nonnull align 8 dereferenceable(8) %cp, i64* nonnull align 8 dereferenceable(8) %sz) #5
  store i8* %call6, i8** %alignc, align 8
  %arraydecay7 = getelementptr inbounds [256 x i32], [256 x i32]* %a, i64 0, i64 0
  %8 = load i8*, i8** %aligna, align 8
  %9 = bitcast i8* %8 to i32*
  %arraydecay8 = getelementptr inbounds [256 x i32], [256 x i32]* %b, i64 0, i64 0
  %10 = load i8*, i8** %alignb, align 8
  %11 = bitcast i8* %10 to i32*
  %arraydecay9 = getelementptr inbounds [256 x i32], [256 x i32]* %c, i64 0, i64 0
  %12 = load i8*, i8** %alignc, align 8
  %13 = bitcast i8* %12 to i32*
  call void @forloop(i32* %arraydecay7, i32* %9, i64 0, i64 256, i64 1, i32* %arraydecay8, i32* %11, i64 0, i64 256, i64 1, i32* %arraydecay9, i32* %13, i64 0, i64 256, i64 1)
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #1

; Function Attrs: noinline nounwind optnone uwtable mustprogress
define linkonce_odr dso_local i8* @_ZSt5alignmmRPvRm(i64 %__align, i64 %__size, i8** nonnull align 8 dereferenceable(8) %__ptr, i64* nonnull align 8 dereferenceable(8) %__space) #2 comdat {
entry:
  %retval = alloca i8*, align 8
  %__align.addr = alloca i64, align 8
  %__size.addr = alloca i64, align 8
  %__ptr.addr = alloca i8**, align 8
  %__space.addr = alloca i64*, align 8
  %__intptr = alloca i64, align 8
  %__aligned = alloca i64, align 8
  %__diff = alloca i64, align 8
  store i64 %__align, i64* %__align.addr, align 8
  store i64 %__size, i64* %__size.addr, align 8
  store i8** %__ptr, i8*** %__ptr.addr, align 8
  store i64* %__space, i64** %__space.addr, align 8
  %0 = load i8**, i8*** %__ptr.addr, align 8
  %1 = load i8*, i8** %0, align 8
  %2 = ptrtoint i8* %1 to i64
  store i64 %2, i64* %__intptr, align 8
  %3 = load i64, i64* %__intptr, align 8
  %sub = sub i64 %3, 1
  %4 = load i64, i64* %__align.addr, align 8
  %add = add i64 %sub, %4
  %5 = load i64, i64* %__align.addr, align 8
  %sub1 = sub i64 0, %5
  %and = and i64 %add, %sub1
  store i64 %and, i64* %__aligned, align 8
  %6 = load i64, i64* %__aligned, align 8
  %7 = load i64, i64* %__intptr, align 8
  %sub2 = sub i64 %6, %7
  store i64 %sub2, i64* %__diff, align 8
  %8 = load i64, i64* %__size.addr, align 8
  %9 = load i64, i64* %__diff, align 8
  %add3 = add i64 %8, %9
  %10 = load i64*, i64** %__space.addr, align 8
  %11 = load i64, i64* %10, align 8
  %cmp = icmp ugt i64 %add3, %11
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i8* null, i8** %retval, align 8
  br label %return

if.else:                                          ; preds = %entry
  %12 = load i64, i64* %__diff, align 8
  %13 = load i64*, i64** %__space.addr, align 8
  %14 = load i64, i64* %13, align 8
  %sub4 = sub i64 %14, %12
  store i64 %sub4, i64* %13, align 8
  %15 = load i64, i64* %__aligned, align 8
  %16 = inttoptr i64 %15 to i8*
  %17 = load i8**, i8*** %__ptr.addr, align 8
  store i8* %16, i8** %17, align 8
  store i8* %16, i8** %retval, align 8
  br label %return

return:                                           ; preds = %if.else, %if.then
  %18 = load i8*, i8** %retval, align 8
  ret i8* %18
}

declare dso_local void @forloop(i32*, i32*, i64, i64, i64, i32*, i32*, i64, i64, i64, i32*, i32*, i64, i64, i64) #3

; Function Attrs: noinline norecurse optnone uwtable mustprogress
define dso_local i32 @main(i32 %argc, i8** %argv) #4 {
entry:
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  store i32 %argc, i32* %argc.addr, align 4
  store i8** %argv, i8*** %argv.addr, align 8
  call void @_Z1fv()
  ret i32 0
}

attributes #0 = { noinline optnone uwtable mustprogress "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn writeonly }
attributes #2 = { noinline nounwind optnone uwtable mustprogress "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { noinline norecurse optnone uwtable mustprogress "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 13.0.0 (https://github.com/parsabee/llvm-project.git 87afefcd22c53f7bdc68b5a13492e7f2bfc9837a)"}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.mustprogress"}
