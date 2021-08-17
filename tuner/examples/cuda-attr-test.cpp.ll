; ModuleID = '/var/folders/gd/yyt093w94d521q2gfssh1kp40000gn/T/temp-refactored-file-dd2bc2..cpp'
source_filename = "/var/folders/gd/yyt093w94d521q2gfssh1kp40000gn/T/temp-refactored-file-dd2bc2..cpp"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1

; Function Attrs: noinline nounwind optnone ssp uwtable mustprogress
define void @_Z1fv() #0 !dbg !359 {
entry:
  %a = alloca [256 x i32], align 16
  %b = alloca [256 x i32], align 16
  %c = alloca [256 x i32], align 16
  %i = alloca i32, align 4
  %i8 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata [256 x i32]* %a, metadata !361, metadata !DIExpression()), !dbg !365
  call void @llvm.dbg.declare(metadata [256 x i32]* %b, metadata !366, metadata !DIExpression()), !dbg !367
  call void @llvm.dbg.declare(metadata [256 x i32]* %c, metadata !368, metadata !DIExpression()), !dbg !369
  %0 = bitcast [256 x i32]* %c to i8*, !dbg !369
  call void @llvm.memset.p0i8.i64(i8* align 16 %0, i8 0, i64 1024, i1 false), !dbg !369
  call void @llvm.dbg.declare(metadata i32* %i, metadata !370, metadata !DIExpression()), !dbg !372
  store i32 0, i32* %i, align 4, !dbg !372
  br label %for.cond, !dbg !373

for.cond:                                         ; preds = %for.inc, %entry
  %1 = load i32, i32* %i, align 4, !dbg !374
  %cmp = icmp slt i32 %1, 256, !dbg !376
  br i1 %cmp, label %for.body, label %for.end, !dbg !377

for.body:                                         ; preds = %for.cond
  %2 = load i32, i32* %i, align 4, !dbg !378
  %idxprom = sext i32 %2 to i64, !dbg !380
  %arrayidx = getelementptr inbounds [256 x i32], [256 x i32]* %a, i64 0, i64 %idxprom, !dbg !380
  store i32 1, i32* %arrayidx, align 4, !dbg !381
  %3 = load i32, i32* %i, align 4, !dbg !382
  %idxprom1 = sext i32 %3 to i64, !dbg !383
  %arrayidx2 = getelementptr inbounds [256 x i32], [256 x i32]* %b, i64 0, i64 %idxprom1, !dbg !383
  store i32 2, i32* %arrayidx2, align 4, !dbg !384
  br label %for.inc, !dbg !385

for.inc:                                          ; preds = %for.body
  %4 = load i32, i32* %i, align 4, !dbg !386
  %inc = add nsw i32 %4, 1, !dbg !386
  store i32 %inc, i32* %i, align 4, !dbg !386
  br label %for.cond, !dbg !387, !llvm.loop !388

for.end:                                          ; preds = %for.cond
  %arraydecay = getelementptr inbounds [256 x i32], [256 x i32]* %a, i64 0, i64 0, !dbg !391
  %arraydecay3 = getelementptr inbounds [256 x i32], [256 x i32]* %a, i64 0, i64 0, !dbg !392
  %arraydecay4 = getelementptr inbounds [256 x i32], [256 x i32]* %b, i64 0, i64 0, !dbg !393
  %arraydecay5 = getelementptr inbounds [256 x i32], [256 x i32]* %b, i64 0, i64 0, !dbg !394
  %arraydecay6 = getelementptr inbounds [256 x i32], [256 x i32]* %c, i64 0, i64 0, !dbg !395
  %arraydecay7 = getelementptr inbounds [256 x i32], [256 x i32]* %c, i64 0, i64 0, !dbg !396
  call void bitcast (void (i32*, i32*, i64, i64, i64, i32*, i32*, i64, i64, i64, i32*, i32*, i64, i64, i64)* @__forloop__Users_parsabagheri_Development_llvm_project_tuner_examples_cuda_attr_test_cpp_16_3 to void (i32*, i32*, i64, i64, i64, i32*, i32*, i64, i64, i64, i32*, i32*, i64, i64, i64)*)(i32* %arraydecay, i32* %arraydecay3, i64 0, i64 256, i64 1, i32* %arraydecay4, i32* %arraydecay5, i64 0, i64 256, i64 1, i32* %arraydecay6, i32* %arraydecay7, i64 0, i64 256, i64 1), !dbg !397
  call void @llvm.dbg.declare(metadata i32* %i8, metadata !398, metadata !DIExpression()), !dbg !400
  store i32 0, i32* %i8, align 4, !dbg !400
  br label %for.cond9, !dbg !401

for.cond9:                                        ; preds = %for.inc14, %for.end
  %5 = load i32, i32* %i8, align 4, !dbg !402
  %cmp10 = icmp slt i32 %5, 256, !dbg !404
  br i1 %cmp10, label %for.body11, label %for.end15, !dbg !405

for.body11:                                       ; preds = %for.cond9
  %6 = load i32, i32* %i8, align 4, !dbg !406
  %idxprom12 = sext i32 %6 to i64, !dbg !408
  %arrayidx13 = getelementptr inbounds [256 x i32], [256 x i32]* %c, i64 0, i64 %idxprom12, !dbg !408
  %7 = load i32, i32* %arrayidx13, align 4, !dbg !408
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0), i32 %7), !dbg !409
  br label %for.inc14, !dbg !410

for.inc14:                                        ; preds = %for.body11
  %8 = load i32, i32* %i8, align 4, !dbg !411
  %add = add nsw i32 %8, 1, !dbg !411
  store i32 %add, i32* %i8, align 4, !dbg !411
  br label %for.cond9, !dbg !412, !llvm.loop !413

for.end15:                                        ; preds = %for.cond9
  ret void, !dbg !415
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #2

declare i32 @printf(i8*, ...) #3

; Function Attrs: noinline norecurse nounwind optnone ssp uwtable mustprogress
define i32 @main(i32 %argc, i8** %argv) #4 !dbg !416 {
entry:
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !419, metadata !DIExpression()), !dbg !420
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !421, metadata !DIExpression()), !dbg !422
  call void @_Z1fv(), !dbg !423
  ret i32 0, !dbg !424
}

define void @__forloop__Users_parsabagheri_Development_llvm_project_tuner_examples_cuda_attr_test_cpp_16_3(i32* %0, i32* %1, i64 %2, i64 %3, i64 %4, i32* %5, i32* %6, i64 %7, i64 %8, i64 %9, i32* %10, i32* %11, i64 %12, i64 %13, i64 %14) !dbg !425 {
  %16 = insertvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } undef, i32* %0, 0, !dbg !429
  %17 = insertvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } %16, i32* %1, 1, !dbg !429
  %18 = insertvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } %17, i64 %2, 2, !dbg !429
  %19 = insertvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } %18, i64 %3, 3, 0, !dbg !429
  %20 = insertvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } %19, i64 %4, 4, 0, !dbg !429
  %21 = insertvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } undef, i32* %5, 0, !dbg !429
  %22 = insertvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } %21, i32* %6, 1, !dbg !429
  %23 = insertvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } %22, i64 %7, 2, !dbg !429
  %24 = insertvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } %23, i64 %8, 3, 0, !dbg !429
  %25 = insertvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } %24, i64 %9, 4, 0, !dbg !429
  %26 = insertvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } undef, i32* %10, 0, !dbg !429
  %27 = insertvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } %26, i32* %11, 1, !dbg !429
  %28 = insertvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } %27, i64 %12, 2, !dbg !429
  %29 = insertvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } %28, i64 %13, 3, 0, !dbg !429
  %30 = insertvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } %29, i64 %14, 4, 0, !dbg !429
  br label %31, !dbg !431

31:                                               ; preds = %56, %15
  %32 = phi i64 [ %57, %56 ], [ 0, %15 ]
  %33 = icmp slt i64 %32, 256, !dbg !431
  br i1 %33, label %34, label %58, !dbg !431

34:                                               ; preds = %31
  br label %35, !dbg !432

35:                                               ; preds = %54, %34
  %36 = phi i64 [ %55, %54 ], [ 0, %34 ]
  %37 = icmp slt i64 %36, 256, !dbg !432
  br i1 %37, label %38, label %56, !dbg !432

38:                                               ; preds = %35
  br label %39, !dbg !433

39:                                               ; preds = %42, %38
  %40 = phi i64 [ %53, %42 ], [ 0, %38 ]
  %41 = icmp slt i64 %40, 1, !dbg !433
  br i1 %41, label %42, label %54, !dbg !433

42:                                               ; preds = %39
  %43 = add i64 %40, %36, !dbg !434
  %44 = extractvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } %20, 1, !dbg !435
  %45 = getelementptr i32, i32* %44, i64 %43, !dbg !435
  %46 = load i32, i32* %45, align 4, !dbg !435
  %47 = extractvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } %25, 1, !dbg !436
  %48 = getelementptr i32, i32* %47, i64 %43, !dbg !436
  %49 = load i32, i32* %48, align 4, !dbg !436
  %50 = add i32 %46, %49, !dbg !437
  %51 = extractvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } %30, 1, !dbg !438
  %52 = getelementptr i32, i32* %51, i64 %43, !dbg !438
  store i32 %50, i32* %52, align 4, !dbg !438
  %53 = add i64 %40, 1, !dbg !433
  br label %39, !dbg !433

54:                                               ; preds = %39
  %55 = add i64 %36, 1, !dbg !432
  br label %35, !dbg !432

56:                                               ; preds = %35
  %57 = add i64 %32, 1, !dbg !431
  br label %31, !dbg !431

58:                                               ; preds = %31
  ret void, !dbg !439
}

attributes #0 = { noinline nounwind optnone ssp uwtable mustprogress "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { argmemonly nofree nosync nounwind willreturn writeonly }
attributes #3 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #4 = { noinline norecurse nounwind optnone ssp uwtable mustprogress "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5}
!llvm.dbg.cu = !{!6, !356}
!llvm.linker.options = !{}
!llvm.ident = !{!358}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 15]}
!1 = !{i32 7, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"PIC Level", i32 2}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !7, producer: "clang version 13.0.0 (https://github.com/parsabee/llvm-project.git f12c995b2adcdb8f69500a45366a12c2aa5f0db6)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !8, imports: !9, splitDebugInlining: false, nameTableKind: None, sysroot: "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.15.sdk", sdk: "MacOSX10.15.sdk")
!7 = !DIFile(filename: "/var/folders/gd/yyt093w94d521q2gfssh1kp40000gn/T/temp-refactored-file-dd2bc2..cpp", directory: "/Users/parsabagheri/Development/llvm-project/tuner/cmake-build-debug")
!8 = !{}
!9 = !{!10, !17, !20, !24, !30, !38, !44, !51, !59, !63, !67, !71, !77, !82, !86, !90, !94, !98, !103, !107, !112, !118, !122, !126, !130, !134, !139, !143, !145, !149, !151, !160, !164, !169, !173, !175, !179, !183, !185, !189, !195, !199, !203, !209, !214, !218, !221, !224, !228, !232, !235, !238, !241, !243, !245, !247, !249, !251, !253, !255, !257, !259, !261, !263, !265, !267, !269, !271, !275, !278, !281, !284, !286, !291, !293, !297, !301, !303, !305, !309, !313, !317, !319, !323, !328, !332, !336, !338, !340, !342, !344, !346, !348, !352}
!10 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !13, file: !16, line: 50)
!11 = !DINamespace(name: "__1", scope: !12, exportSymbols: true)
!12 = !DINamespace(name: "std", scope: null)
!13 = !DIDerivedType(tag: DW_TAG_typedef, name: "ptrdiff_t", file: !14, line: 51, baseType: !15)
!14 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/11.0.0/include/stddef.h", directory: "")
!15 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!16 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/../include/c++/v1/cstddef", directory: "")
!17 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !18, file: !16, line: 51)
!18 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !14, line: 62, baseType: !19)
!19 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!20 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !21, file: !16, line: 56)
!21 = !DIDerivedType(tag: DW_TAG_typedef, name: "max_align_t", file: !22, line: 32, baseType: !23)
!22 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/11.0.0/include/__stddef_max_align_t.h", directory: "")
!23 = !DIBasicType(name: "long double", size: 128, encoding: DW_ATE_float)
!24 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !25, file: !29, line: 100)
!25 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !26, line: 31, baseType: !27)
!26 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/sys/_types/_size_t.h", directory: "")
!27 = !DIDerivedType(tag: DW_TAG_typedef, name: "__darwin_size_t", file: !28, line: 92, baseType: !19)
!28 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/i386/_types.h", directory: "")
!29 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/../include/c++/v1/cstdlib", directory: "")
!30 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !31, file: !29, line: 101)
!31 = !DIDerivedType(tag: DW_TAG_typedef, name: "div_t", file: !32, line: 86, baseType: !33)
!32 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/stdlib.h", directory: "")
!33 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !32, line: 83, size: 64, flags: DIFlagTypePassByValue, elements: !34, identifier: "_ZTS5div_t")
!34 = !{!35, !37}
!35 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !33, file: !32, line: 84, baseType: !36, size: 32)
!36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!37 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !33, file: !32, line: 85, baseType: !36, size: 32, offset: 32)
!38 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !39, file: !29, line: 102)
!39 = !DIDerivedType(tag: DW_TAG_typedef, name: "ldiv_t", file: !32, line: 91, baseType: !40)
!40 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !32, line: 88, size: 128, flags: DIFlagTypePassByValue, elements: !41, identifier: "_ZTS6ldiv_t")
!41 = !{!42, !43}
!42 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !40, file: !32, line: 89, baseType: !15, size: 64)
!43 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !40, file: !32, line: 90, baseType: !15, size: 64, offset: 64)
!44 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !45, file: !29, line: 104)
!45 = !DIDerivedType(tag: DW_TAG_typedef, name: "lldiv_t", file: !32, line: 97, baseType: !46)
!46 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !32, line: 94, size: 128, flags: DIFlagTypePassByValue, elements: !47, identifier: "_ZTS7lldiv_t")
!47 = !{!48, !50}
!48 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !46, file: !32, line: 95, baseType: !49, size: 64)
!49 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!50 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !46, file: !32, line: 96, baseType: !49, size: 64, offset: 64)
!51 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !52, file: !29, line: 106)
!52 = !DISubprogram(name: "atof", scope: !32, file: !32, line: 134, type: !53, flags: DIFlagPrototyped, spFlags: 0)
!53 = !DISubroutineType(types: !54)
!54 = !{!55, !56}
!55 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!56 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !57, size: 64)
!57 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !58)
!58 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!59 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !60, file: !29, line: 107)
!60 = !DISubprogram(name: "atoi", scope: !32, file: !32, line: 135, type: !61, flags: DIFlagPrototyped, spFlags: 0)
!61 = !DISubroutineType(types: !62)
!62 = !{!36, !56}
!63 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !64, file: !29, line: 108)
!64 = !DISubprogram(name: "atol", scope: !32, file: !32, line: 136, type: !65, flags: DIFlagPrototyped, spFlags: 0)
!65 = !DISubroutineType(types: !66)
!66 = !{!15, !56}
!67 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !68, file: !29, line: 110)
!68 = !DISubprogram(name: "atoll", scope: !32, file: !32, line: 139, type: !69, flags: DIFlagPrototyped, spFlags: 0)
!69 = !DISubroutineType(types: !70)
!70 = !{!49, !56}
!71 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !72, file: !29, line: 112)
!72 = !DISubprogram(name: "strtod", linkageName: "\01_strtod", scope: !32, file: !32, line: 165, type: !73, flags: DIFlagPrototyped, spFlags: 0)
!73 = !DISubroutineType(types: !74)
!74 = !{!55, !56, !75}
!75 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !76, size: 64)
!76 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !58, size: 64)
!77 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !78, file: !29, line: 113)
!78 = !DISubprogram(name: "strtof", linkageName: "\01_strtof", scope: !32, file: !32, line: 166, type: !79, flags: DIFlagPrototyped, spFlags: 0)
!79 = !DISubroutineType(types: !80)
!80 = !{!81, !56, !75}
!81 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!82 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !83, file: !29, line: 114)
!83 = !DISubprogram(name: "strtold", scope: !32, file: !32, line: 169, type: !84, flags: DIFlagPrototyped, spFlags: 0)
!84 = !DISubroutineType(types: !85)
!85 = !{!23, !56, !75}
!86 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !87, file: !29, line: 115)
!87 = !DISubprogram(name: "strtol", scope: !32, file: !32, line: 167, type: !88, flags: DIFlagPrototyped, spFlags: 0)
!88 = !DISubroutineType(types: !89)
!89 = !{!15, !56, !75, !36}
!90 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !91, file: !29, line: 117)
!91 = !DISubprogram(name: "strtoll", scope: !32, file: !32, line: 172, type: !92, flags: DIFlagPrototyped, spFlags: 0)
!92 = !DISubroutineType(types: !93)
!93 = !{!49, !56, !75, !36}
!94 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !95, file: !29, line: 119)
!95 = !DISubprogram(name: "strtoul", scope: !32, file: !32, line: 175, type: !96, flags: DIFlagPrototyped, spFlags: 0)
!96 = !DISubroutineType(types: !97)
!97 = !{!19, !56, !75, !36}
!98 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !99, file: !29, line: 121)
!99 = !DISubprogram(name: "strtoull", scope: !32, file: !32, line: 178, type: !100, flags: DIFlagPrototyped, spFlags: 0)
!100 = !DISubroutineType(types: !101)
!101 = !{!102, !56, !75, !36}
!102 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!103 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !104, file: !29, line: 126)
!104 = !DISubprogram(name: "rand", scope: !32, file: !32, line: 162, type: !105, flags: DIFlagPrototyped, spFlags: 0)
!105 = !DISubroutineType(types: !106)
!106 = !{!36}
!107 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !108, file: !29, line: 127)
!108 = !DISubprogram(name: "srand", scope: !32, file: !32, line: 164, type: !109, flags: DIFlagPrototyped, spFlags: 0)
!109 = !DISubroutineType(types: !110)
!110 = !{null, !111}
!111 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!112 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !113, file: !29, line: 128)
!113 = !DISubprogram(name: "calloc", scope: !114, file: !114, line: 41, type: !115, flags: DIFlagPrototyped, spFlags: 0)
!114 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/malloc/_malloc.h", directory: "")
!115 = !DISubroutineType(types: !116)
!116 = !{!117, !25, !25}
!117 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!118 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !119, file: !29, line: 129)
!119 = !DISubprogram(name: "free", scope: !114, file: !114, line: 42, type: !120, flags: DIFlagPrototyped, spFlags: 0)
!120 = !DISubroutineType(types: !121)
!121 = !{null, !117}
!122 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !123, file: !29, line: 130)
!123 = !DISubprogram(name: "malloc", scope: !114, file: !114, line: 40, type: !124, flags: DIFlagPrototyped, spFlags: 0)
!124 = !DISubroutineType(types: !125)
!125 = !{!117, !25}
!126 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !127, file: !29, line: 131)
!127 = !DISubprogram(name: "realloc", scope: !114, file: !114, line: 43, type: !128, flags: DIFlagPrototyped, spFlags: 0)
!128 = !DISubroutineType(types: !129)
!129 = !{!117, !117, !25}
!130 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !131, file: !29, line: 135)
!131 = !DISubprogram(name: "abort", scope: !32, file: !32, line: 131, type: !132, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!132 = !DISubroutineType(types: !133)
!133 = !{null}
!134 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !135, file: !29, line: 139)
!135 = !DISubprogram(name: "atexit", scope: !32, file: !32, line: 133, type: !136, flags: DIFlagPrototyped, spFlags: 0)
!136 = !DISubroutineType(types: !137)
!137 = !{!36, !138}
!138 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !132, size: 64)
!139 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !140, file: !29, line: 140)
!140 = !DISubprogram(name: "exit", scope: !32, file: !32, line: 145, type: !141, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!141 = !DISubroutineType(types: !142)
!142 = !{null, !36}
!143 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !144, file: !29, line: 141)
!144 = !DISubprogram(name: "_Exit", scope: !32, file: !32, line: 198, type: !141, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!145 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !146, file: !29, line: 143)
!146 = !DISubprogram(name: "getenv", scope: !32, file: !32, line: 147, type: !147, flags: DIFlagPrototyped, spFlags: 0)
!147 = !DISubroutineType(types: !148)
!148 = !{!76, !56}
!149 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !150, file: !29, line: 144)
!150 = !DISubprogram(name: "system", linkageName: "\01_system", scope: !32, file: !32, line: 190, type: !61, flags: DIFlagPrototyped, spFlags: 0)
!151 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !152, file: !29, line: 149)
!152 = !DISubprogram(name: "bsearch", scope: !32, file: !32, line: 141, type: !153, flags: DIFlagPrototyped, spFlags: 0)
!153 = !DISubroutineType(types: !154)
!154 = !{!117, !155, !155, !25, !25, !157}
!155 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !156, size: 64)
!156 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!157 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !158, size: 64)
!158 = !DISubroutineType(types: !159)
!159 = !{!36, !155, !155}
!160 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !161, file: !29, line: 150)
!161 = !DISubprogram(name: "qsort", scope: !32, file: !32, line: 160, type: !162, flags: DIFlagPrototyped, spFlags: 0)
!162 = !DISubroutineType(types: !163)
!163 = !{null, !117, !25, !25, !157}
!164 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !165, file: !29, line: 151)
!165 = !DISubprogram(name: "abs", linkageName: "_ZL3absx", scope: !166, file: !166, line: 113, type: !167, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!166 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/../include/c++/v1/stdlib.h", directory: "")
!167 = !DISubroutineType(types: !168)
!168 = !{!49, !49}
!169 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !170, file: !29, line: 152)
!170 = !DISubprogram(name: "labs", scope: !32, file: !32, line: 148, type: !171, flags: DIFlagPrototyped, spFlags: 0)
!171 = !DISubroutineType(types: !172)
!172 = !{!15, !15}
!173 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !174, file: !29, line: 154)
!174 = !DISubprogram(name: "llabs", scope: !32, file: !32, line: 152, type: !167, flags: DIFlagPrototyped, spFlags: 0)
!175 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !176, file: !29, line: 156)
!176 = !DISubprogram(name: "div", linkageName: "_ZL3divxx", scope: !166, file: !166, line: 118, type: !177, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!177 = !DISubroutineType(types: !178)
!178 = !{!45, !49, !49}
!179 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !180, file: !29, line: 157)
!180 = !DISubprogram(name: "ldiv", scope: !32, file: !32, line: 149, type: !181, flags: DIFlagPrototyped, spFlags: 0)
!181 = !DISubroutineType(types: !182)
!182 = !{!39, !15, !15}
!183 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !184, file: !29, line: 159)
!184 = !DISubprogram(name: "lldiv", scope: !32, file: !32, line: 153, type: !177, flags: DIFlagPrototyped, spFlags: 0)
!185 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !186, file: !29, line: 161)
!186 = !DISubprogram(name: "mblen", scope: !32, file: !32, line: 156, type: !187, flags: DIFlagPrototyped, spFlags: 0)
!187 = !DISubroutineType(types: !188)
!188 = !{!36, !56, !25}
!189 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !190, file: !29, line: 162)
!190 = !DISubprogram(name: "mbtowc", scope: !32, file: !32, line: 158, type: !191, flags: DIFlagPrototyped, spFlags: 0)
!191 = !DISubroutineType(types: !192)
!192 = !{!36, !193, !56, !25}
!193 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !194, size: 64)
!194 = !DIBasicType(name: "wchar_t", size: 32, encoding: DW_ATE_signed)
!195 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !196, file: !29, line: 163)
!196 = !DISubprogram(name: "wctomb", scope: !32, file: !32, line: 195, type: !197, flags: DIFlagPrototyped, spFlags: 0)
!197 = !DISubroutineType(types: !198)
!198 = !{!36, !76, !194}
!199 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !200, file: !29, line: 164)
!200 = !DISubprogram(name: "mbstowcs", scope: !32, file: !32, line: 157, type: !201, flags: DIFlagPrototyped, spFlags: 0)
!201 = !DISubroutineType(types: !202)
!202 = !{!25, !193, !56, !25}
!203 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !204, file: !29, line: 165)
!204 = !DISubprogram(name: "wcstombs", scope: !32, file: !32, line: 194, type: !205, flags: DIFlagPrototyped, spFlags: 0)
!205 = !DISubroutineType(types: !206)
!206 = !{!25, !76, !207, !25}
!207 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !208, size: 64)
!208 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !194)
!209 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !210, file: !213, line: 153)
!210 = !DIDerivedType(tag: DW_TAG_typedef, name: "int8_t", file: !211, line: 30, baseType: !212)
!211 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/sys/_types/_int8_t.h", directory: "")
!212 = !DIBasicType(name: "signed char", size: 8, encoding: DW_ATE_signed_char)
!213 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/../include/c++/v1/cstdint", directory: "")
!214 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !215, file: !213, line: 154)
!215 = !DIDerivedType(tag: DW_TAG_typedef, name: "int16_t", file: !216, line: 30, baseType: !217)
!216 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/sys/_types/_int16_t.h", directory: "")
!217 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!218 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !219, file: !213, line: 155)
!219 = !DIDerivedType(tag: DW_TAG_typedef, name: "int32_t", file: !220, line: 30, baseType: !36)
!220 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/sys/_types/_int32_t.h", directory: "")
!221 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !222, file: !213, line: 156)
!222 = !DIDerivedType(tag: DW_TAG_typedef, name: "int64_t", file: !223, line: 30, baseType: !49)
!223 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/sys/_types/_int64_t.h", directory: "")
!224 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !225, file: !213, line: 158)
!225 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint8_t", file: !226, line: 31, baseType: !227)
!226 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/_types/_uint8_t.h", directory: "")
!227 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!228 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !229, file: !213, line: 159)
!229 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint16_t", file: !230, line: 31, baseType: !231)
!230 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/_types/_uint16_t.h", directory: "")
!231 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
!232 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !233, file: !213, line: 160)
!233 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint32_t", file: !234, line: 31, baseType: !111)
!234 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/_types/_uint32_t.h", directory: "")
!235 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !236, file: !213, line: 161)
!236 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint64_t", file: !237, line: 31, baseType: !102)
!237 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/_types/_uint64_t.h", directory: "")
!238 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !239, file: !213, line: 163)
!239 = !DIDerivedType(tag: DW_TAG_typedef, name: "int_least8_t", file: !240, line: 29, baseType: !210)
!240 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/stdint.h", directory: "")
!241 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !242, file: !213, line: 164)
!242 = !DIDerivedType(tag: DW_TAG_typedef, name: "int_least16_t", file: !240, line: 30, baseType: !215)
!243 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !244, file: !213, line: 165)
!244 = !DIDerivedType(tag: DW_TAG_typedef, name: "int_least32_t", file: !240, line: 31, baseType: !219)
!245 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !246, file: !213, line: 166)
!246 = !DIDerivedType(tag: DW_TAG_typedef, name: "int_least64_t", file: !240, line: 32, baseType: !222)
!247 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !248, file: !213, line: 168)
!248 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint_least8_t", file: !240, line: 33, baseType: !225)
!249 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !250, file: !213, line: 169)
!250 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint_least16_t", file: !240, line: 34, baseType: !229)
!251 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !252, file: !213, line: 170)
!252 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint_least32_t", file: !240, line: 35, baseType: !233)
!253 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !254, file: !213, line: 171)
!254 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint_least64_t", file: !240, line: 36, baseType: !236)
!255 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !256, file: !213, line: 173)
!256 = !DIDerivedType(tag: DW_TAG_typedef, name: "int_fast8_t", file: !240, line: 40, baseType: !210)
!257 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !258, file: !213, line: 174)
!258 = !DIDerivedType(tag: DW_TAG_typedef, name: "int_fast16_t", file: !240, line: 41, baseType: !215)
!259 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !260, file: !213, line: 175)
!260 = !DIDerivedType(tag: DW_TAG_typedef, name: "int_fast32_t", file: !240, line: 42, baseType: !219)
!261 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !262, file: !213, line: 176)
!262 = !DIDerivedType(tag: DW_TAG_typedef, name: "int_fast64_t", file: !240, line: 43, baseType: !222)
!263 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !264, file: !213, line: 178)
!264 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint_fast8_t", file: !240, line: 44, baseType: !225)
!265 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !266, file: !213, line: 179)
!266 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint_fast16_t", file: !240, line: 45, baseType: !229)
!267 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !268, file: !213, line: 180)
!268 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint_fast32_t", file: !240, line: 46, baseType: !233)
!269 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !270, file: !213, line: 181)
!270 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint_fast64_t", file: !240, line: 47, baseType: !236)
!271 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !272, file: !213, line: 183)
!272 = !DIDerivedType(tag: DW_TAG_typedef, name: "intptr_t", file: !273, line: 32, baseType: !274)
!273 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/sys/_types/_intptr_t.h", directory: "")
!274 = !DIDerivedType(tag: DW_TAG_typedef, name: "__darwin_intptr_t", file: !28, line: 49, baseType: !15)
!275 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !276, file: !213, line: 184)
!276 = !DIDerivedType(tag: DW_TAG_typedef, name: "uintptr_t", file: !277, line: 30, baseType: !19)
!277 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/sys/_types/_uintptr_t.h", directory: "")
!278 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !279, file: !213, line: 186)
!279 = !DIDerivedType(tag: DW_TAG_typedef, name: "intmax_t", file: !280, line: 32, baseType: !15)
!280 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/_types/_intmax_t.h", directory: "")
!281 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !282, file: !213, line: 187)
!282 = !DIDerivedType(tag: DW_TAG_typedef, name: "uintmax_t", file: !283, line: 32, baseType: !19)
!283 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/_types/_uintmax_t.h", directory: "")
!284 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !25, file: !285, line: 69)
!285 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/../include/c++/v1/cstring", directory: "")
!286 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !287, file: !285, line: 70)
!287 = !DISubprogram(name: "memcpy", scope: !288, file: !288, line: 72, type: !289, flags: DIFlagPrototyped, spFlags: 0)
!288 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/string.h", directory: "")
!289 = !DISubroutineType(types: !290)
!290 = !{!117, !117, !155, !25}
!291 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !292, file: !285, line: 71)
!292 = !DISubprogram(name: "memmove", scope: !288, file: !288, line: 73, type: !289, flags: DIFlagPrototyped, spFlags: 0)
!293 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !294, file: !285, line: 72)
!294 = !DISubprogram(name: "strcpy", scope: !288, file: !288, line: 79, type: !295, flags: DIFlagPrototyped, spFlags: 0)
!295 = !DISubroutineType(types: !296)
!296 = !{!76, !76, !56}
!297 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !298, file: !285, line: 73)
!298 = !DISubprogram(name: "strncpy", scope: !288, file: !288, line: 85, type: !299, flags: DIFlagPrototyped, spFlags: 0)
!299 = !DISubroutineType(types: !300)
!300 = !{!76, !76, !56, !25}
!301 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !302, file: !285, line: 74)
!302 = !DISubprogram(name: "strcat", scope: !288, file: !288, line: 75, type: !295, flags: DIFlagPrototyped, spFlags: 0)
!303 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !304, file: !285, line: 75)
!304 = !DISubprogram(name: "strncat", scope: !288, file: !288, line: 83, type: !299, flags: DIFlagPrototyped, spFlags: 0)
!305 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !306, file: !285, line: 76)
!306 = !DISubprogram(name: "memcmp", scope: !288, file: !288, line: 71, type: !307, flags: DIFlagPrototyped, spFlags: 0)
!307 = !DISubroutineType(types: !308)
!308 = !{!36, !155, !155, !25}
!309 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !310, file: !285, line: 77)
!310 = !DISubprogram(name: "strcmp", scope: !288, file: !288, line: 77, type: !311, flags: DIFlagPrototyped, spFlags: 0)
!311 = !DISubroutineType(types: !312)
!312 = !{!36, !56, !56}
!313 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !314, file: !285, line: 78)
!314 = !DISubprogram(name: "strncmp", scope: !288, file: !288, line: 84, type: !315, flags: DIFlagPrototyped, spFlags: 0)
!315 = !DISubroutineType(types: !316)
!316 = !{!36, !56, !56, !25}
!317 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !318, file: !285, line: 79)
!318 = !DISubprogram(name: "strcoll", scope: !288, file: !288, line: 78, type: !311, flags: DIFlagPrototyped, spFlags: 0)
!319 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !320, file: !285, line: 80)
!320 = !DISubprogram(name: "strxfrm", scope: !288, file: !288, line: 91, type: !321, flags: DIFlagPrototyped, spFlags: 0)
!321 = !DISubroutineType(types: !322)
!322 = !{!25, !76, !56, !25}
!323 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !324, file: !285, line: 81)
!324 = !DISubprogram(name: "memchr", linkageName: "_ZL6memchrUa9enable_ifILb1EEPvim", scope: !325, file: !325, line: 99, type: !326, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!325 = !DIFile(filename: "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/../include/c++/v1/string.h", directory: "")
!326 = !DISubroutineType(types: !327)
!327 = !{!117, !117, !36, !25}
!328 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !329, file: !285, line: 82)
!329 = !DISubprogram(name: "strchr", linkageName: "_ZL6strchrUa9enable_ifILb1EEPci", scope: !325, file: !325, line: 78, type: !330, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!330 = !DISubroutineType(types: !331)
!331 = !{!76, !76, !36}
!332 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !333, file: !285, line: 83)
!333 = !DISubprogram(name: "strcspn", scope: !288, file: !288, line: 80, type: !334, flags: DIFlagPrototyped, spFlags: 0)
!334 = !DISubroutineType(types: !335)
!335 = !{!25, !56, !56}
!336 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !337, file: !285, line: 84)
!337 = !DISubprogram(name: "strpbrk", linkageName: "_ZL7strpbrkUa9enable_ifILb1EEPcPKc", scope: !325, file: !325, line: 85, type: !295, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!338 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !339, file: !285, line: 85)
!339 = !DISubprogram(name: "strrchr", linkageName: "_ZL7strrchrUa9enable_ifILb1EEPci", scope: !325, file: !325, line: 92, type: !330, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!340 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !341, file: !285, line: 86)
!341 = !DISubprogram(name: "strspn", scope: !288, file: !288, line: 88, type: !334, flags: DIFlagPrototyped, spFlags: 0)
!342 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !343, file: !285, line: 87)
!343 = !DISubprogram(name: "strstr", linkageName: "_ZL6strstrUa9enable_ifILb1EEPcPKc", scope: !325, file: !325, line: 106, type: !295, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!344 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !345, file: !285, line: 89)
!345 = !DISubprogram(name: "strtok", scope: !288, file: !288, line: 90, type: !295, flags: DIFlagPrototyped, spFlags: 0)
!346 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !347, file: !285, line: 91)
!347 = !DISubprogram(name: "memset", scope: !288, file: !288, line: 74, type: !326, flags: DIFlagPrototyped, spFlags: 0)
!348 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !349, file: !285, line: 92)
!349 = !DISubprogram(name: "strerror", linkageName: "\01_strerror", scope: !288, file: !288, line: 81, type: !350, flags: DIFlagPrototyped, spFlags: 0)
!350 = !DISubroutineType(types: !351)
!351 = !{!76, !36}
!352 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !11, entity: !353, file: !285, line: 93)
!353 = !DISubprogram(name: "strlen", scope: !288, file: !288, line: 82, type: !354, flags: DIFlagPrototyped, spFlags: 0)
!354 = !DISubroutineType(types: !355)
!355 = !{!25, !56}
!356 = distinct !DICompileUnit(language: DW_LANG_C, file: !357, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!357 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!358 = !{!"clang version 13.0.0 (https://github.com/parsabee/llvm-project.git f12c995b2adcdb8f69500a45366a12c2aa5f0db6)"}
!359 = distinct !DISubprogram(name: "f", linkageName: "_Z1fv", scope: !360, file: !360, line: 7, type: !132, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !8)
!360 = !DIFile(filename: "/var/folders/gd/yyt093w94d521q2gfssh1kp40000gn/T/temp-refactored-file-dd2bc2..cpp", directory: "")
!361 = !DILocalVariable(name: "a", scope: !359, file: !360, line: 8, type: !362)
!362 = !DICompositeType(tag: DW_TAG_array_type, baseType: !36, size: 8192, elements: !363)
!363 = !{!364}
!364 = !DISubrange(count: 256)
!365 = !DILocation(line: 8, column: 7, scope: !359)
!366 = !DILocalVariable(name: "b", scope: !359, file: !360, line: 9, type: !362)
!367 = !DILocation(line: 9, column: 7, scope: !359)
!368 = !DILocalVariable(name: "c", scope: !359, file: !360, line: 10, type: !362)
!369 = !DILocation(line: 10, column: 7, scope: !359)
!370 = !DILocalVariable(name: "i", scope: !371, file: !360, line: 11, type: !36)
!371 = distinct !DILexicalBlock(scope: !359, file: !360, line: 11, column: 3)
!372 = !DILocation(line: 11, column: 12, scope: !371)
!373 = !DILocation(line: 11, column: 8, scope: !371)
!374 = !DILocation(line: 11, column: 19, scope: !375)
!375 = distinct !DILexicalBlock(scope: !371, file: !360, line: 11, column: 3)
!376 = !DILocation(line: 11, column: 21, scope: !375)
!377 = !DILocation(line: 11, column: 3, scope: !371)
!378 = !DILocation(line: 12, column: 7, scope: !379)
!379 = distinct !DILexicalBlock(scope: !375, file: !360, line: 11, column: 31)
!380 = !DILocation(line: 12, column: 5, scope: !379)
!381 = !DILocation(line: 12, column: 10, scope: !379)
!382 = !DILocation(line: 13, column: 7, scope: !379)
!383 = !DILocation(line: 13, column: 5, scope: !379)
!384 = !DILocation(line: 13, column: 10, scope: !379)
!385 = !DILocation(line: 14, column: 3, scope: !379)
!386 = !DILocation(line: 11, column: 27, scope: !375)
!387 = !DILocation(line: 11, column: 3, scope: !375)
!388 = distinct !{!388, !377, !389, !390}
!389 = !DILocation(line: 14, column: 3, scope: !371)
!390 = !{!"llvm.loop.mustprogress"}
!391 = !DILocation(line: 16, column: 97, scope: !359)
!392 = !DILocation(line: 16, column: 100, scope: !359)
!393 = !DILocation(line: 16, column: 114, scope: !359)
!394 = !DILocation(line: 16, column: 117, scope: !359)
!395 = !DILocation(line: 16, column: 131, scope: !359)
!396 = !DILocation(line: 16, column: 134, scope: !359)
!397 = !DILocation(line: 16, column: 3, scope: !359)
!398 = !DILocalVariable(name: "i", scope: !399, file: !360, line: 19, type: !36)
!399 = distinct !DILexicalBlock(scope: !359, file: !360, line: 19, column: 3)
!400 = !DILocation(line: 19, column: 12, scope: !399)
!401 = !DILocation(line: 19, column: 8, scope: !399)
!402 = !DILocation(line: 19, column: 19, scope: !403)
!403 = distinct !DILexicalBlock(scope: !399, file: !360, line: 19, column: 3)
!404 = !DILocation(line: 19, column: 21, scope: !403)
!405 = !DILocation(line: 19, column: 3, scope: !399)
!406 = !DILocation(line: 20, column: 22, scope: !407)
!407 = distinct !DILexicalBlock(scope: !403, file: !360, line: 19, column: 33)
!408 = !DILocation(line: 20, column: 20, scope: !407)
!409 = !DILocation(line: 20, column: 5, scope: !407)
!410 = !DILocation(line: 21, column: 3, scope: !407)
!411 = !DILocation(line: 19, column: 27, scope: !403)
!412 = !DILocation(line: 19, column: 3, scope: !403)
!413 = distinct !{!413, !405, !414, !390}
!414 = !DILocation(line: 21, column: 3, scope: !399)
!415 = !DILocation(line: 22, column: 1, scope: !359)
!416 = distinct !DISubprogram(name: "main", scope: !360, file: !360, line: 24, type: !417, scopeLine: 24, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !8)
!417 = !DISubroutineType(types: !418)
!418 = !{!36, !36, !75}
!419 = !DILocalVariable(name: "argc", arg: 1, scope: !416, file: !360, line: 24, type: !36)
!420 = !DILocation(line: 24, column: 14, scope: !416)
!421 = !DILocalVariable(name: "argv", arg: 2, scope: !416, file: !360, line: 24, type: !75)
!422 = !DILocation(line: 24, column: 27, scope: !416)
!423 = !DILocation(line: 25, column: 3, scope: !416)
!424 = !DILocation(line: 26, column: 1, scope: !416)
!425 = distinct !DISubprogram(name: "__forloop__Users_parsabagheri_Development_llvm_project_tuner_examples_cuda_attr_test_cpp_16_3", linkageName: "__forloop__Users_parsabagheri_Development_llvm_project_tuner_examples_cuda_attr_test_cpp_16_3", scope: null, file: !426, line: 2, type: !427, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !356, retainedNodes: !428)
!426 = !DIFile(filename: "/var/folders/gd/yyt093w94d521q2gfssh1kp40000gn/T/tmp-opt-result-file-37da43.mlir", directory: "")
!427 = !DISubroutineType(types: !428)
!428 = !{}
!429 = !DILocation(line: 2, column: 3, scope: !430)
!430 = !DILexicalBlockFile(scope: !425, file: !426, discriminator: 0)
!431 = !DILocation(line: 6, column: 5, scope: !430)
!432 = !DILocation(line: 13, column: 7, scope: !430)
!433 = !DILocation(line: 14, column: 9, scope: !430)
!434 = !DILocation(line: 15, column: 16, scope: !430)
!435 = !DILocation(line: 16, column: 16, scope: !430)
!436 = !DILocation(line: 17, column: 16, scope: !430)
!437 = !DILocation(line: 18, column: 16, scope: !430)
!438 = !DILocation(line: 19, column: 11, scope: !430)
!439 = !DILocation(line: 26, column: 5, scope: !430)
