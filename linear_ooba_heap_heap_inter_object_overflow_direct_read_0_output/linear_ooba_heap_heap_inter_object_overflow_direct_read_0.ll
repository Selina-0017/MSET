; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@__asan_ctor_placeholder = internal constant i8 0
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @asan.module_ctor, ptr null }]

declare void @__asan_version_mismatch_check_v8()

declare void @__asan_init()

declare void @free(ptr)

declare ptr @malloc(i64)

declare void @__asan_loadN(ptr, i64)

define void @use(i8 %0) {
  ret void
}

declare ptr @memset(ptr, i32, i64)

declare ptr @memcpy(ptr, ptr, i64)

declare void @exit(i32)

define i32 @f() {
  %1 = call ptr @malloc(i64 81)
  br label %2

2:                                                ; preds = %5, %0
  %3 = phi i64 [ %7, %5 ], [ 0, %0 ]
  %4 = icmp slt i64 %3, 8
  br i1 %4, label %5, label %8

5:                                                ; preds = %2
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 %3
  store i8 -86, ptr %6, align 1
  %7 = add i64 %3, 1
  br label %2

8:                                                ; preds = %2
  br label %9

9:                                                ; preds = %12, %8
  %10 = phi i64 [ %15, %12 ], [ 0, %8 ]
  %11 = icmp slt i64 %10, 8
  br i1 %11, label %12, label %16

12:                                               ; preds = %9
  %13 = getelementptr i8, ptr %1, i32 73
  %14 = getelementptr inbounds nuw i8, ptr %13, i64 %10
  store i8 -69, ptr %14, align 1
  %15 = add i64 %10, 1
  br label %9

16:                                               ; preds = %9
  %17 = getelementptr i8, ptr %1, i32 73
  %18 = load i8, ptr %17, align 1
  call void @use(i8 %18)
  %19 = load i8, ptr %1, align 1
  call void @use(i8 %19)
  %20 = ptrtoint ptr %1 to i64
  %21 = inttoptr i64 %20 to ptr
  call void @__asan_loadN(ptr %21, i64 73)
  br label %22

22:                                               ; preds = %25, %16
  %23 = phi i64 [ %28, %25 ], [ 0, %16 ]
  %24 = icmp slt i64 %23, 73
  br i1 %24, label %25, label %29

25:                                               ; preds = %22
  %26 = getelementptr inbounds nuw i8, ptr %1, i64 %23
  %27 = load i8, ptr %26, align 1
  call void @use(i8 %27)
  %28 = add i64 %23, 1
  br label %22

29:                                               ; preds = %22
  br label %30

30:                                               ; preds = %33, %29
  %31 = phi i64 [ %36, %33 ], [ 0, %29 ]
  %32 = icmp slt i64 %31, 8
  br i1 %32, label %33, label %37

33:                                               ; preds = %30
  %34 = getelementptr inbounds nuw i8, ptr %1, i64 %31
  %35 = load i8, ptr %34, align 1
  call void @use(i8 %35)
  %36 = add i64 %31, 1
  br label %30

37:                                               ; preds = %30
  call void @exit(i32 42)
  call void @free(ptr %1)
  ret i32 0
}

define i32 @main() {
  %1 = call i32 @f()
  ret i32 0
}

define internal void @asan.module_ctor() {
  call void @__asan_init()
  call void @__asan_version_mismatch_check_v8()
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
