; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@__asan_dtor_placeholder = internal constant i8 0
@__asan_ctor_placeholder = internal constant i8 0
@__asan_global_name.parent = private unnamed_addr constant [7 x i8] c"parent\00"
@__asan_module_name = private unnamed_addr constant [12 x i8] c"llvm_module\00"
@parent = global { [286 x i8], [130 x i8] } { [286 x i8] c"\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA\AA", [130 x i8] zeroinitializer }, align 32
@__asan_globals_registered = internal constant [1 x { ptr, i64, i64, ptr, ptr, i64, ptr, i64 }] [{ ptr, i64, i64, ptr, ptr, i64, ptr, i64 } { ptr @parent, i64 286, i64 416, ptr @__asan_global_name.parent, ptr @__asan_module_name, i64 0, ptr null, i64 0 }]
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @asan.module_ctor, ptr null }]
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @asan.module_dtor, ptr null }]

declare void @__asan_unregister_globals(ptr, i64)

declare void @__asan_register_globals(ptr, i64)

declare void @__asan_version_mismatch_check_v8()

declare void @__asan_init()

declare void @__asan_store1(ptr)

declare void @__asan_load1(ptr)

declare void @__asan_loadN(ptr, i64)

define void @use(i8 %0) {
  ret void
}

declare ptr @memset(ptr, i32, i64)

declare ptr @memcpy(ptr, ptr, i64)

declare void @exit(i32)

define i32 @f() {
  call void @__asan_loadN(ptr @parent, i64 8)
  call void @__asan_loadN(ptr getelementptr inbounds nuw (i8, ptr @parent, i64 278), i64 8)
  call void @__asan_load1(ptr @parent)
  %1 = load i8, ptr @parent, align 1
  call void @use(i8 %1)
  call void @__asan_load1(ptr getelementptr inbounds nuw (i8, ptr @parent, i64 278))
  %2 = load i8, ptr getelementptr inbounds nuw (i8, ptr @parent, i64 278), align 1
  call void @use(i8 %2)
  br label %3

3:                                                ; preds = %6, %0
  %4 = phi i64 [ %11, %6 ], [ 0, %0 ]
  %5 = icmp slt i64 %4, 278
  br i1 %5, label %6, label %12

6:                                                ; preds = %3
  %7 = sub i64 0, %4
  %8 = add i64 %7, 278
  %9 = getelementptr i8, ptr @parent, i64 %8
  call void @__asan_store1(ptr %9)
  %10 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @parent, i64 278), i64 %7
  store i8 -1, ptr %10, align 1
  %11 = add i64 %4, 1
  br label %3

12:                                               ; preds = %3
  br label %13

13:                                               ; preds = %16, %12
  %14 = phi i64 [ %20, %16 ], [ 0, %12 ]
  %15 = icmp slt i64 %14, 8
  br i1 %15, label %16, label %21

16:                                               ; preds = %13
  %17 = add i64 %14, 278
  %18 = getelementptr i8, ptr @parent, i64 %17
  call void @__asan_store1(ptr %18)
  %19 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @parent, i64 278), i64 %14
  store i8 -1, ptr %19, align 1
  %20 = add i64 %14, 1
  br label %13

21:                                               ; preds = %13
  call void @exit(i32 42)
  ret i32 0
}

define i32 @main() {
  %1 = call i32 @f()
  ret i32 0
}

define internal void @asan.module_ctor() {
  call void @__asan_init()
  call void @__asan_version_mismatch_check_v8()
  call void @__asan_register_globals(ptr @__asan_globals_registered, i64 1)
  ret void
}

define internal void @asan.module_dtor() {
  call void @__asan_unregister_globals(ptr @__asan_globals_registered, i64 1)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
