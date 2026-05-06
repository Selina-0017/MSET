module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"} {
  llvm.mlir.global internal constant @__asan_ctor_placeholder(0 : i8) {addr_space = 0 : i32, crisp.asan.generated} : i8
  llvm.func @__asan_version_mismatch_check_v8()
  llvm.func @__asan_init()
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @__asan_loadN(!llvm.ptr, i64)
  llvm.func @use(%arg0: i8) {
    llvm.return
  }
  llvm.func @memset(!llvm.ptr, i32, i64) -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func @memcpy(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func @exit(i32) attributes {sym_visibility = "private"}
  llvm.func @f() -> i32 {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(73 : index) : i64
    %3 = llvm.mlir.constant(-69 : i8) : i8
    %4 = llvm.mlir.constant(-86 : i8) : i8
    %5 = llvm.mlir.constant(8 : index) : i64
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.mlir.constant(73 : i64) : i64
    %8 = llvm.mlir.constant(42 : i32) : i32
    %9 = llvm.mlir.constant(0 : index) : i64
    %10 = llvm.getelementptr %0[81] : (!llvm.ptr) -> !llvm.ptr, i8
    %11 = llvm.ptrtoint %10 : !llvm.ptr to i64
    %12 = llvm.call @malloc(%11) : (i64) -> !llvm.ptr
    llvm.br ^bb1(%9 : i64)
  ^bb1(%13: i64):  // 2 preds: ^bb0, ^bb2
    %14 = llvm.icmp "slt" %13, %5 : i64
    llvm.cond_br %14, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %15 = llvm.getelementptr inbounds|nuw %12[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %4, %15 : i8, !llvm.ptr
    %16 = llvm.add %13, %6 : i64
    llvm.br ^bb1(%16 : i64)
  ^bb3:  // pred: ^bb1
    llvm.br ^bb4(%9 : i64)
  ^bb4(%17: i64):  // 2 preds: ^bb3, ^bb5
    %18 = llvm.icmp "slt" %17, %5 : i64
    llvm.cond_br %18, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %19 = llvm.getelementptr %12[73] : (!llvm.ptr) -> !llvm.ptr, i8
    %20 = llvm.getelementptr inbounds|nuw %19[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %3, %20 : i8, !llvm.ptr
    %21 = llvm.add %17, %6 : i64
    llvm.br ^bb4(%21 : i64)
  ^bb6:  // pred: ^bb4
    %22 = llvm.getelementptr %12[73] : (!llvm.ptr) -> !llvm.ptr, i8
    %23 = llvm.load %22 : !llvm.ptr -> i8
    llvm.call @use(%23) : (i8) -> ()
    %24 = llvm.load %12 : !llvm.ptr -> i8
    llvm.call @use(%24) : (i8) -> ()
    %25 = llvm.ptrtoint %12 : !llvm.ptr to i64
    %26 = llvm.inttoptr %25 : i64 to !llvm.ptr
    llvm.call @__asan_loadN(%26, %7) : (!llvm.ptr, i64) -> ()
    llvm.br ^bb7(%9 : i64)
  ^bb7(%27: i64):  // 2 preds: ^bb6, ^bb8
    %28 = llvm.icmp "slt" %27, %2 : i64
    llvm.cond_br %28, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %29 = llvm.getelementptr inbounds|nuw %12[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %30 = llvm.load %29 : !llvm.ptr -> i8
    llvm.call @use(%30) : (i8) -> ()
    %31 = llvm.add %27, %6 : i64
    llvm.br ^bb7(%31 : i64)
  ^bb9:  // pred: ^bb7
    llvm.br ^bb10(%9 : i64)
  ^bb10(%32: i64):  // 2 preds: ^bb9, ^bb11
    %33 = llvm.icmp "slt" %32, %5 : i64
    llvm.cond_br %33, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %34 = llvm.getelementptr inbounds|nuw %12[%32] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %35 = llvm.load %34 : !llvm.ptr -> i8
    llvm.call @use(%35) : (i8) -> ()
    %36 = llvm.add %32, %6 : i64
    llvm.br ^bb10(%36 : i64)
  ^bb12:  // pred: ^bb10
    llvm.call @exit(%8) : (i32) -> ()
    llvm.call @free(%12) : (!llvm.ptr) -> ()
    llvm.return %1 : i32
  }
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.call @f() : () -> i32
    llvm.return %0 : i32
  }
  llvm.func internal @asan.module_ctor() {
    llvm.call @__asan_init() : () -> ()
    llvm.call @__asan_version_mismatch_check_v8() : () -> ()
    llvm.return
  }
  llvm.mlir.global_ctors ctors = [@asan.module_ctor], priorities = [1 : i32], data = [@__asan_ctor_placeholder]
}

