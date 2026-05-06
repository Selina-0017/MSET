module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"} {
  llvm.mlir.global internal constant @__asan_dtor_placeholder(0 : i8) {addr_space = 0 : i32, crisp.asan.generated} : i8
  llvm.mlir.global internal constant @__asan_ctor_placeholder(0 : i8) {addr_space = 0 : i32, crisp.asan.generated} : i8
  llvm.mlir.global private unnamed_addr constant @__asan_global_name.parent("parent\00") {addr_space = 0 : i32, crisp.asan.generated}
  llvm.mlir.global private unnamed_addr constant @__asan_module_name("llvm_module\00") {addr_space = 0 : i32, crisp.asan.generated}
  llvm.func @__asan_unregister_globals(!llvm.ptr, i64)
  llvm.func @__asan_register_globals(!llvm.ptr, i64)
  llvm.func @__asan_version_mismatch_check_v8()
  llvm.func @__asan_init()
  llvm.func @__asan_store1(!llvm.ptr)
  llvm.func @__asan_load1(!llvm.ptr)
  llvm.func @__asan_loadN(!llvm.ptr, i64)
  llvm.func @use(%arg0: i8) {
    llvm.return
  }
  llvm.func @memset(!llvm.ptr, i32, i64) -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func @memcpy(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func @exit(i32) attributes {sym_visibility = "private"}
  llvm.mlir.global external @parent() {addr_space = 0 : i32, alignment = 32 : i64, crisp.asan.wrapped} : !llvm.struct<(array<286 x i8>, array<130 x i8>)> {
    %0 = llvm.mlir.constant(dense<-86> : tensor<286xi8>) : !llvm.array<286 x i8>
    %1 = llvm.mlir.zero : !llvm.array<130 x i8>
    %2 = llvm.mlir.undef : !llvm.struct<(array<286 x i8>, array<130 x i8>)>
    %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(array<286 x i8>, array<130 x i8>)> 
    %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(array<286 x i8>, array<130 x i8>)> 
    llvm.return %4 : !llvm.struct<(array<286 x i8>, array<130 x i8>)>
  }
  llvm.func @f() -> i32 {
    %0 = llvm.mlir.addressof @parent : !llvm.ptr
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(8 : index) : i64
    %3 = llvm.mlir.constant(-1 : i8) : i8
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.mlir.constant(278 : index) : i64
    %6 = llvm.mlir.constant(8 : i64) : i64
    %7 = llvm.mlir.constant(42 : i32) : i32
    %8 = llvm.mlir.constant(0 : index) : i64
    %9 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<286 x i8>
    %10 = llvm.ptrtoint %9 : !llvm.ptr to i64
    %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
    llvm.call @__asan_loadN(%11, %6) : (!llvm.ptr, i64) -> ()
    %12 = llvm.getelementptr %11[278] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.call @__asan_loadN(%12, %6) : (!llvm.ptr, i64) -> ()
    llvm.call @__asan_load1(%11) : (!llvm.ptr) -> ()
    %13 = llvm.load %9 : !llvm.ptr -> i8
    llvm.call @use(%13) : (i8) -> ()
    llvm.call @__asan_load1(%12) : (!llvm.ptr) -> ()
    %14 = llvm.getelementptr %9[278] : (!llvm.ptr) -> !llvm.ptr, i8
    %15 = llvm.load %14 : !llvm.ptr -> i8
    llvm.call @use(%15) : (i8) -> ()
    llvm.br ^bb1(%8 : i64)
  ^bb1(%16: i64):  // 2 preds: ^bb0, ^bb2
    %17 = llvm.icmp "slt" %16, %5 : i64
    llvm.cond_br %17, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %18 = llvm.sub %8, %16 : i64
    %19 = llvm.add %18, %5 : i64
    %20 = llvm.getelementptr %11[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.call @__asan_store1(%20) : (!llvm.ptr) -> ()
    %21 = llvm.getelementptr inbounds|nuw %14[%18] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %3, %21 : i8, !llvm.ptr
    %22 = llvm.add %16, %4 : i64
    llvm.br ^bb1(%22 : i64)
  ^bb3:  // pred: ^bb1
    llvm.br ^bb4(%8 : i64)
  ^bb4(%23: i64):  // 2 preds: ^bb3, ^bb5
    %24 = llvm.icmp "slt" %23, %2 : i64
    llvm.cond_br %24, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %25 = llvm.add %23, %5 : i64
    %26 = llvm.getelementptr %11[%25] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.call @__asan_store1(%26) : (!llvm.ptr) -> ()
    %27 = llvm.getelementptr inbounds|nuw %14[%23] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %3, %27 : i8, !llvm.ptr
    %28 = llvm.add %23, %4 : i64
    llvm.br ^bb4(%28 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @exit(%7) : (i32) -> ()
    llvm.return %1 : i32
  }
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.call @f() : () -> i32
    llvm.return %0 : i32
  }
  llvm.mlir.global internal constant @__asan_globals_registered() {addr_space = 0 : i32, crisp.asan.generated} : !llvm.array<1 x struct<(ptr, i64, i64, ptr, ptr, i64, ptr, i64)>> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.addressof @__asan_module_name : !llvm.ptr
    %3 = llvm.mlir.addressof @__asan_global_name.parent : !llvm.ptr
    %4 = llvm.mlir.constant(416 : i64) : i64
    %5 = llvm.mlir.constant(286 : i64) : i64
    %6 = llvm.mlir.undef : !llvm.array<1 x struct<(ptr, i64, i64, ptr, ptr, i64, ptr, i64)>>
    %7 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i64, ptr, ptr, i64, ptr, i64)>
    %8 = llvm.mlir.addressof @parent : !llvm.ptr
    %9 = llvm.insertvalue %8, %7[0] : !llvm.struct<(ptr, i64, i64, ptr, ptr, i64, ptr, i64)> 
    %10 = llvm.insertvalue %5, %9[1] : !llvm.struct<(ptr, i64, i64, ptr, ptr, i64, ptr, i64)> 
    %11 = llvm.insertvalue %4, %10[2] : !llvm.struct<(ptr, i64, i64, ptr, ptr, i64, ptr, i64)> 
    %12 = llvm.insertvalue %3, %11[3] : !llvm.struct<(ptr, i64, i64, ptr, ptr, i64, ptr, i64)> 
    %13 = llvm.insertvalue %2, %12[4] : !llvm.struct<(ptr, i64, i64, ptr, ptr, i64, ptr, i64)> 
    %14 = llvm.insertvalue %1, %13[5] : !llvm.struct<(ptr, i64, i64, ptr, ptr, i64, ptr, i64)> 
    %15 = llvm.insertvalue %0, %14[6] : !llvm.struct<(ptr, i64, i64, ptr, ptr, i64, ptr, i64)> 
    %16 = llvm.insertvalue %1, %15[7] : !llvm.struct<(ptr, i64, i64, ptr, ptr, i64, ptr, i64)> 
    %17 = llvm.insertvalue %16, %6[0] : !llvm.array<1 x struct<(ptr, i64, i64, ptr, ptr, i64, ptr, i64)>> 
    llvm.return %17 : !llvm.array<1 x struct<(ptr, i64, i64, ptr, ptr, i64, ptr, i64)>>
  }
  llvm.func internal @asan.module_ctor() {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.addressof @__asan_globals_registered : !llvm.ptr
    llvm.call @__asan_init() : () -> ()
    llvm.call @__asan_version_mismatch_check_v8() : () -> ()
    llvm.call @__asan_register_globals(%1, %0) : (!llvm.ptr, i64) -> ()
    llvm.return
  }
  llvm.func internal @asan.module_dtor() {
    %0 = llvm.mlir.addressof @__asan_globals_registered : !llvm.ptr
    %1 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @__asan_unregister_globals(%0, %1) : (!llvm.ptr, i64) -> ()
    llvm.return
  }
  llvm.mlir.global_ctors ctors = [@asan.module_ctor], priorities = [1 : i32], data = [@__asan_ctor_placeholder]
  llvm.mlir.global_dtors dtors = [@asan.module_dtor], priorities = [1 : i32], data = [@__asan_dtor_placeholder]
}

