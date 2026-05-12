module {
  memref.global "private" constant @__const_leak_stack_s
      : memref<8xi8> = dense<[97, 98, 99, 0, 0, 0, 0, 0]>

  memref.global "private" constant @str
      : memref<1xi8> = dense<[0]>

  func.func @use(%arg0: memref<?xi8>) -> () {
    func.return
  }

  func.func private @strcat(!llvm.ptr, !llvm.ptr) -> !llvm.ptr

  func.func @leak_stack() -> memref<8xi8> {
    %src = memref.get_global @__const_leak_stack_s : memref<8xi8>
    %dst = memref.alloca() {alignment = 1 : i64} : memref<8xi8>
    memref.copy %src, %dst : memref<8xi8> to memref<8xi8>
    return %dst : memref<8xi8>
  }

  func.func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %zero_i32 = arith.constant 0 : i32

    %ret_slot = memref.alloca() {alignment = 4 : i64} : memref<1xi32>
    memref.store %zero_i32, %ret_slot[%c0] : memref<1xi32>

    // p = leak_stack()
    %p = func.call @leak_stack() : () -> memref<8xi8>

    %p_dyn = memref.cast %p : memref<8xi8> to memref<?xi8>
    func.call @use(%p_dyn) : (memref<?xi8>) -> ()

    %empty = memref.get_global @str : memref<1xi8>

    // ---- boundary: memref<8xi8> -> !llvm.ptr ----
    %pi = memref.extract_aligned_pointer_as_index %p
        : memref<8xi8> -> index
    %pi64 = arith.index_cast %pi : index to i64
    %pp = llvm.inttoptr %pi64 : i64 to !llvm.ptr

    // ---- boundary: memref<1xi8> -> !llvm.ptr ----
    %ei = memref.extract_aligned_pointer_as_index %empty
        : memref<1xi8> -> index
    %ei64 = arith.index_cast %ei : index to i64
    %ep = llvm.inttoptr %ei64 : i64 to !llvm.ptr

    // strcat(p, "")
    %ignored = func.call @strcat(%pp, %ep)
        : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr

    return %zero_i32 : i32
  }
}
