module {
  memref.global @sink : memref<1xi32> = uninitialized {alignment = 4 : i64}

  func.func @use(%arg0: memref<1xi32>) -> () {
    func.return
  }

  func.func @leak_stack() -> memref<1xi32> attributes {always_inline} {
    %c123_i32 = arith.constant 123 : i32
    %c0 = arith.constant 0 : index
    %x = memref.alloca() : memref<1xi32>
    memref.store %c123_i32, %x[%c0] : memref<1xi32>
    func.return %x : memref<1xi32>
  }

  func.func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %sink = memref.get_global @sink : memref<1xi32>
    %local_main = memref.alloca() : memref<1xi32>
    memref.store %c0_i32, %local_main[%c0] : memref<1xi32>
    %leaked = func.call @leak_stack() : () -> memref<1xi32>
    %v = memref.load %leaked[%c0] : memref<1xi32>
    memref.store %v, %sink[%c0] : memref<1xi32>
    func.call @use(%leaked) : (memref<1xi32>) -> ()
    %ret = memref.load %sink[%c0] : memref<1xi32>
    func.return %ret : i32
  }
}
