module {
  memref.global @sink : memref<1xi8> = dense<0>

  func.func @use(%arg0: memref<1xi8>) -> () {
    func.return
  }

  func.func @main() -> i8 {
    %c0_index = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c123_i32 = arith.constant 123 : i32

    %ret_slot = memref.alloca() : memref<1xi32>
    memref.store %c0_i32, %ret_slot[%c0_index] : memref<1xi32>

    %offset_slot = memref.alloca() : memref<1xi64>
    %raw = memref.alloca() : memref<4096xi8>

    %raw_addr_index =
      memref.extract_aligned_pointer_as_index %raw
        : memref<4096xi8> -> index
    %raw_addr_i64 = arith.index_cast %raw_addr_index : index to i64

    memref.alloca_scope {
      %x = memref.alloca() : memref<1xi32>
      %x_addr_index =
        memref.extract_aligned_pointer_as_index %x
          : memref<1xi32> -> index
      %x_addr_i64 = arith.index_cast %x_addr_index : index to i64
      memref.store %c123_i32, %x[%c0_index] : memref<1xi32>
      %off_i64 = arith.subi %x_addr_i64, %raw_addr_i64 : i64
      memref.store %off_i64, %offset_slot[%c0_index] : memref<1xi64>
    }

    %off_i64_after = memref.load %offset_slot[%c0_index] : memref<1xi64>
    %off_after = arith.index_cast %off_i64_after : i64 to index

    // use-after-scope。
    %uaf = memref.load %raw[%off_after] : memref<4096xi8>

    %sink = memref.get_global @sink : memref<1xi8>
    memref.store %uaf, %sink[%c0_index] : memref<1xi8>
    func.call @use(%sink) : (memref<1xi8>) -> ()

    %ret = memref.load %sink[%c0_index] : memref<1xi8>
    func.return %ret : i8
  }
}
