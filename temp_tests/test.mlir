
module {
    memref.global @g_init : memref<1xi8> = uninitialized
    func.func @main() -> i8 {
        %c3_i8 = arith.constant 3 : i8
        %c2_i8 = arith.constant 2 : i8
        %c1_i8 = arith.constant 1 : i8
        %c0_i8 = arith.constant 0 : i8
        %c3 = arith.constant 3 : index
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %ff = arith.constant 0xff : i8
        // --- 1.int buf[3]={1,2,3};
        %alloc = memref.alloc() : memref<3xi8>
        memref.store %c1_i8, %alloc[%c0] : memref<3xi8>
        memref.store %c2_i8, %alloc[%c1] : memref<3xi8>
        memref.store %c3_i8, %alloc[%c2] : memref<3xi8>
        %g = memref.get_global @g_init : memref<1xi8>
        %ptr = memref.view %g[%c0][%c1] : memref<1xi8> to memref<?xi8>
        %p_ptr = memref.view %ptr[%c0][%c1] : memref<?xi8> to memref<?xi8>

        //--- ptr = 1 ---/
        memref.store %c1_i8, %ptr[%c0] : memref<?xi8>
        //--- p_ptr = &buf[3] ---/
        %buf3 = memref.load %alloc[%c3] : memref<3xi8>
        memref.store %buf3, %p_ptr[%c0] : memref<?xi8>
        //--- ptr = 0xff ---/
        memref.store %ff, %ptr[%c0] : memref<?xi8>

        return %c0_i8 : i8
    }
}