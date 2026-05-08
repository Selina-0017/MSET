
module {
    memref.global @b_init : memref<2xi8> = uninitialized
    memref.global constant @k_init : memref<1xi8> = dense<0>
    func.func @main() -> i8 {
        %b = memref.get_global @b_init : memref<2xi8>
        %c0_i8 = arith.constant 0 : i8
        %c2_i8 = arith.constant 2 : i8
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        // --- 3. *c = b, *d = b ---
        %c_view = memref.view %b[%c0][%c1] : memref<2xi8> to memref<?xi8>
        %d_view = memref.view %b[%c0][%c1] : memref<2xi8> to memref<?xi8>
        // --- 4. *c = *b ---
        %b0_val = memref.load %b[%c0] : memref<2xi8>
        memref.store %b0_val, %c_view[%c0] : memref<?xi8>
        // --- 5. k=2 ---
        %k = memref.get_global @k_init : memref<1xi8>
        memref.store %c2_i8, %k[%c0] : memref<1xi8>
        // --- 6. * (d+k) ---
        %val = memref.load %k[%c0] : memref<1xi8>
        %idx = arith.index_cast %val : i8 to index
        %dk_view = memref.view %d_view[%idx][%c1] : memref<?xi8> to memref<?xi8>
        // --- 7. return c->x ---
        %ret = memref.load %c_view[%c0] : memref<?xi8>
        func.return %ret : i8
    }
}