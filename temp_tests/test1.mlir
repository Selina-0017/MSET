module {
    memref.global @a : memref<5xi32> = uninitialized

    func.func @b(%arg0: i32) {
        %c7_i32 = arith.constant 7 : i32
        %c0_i32 = arith.constant 0 : i32
        %c5 = arith.constant 5 : index
        %0 = arith.cmpi ne, %arg0, %c0_i32 : i32
        // ----1. if(x)----
        scf.if %0 {
            // -----2. a[5]=7---
            %1 = memref.get_global @a : memref<5xi32>
            memref.store %c7_i32, %1[%c5] : memref<5xi32>
        }
        return
    }

    func.func @main() -> i32 {
        %c1 = arith.constant 1 : i32
        func.call @b(%c1) : (i32) -> ()
        return %c1 : i32
    }
}