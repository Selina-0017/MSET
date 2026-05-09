module {
  memref.global @a : memref<1xi32> = uninitialized

  func.func @b() {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c = memref.alloca() : memref<1xi32>
    memref.store %c0_i32, %c[%c0] : memref<1xi32>
    return
  }

  func.func @make_f() -> memref<1xi32> {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %f = memref.alloca() : memref<1xi32>
    memref.store %c0_i32, %f[%c0] : memref<1xi32>
    return %f : memref<1xi32>
  }

  func.func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32

    %c0_idx = arith.constant 0 : index
    %c6_idx = arith.constant 6 : index
    %c1_idx = arith.constant 1 : index

    %a_ref = memref.get_global @a : memref<1xi32>

    // int d[1] = {1};
    // int *e = d;
    %d = memref.alloca() : memref<1xi32>
    memref.store %c1_i32, %d[%c0] : memref<1xi32>

    // a = 0;
    memref.store %c0_i32, %a_ref[%c0] : memref<1xi32>

    // for (; a <= 5; ++a) { e = make_f(); }
    %final_e = scf.for %iv = %c0_idx to %c6_idx step %c1_idx
        iter_args(%e_iter = %d) -> (memref<1xi32>) {

      %f = func.call @make_f() : () -> memref<1xi32>

      func.call @b() : () -> ()

      // ++a
      %a_old = memref.load %a_ref[%c0] : memref<1xi32>
      %a_next = arith.addi %a_old, %c1_i32 : i32
      memref.store %a_next, %a_ref[%c0] : memref<1xi32>

      scf.yield %f : memref<1xi32>
    }

    // stack-use-after-return
    %ret = memref.load %final_e[%c0] : memref<1xi32>
    return %ret : i32
  }
}
