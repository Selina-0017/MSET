# Polygeist 操作/属性汇总（cu_out MLIR 样例）

本文档整理自 `cu_out/` 目录下 11 个由 cgeist 生成的 MLIR 样例中出现的全部 `polygeist` 相关操作与属性，按出现频次降序排列，并附典型例句。

---

## 1. `polygeist.memref2pointer`（610 次）

**说明**：将 `memref` 转换为 LLVM 指针类型。

**例句**：
```mlir
%47 = "polygeist.memref2pointer"(%44) : (memref<?xi32>) -> !llvm.ptr<i32>
```

---

## 2. `polygeist.undef`（485 次）

**说明**：生成一个未定义值（undef），常用于占位或初始化。

**例句**：
```mlir
%0 = "polygeist.undef"() : () -> i32
```

---

## 3. `polygeist.subindex`（343 次）

**说明**：对 `memref` 进行子索引/偏移，得到新的 `memref` 视图。

**例句**：
```mlir
%11 = "polygeist.subindex"(%10, %c0) : (memref<?xi32>, index) -> memref<?xi32>
```

---

## 4. `polygeist.pointer2memref`（250 次）

**说明**：将 LLVM 指针类型转换回 `memref`。

**例句**：
```mlir
%46 = "polygeist.pointer2memref"(%45) : (!llvm.ptr<i8>) -> memref<?xi32>
```

---

## 5. `polygeist.typeSize`（113 次）

**说明**：获取某类型的字节大小，返回 `index` 类型。

**例句**：
```mlir
%44 = "polygeist.typeSize"() <{source = i32}> : () -> index
```

---

## 6. `polygeist.device_only_func`（46 次）

**说明**：函数属性，标记该函数为仅设备端（device-only）的 CUDA kernel。

**例句**：
```mlir
func.func private @_Z32__device_stub__allocDecodeKernelPKiPii(...)
  attributes {..., polygeist.device_only_func = "1"} {
```

---

## 7. `polygeist.target`（22 次，模块属性）

**说明**：模块级属性，描述目标 CPU 及特性（`target-cpu`、`target-features`）。

**例句**：
```mlir
module attributes {
  ...,
  "polygeist.target-cpu" = "x86-64",
  "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87",
  ...
} {
```

---

## 8. `polygeist.gpu_module`（22 次，模块属性）

**说明**：模块级属性，记录 GPU 模块对应的数据布局与目标三元组。

**例句**：
```mlir
module attributes {
  ...,
  polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",
  polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda",
  ...
} {
```

---

## 9. `polygeist.tune`（11 次，模块属性）

**说明**：模块级属性，记录调优用的 CPU 名称（`tune-cpu`）。

**例句**：
```mlir
module attributes {
  ...,
  "polygeist.tune-cpu" = "generic",
  ...
} {
```

---

## 10. `polygeist.cuda_constant`（1 次）

**说明**：全局变量属性，标记该值为 CUDA 常量（如 `__constant__` 或编译期常量）。

**例句**：
```mlir
memref.global "private" @_ZL15SHARED_ELEMENTS : memref<1xi32> = dense<64> {polygeist.cuda_constant}
```