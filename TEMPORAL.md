# Temporal Bug Types MLIR 化完整修改计划

> 原则：最小修改代码、不考虑 MLIR SSA 特性、以保留 C 语义为主、仅字符串替换。
> 本文档整合自原 `PROJECT_UNDERSTANDING.md` 与 `PHASE5_USEAFTERSTAR_PLAN.md`。

---
**所有涉及全局变量的 access 调用前，都必须手动插入 `memref.get_global` + `memref.load` 这两行，得到局部 SSA value 后再传入 `generate` / `generate_split_const_vars`**。


## 文件 1：`src/generator/primitives/bug_types/temporal/double_free.cpp`

### 原实现方法
直接嵌入 C 代码字符串：
- `#ifndef __GLIBC__` / `_exit(PRECONDITIONS_FAILED_VALUE)` —— glibc 预检查
- `char *pointer_to_double_free; char *pointer_to_use;` —— 指针声明
- `malloc(10)` / `malloc(8)` / `free(...)` —— 堆分配与释放
- `pointer_to_double_free[sizeof(void *)] = 0;` —— use-after-free 写操作
- `char *tmp, *tmp2, *tmp3;` —— 辅助指针（仅 variant_without_use_after_free）
- `_use(tmp2); _use(tmp3);` —— 防优化标记
- `pointer_to_use = (char *)malloc(8);` → access via `pointer_to_use` → `exit(TEST_CASE_SUCCESSFUL_VALUE)`

### 现修改计划
1. **移除** `#ifndef __GLIBC__` 判断（MLIR 不依赖 glibc）。
2. **保留** `_use(tmp2)` / `_use(tmp3)` / `_use(pointer_to_use)` 等调用，翻译为 `func.call @use(%ptr) : (memref<8xi8>) -> ()`。需要在 globals 区域补充定义：
   ```mlir
     func.func @use(%arg0: memref<8xi8>) -> memref<8xi8> { return %arg0 : memref<8xi8> }");
  
   ```
3. `tmp`/`tmp2`/`tmp3` 的 `malloc(8)` 保留，翻译为 `memref.alloc() : memref<8xi8>`（variant_without_use_after_free 中的干扰分配）。
4. `pointer_to_double_free = (char *)malloc(10);` → `memref.alloc() : memref<10xi8>`（核心对象）。
5. `pointer_to_use = (char *)malloc(8);` → `%pointer_to_use = memref.alloc() : memref<8xi8>`（新分配对象），随后 `access_location->generate(action, "pointer_to_use", 8)` 继续复用已有接口。
6. `free(pointer_to_double_free)` → `memref.dealloc %pointer_to_double_free : memref<10xi8>`。
7. `pointer_to_double_free[sizeof(void *)] = 0;` → `memref.store %c0_i8, %pointer_to_double_free[%c8] : memref<10xi8>`（variant_with_use_after_free 保留）。
8. `exit(TEST_CASE_SUCCESSFUL_VALUE)` → `func.return %test_success : i32`。

### 原因
- `HeapRegion` 已 MLIR 化，但 `DoubleFree` 额外硬编码的 `malloc/free` 字符串与其不兼容，必须逐条替换。
- `_use()` 在 C 中用于防止编译器优化掉未使用的变量；MLIR 前端虽无此问题，但保留 `_use` 调用可维持测试用例的原始语义结构，翻译为 `func.call @use` 即可。
- 核心语义必须保留：alloc → dealloc →（可选 dealloc 后 store）→ double-dealloc → **alloc 新对象 `pointer_to_use` → access `pointer_to_use`** → return success。

---

## 文件 2：`src/generator/primitives/bug_types/temporal/misuse_of_free.cpp`

### 原实现方法
直接嵌入 C 代码字符串：
- `target[8] = 0x40;` / `target[13*8] = 0x40;` —— 向 target 写 magic value 伪造 chunk
- `#ifndef __GLIBC__` / `_exit(PRECONDITIONS_FAILED_VALUE)` —— glibc 预检查
- `unsigned long *crafted_ptr; crafted_ptr = &(target[2*8]); free(crafted_ptr);` —— 取子区域地址并释放
- `char* heap_obj; heap_obj = (char *)malloc(8);` —— 重新分配堆对象
- `if ( &<target_name>[2*8] != heap_obj ) _exit(PRECONDITIONS_FAILED_VALUE);` —— 地址匹配检查（UsedMemory）
- `exit(TEST_CASE_SUCCESSFUL_VALUE)` —— 成功退出

### 现修改计划
1. **移除** `#ifndef __GLIBC__` 判断。
2. `target[8] = 0x40;` → `memref.store %c0x40_i8, %target[%c8] : memref<160xi8>`（magic value 保留为 `memref.store`，注释说明无 chunk 语义）。
3. `crafted_ptr = &(target[2*8]); free(crafted_ptr);` → `memref.subview` 取子区域 + `memref.dealloc`：
   ```mlir
   %crafted = memref.subview %target[16][8][1]
             : memref<160xi8> to memref<8xi8, strided<[1], offset: 16>>
   memref.dealloc %crafted : memref<8xi8, strided<[1], offset: 16>>
   ```
   （已验证 `memref.dealloc %subview` verifier 通过。）
4. `heap_obj = (char *)malloc(8);` → `%heap_obj = memref.alloc() : memref<8xi8>`。
5. UsedMemory 的地址匹配检查 `if (&target[2*8] != heap_obj)` → `memref.extract_aligned_pointer_as_index` 比较（若需要保留）
6. `exit(...)` → `func.return %test_success : i32` / `func.return %precond_fail : i32`。

### 原因
- 核心 bug 语义是"释放子区域后 access"，`memref.subview` + `memref.dealloc` 能保留这一语义。
- Magic values（`0x20`/`0x40`/`0x60`）在 MLIR 中无 chunk 语义，保留为 `memref.store` 仅维持代码结构。
- `accepts(region)` 原返回 `true`，但实际代码只处理 heap，MLIR 化时暂不修复，保持原接口兼容。

---

## 文件 3：`src/generator/primitives/bug_types/temporal/use_after_star.cpp`

> **方案C（选定）**：扩展 `AccessLocation` 接口，新增基于 `!llvm.ptr` 的 `generate_llvm_ptr()` 方法；`use_after_star` 去掉 `@target_ptr`，仅保留 `@target_address`（改为 `memref<1xi64>`），通过 `llvm.inttoptr` 重建指针，删除所有前置条件地址比较。

### 原实现方法
直接嵌入 C 代码字符串，包含多个子函数：
- `_generate_unused_mem_heap`：全局变量 `char *target_address;` → alloc → init → free → 在 `deallocation_pos` 保存 `&target[0]` → access via `target_address`
- `_generate_unused_mem_stack`：全局变量 `char *target_address;` → alloca in `f()` → `target_address = &target[0];` → access in `main()` via `target_address`
- `_generate_reused_mem_heap`：alloc → `target_address = &target[0]` → free → `realloc` 循环 → `GET_ADDR_BITS` 地址比较 → access
- `_generate_reused_mem_stack`：跨 `f()`/`other_f()`，含 simple/repeated/array/repeated+array 4 个变体，使用 `do { other_f(); } while(counter++ < N)` 重试循环
- `_exit(42)`/`_exit(43)`、`volatile`、`_use(ptr)`、全局数组 `char *target_addresses[16]`

### test.mlir 与现有实现的关键差异

| 维度 | test.mlir | 现有 use_after_star.cpp |
|:---|:---|:---|
| 全局变量 | 仅 `@target_address`（`memref<1xi64>`），无 `@target_ptr` | `@target_address`（`memref<1xindex>`）+ `@target_ptr`（`memref<1xmemref<8xi8>>`） |
| 保存内容 | 仅保存 raw address（`i64`） | 同时保存 `index` 地址值和 `memref<8xi8>` 对象本身 |
| 加载转换 | `memref.load` → `llvm.inttoptr` → `!llvm.ptr` | `memref.load` 直接得到 `memref<8xi8>` |
| 前置条件 | **已删除**：不比较地址，无论是否相等继续执行 | 有 `%eq = arith.cmpi eq` + `scf.if` 判断，不等则 `precond_fail` |
| 访问方式 | 通过 `!llvm.ptr` 调用函数（`_use`、`memset`） | 通过 `memref` 操作（`memref.load/store/copy`，由 `AccessLocation` 生成） |

### 方案C：AccessLocation 接口扩展

#### 新增接口

在 `access_location.h` 中添加纯虚方法：

```cpp
// Generate access code using !llvm.ptr instead of memref.
// The generated code operates on a variable of type !llvm.ptr.
// Callers must ensure required external functions (e.g., @memset, @memcpy)
// are declared in the global scope when using StdlibLocation.
virtual std::vector<std::string> generate_llvm_ptr(
    std::shared_ptr<AccessAction> action,
    const std::string &llvm_ptr_var_name,
    size_t size
) const = 0;
```

在 `direct_location.h/.cpp` 和 `stdlib_location.h/.cpp` 中分别实现 `override`。

#### DirectLocation::generate_llvm_ptr 实现

- **ReadAction**：逐字节通过 `llvm.getelementptr` + `llvm.load` + `func.call @use`
  ```mlir
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %gep = llvm.getelementptr %ptr[%i] : (!llvm.ptr, index) -> !llvm.ptr, i8
    %val = llvm.load %gep : !llvm.ptr -> i8
    func.call @use(%val) : (i8) -> ()
  }
  ```
- **WriteAction**：逐字节通过 `llvm.getelementptr` + `llvm.store`
  ```mlir
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0xFF = arith.constant 255 : i8
  scf.for %i = %c0 to %c8 step %c1 {
    %gep = llvm.getelementptr %ptr[%i] : (!llvm.ptr, index) -> !llvm.ptr, i8
    llvm.store %c0xFF, %gep : i8, !llvm.ptr
  }
  ```

> **注意**：DirectLocation 的 `!llvm.ptr` 版本不需要额外外部函数声明，仅使用 `llvm.*` 操作和默认的 `@use(i8)`。

#### StdlibLocation::generate_llvm_ptr 实现

- **ReadAction**：`func.call @memcpy` 到临时 buffer，再读取调用 `@use`
  ```mlir
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c8_i64 = arith.constant 8 : i64
  %tmp_buf = memref.alloca() : memref<8xi8>
  %tmp_idx = memref.extract_aligned_pointer_as_index %tmp_buf : memref<8xi8> -> index
  %tmp_i64 = arith.index_cast %tmp_idx : index to i64
  %tmp_ptr = llvm.inttoptr %tmp_i64 : i64 to !llvm.ptr
  func.call @memcpy(%tmp_ptr, %ptr, %c8_i64) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
  %use_val = memref.load %tmp_buf[%c0] : memref<8xi8>
  func.call @use(%use_val) : (i8) -> ()
  ```
- **WriteAction**：`func.call @memset`
  ```mlir
  %c0xFF_i32 = arith.constant 255 : i32
  %c8_i64 = arith.constant 8 : i64
  func.call @memset(%ptr, %c0xFF_i32, %c8_i64) : (!llvm.ptr, i32, i64) -> !llvm.ptr
  ```

> **外部函数声明要求**：使用 StdlibLocation 的 `generate_llvm_ptr` 时，调用方需在 globals 区域添加：
> ```mlir
> func.func private @memset(!llvm.ptr, i32, i64) -> !llvm.ptr
> func.func private @memcpy(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
> ```

### use_after_star.cpp 修改计划

#### 全局变量替换（单全局 + i64 方案）

**去掉所有 `@target_ptr`**（`memref<1xmemref<8xi8>>`），涉及约 14 处 `code.add_global`。

**`@target_address` 类型修改**：
- 从 `memref<1xindex>` 改为 `memref<1xi64>`
- 涉及约 7 处 `code.add_global`

**数组变体**：`@target_addresses` 从 `memref<16xindex>` 改为 `memref<16xi64>`；`@target_arr`（`memref<1xmemref<16x8xi8>>`）可删除（不再需要保存 memref 对象本身）。

#### 指针保存逻辑修改

**所有 during-lifetime / f-body / deallocation_pos 处的保存代码**，从：
```mlir
%target_addr = memref.extract_aligned_pointer_as_index %target : memref<8xi8> -> index
%global_addr = memref.get_global @target_address : memref<1xindex>
memref.store %target_addr, %global_addr[%c0] : memref<1xindex>
%global_ptr = memref.get_global @target_ptr : memref<1xmemref<8xi8>>
memref.store %target, %global_ptr[%c0] : memref<1xmemref<8xi8>>
```
**改为**：
```mlir
%target_ptr_idx = memref.extract_aligned_pointer_as_index %target : memref<8xi8> -> index
%target_ptr_i64 = arith.index_cast %target_ptr_idx : index to i64
%global_addr = memref.get_global @target_address : memref<1xi64>
memref.store %target_ptr_i64, %global_addr[%c0] : memref<1xi64>
```

#### 指针恢复/加载逻辑修改

**所有 access 前的加载代码**，从：
```mlir
%global_ptr_main = memref.get_global @target_ptr : memref<1xmemref<8xi8>>
%saved_ptr = memref.load %global_ptr_main[%c0] : memref<1xmemref<8xi8>>
```
**改为**：
```mlir
%global_addr_main = memref.get_global @target_address : memref<1xi64>
%stored_ptr_i64 = memref.load %global_addr_main[%c0] : memref<1xi64>
%stored_ptr = llvm.inttoptr %stored_ptr_i64 : i64 to !llvm.ptr
```

#### 访问方式切换

所有 `access_location->generate(action, "saved_ptr", 8)` 和 `access_location->generate_split_const_vars(action, "saved_ptr", 8)` **替换为**：
```cpp
access_location->generate_llvm_ptr(action, "stored_ptr", 8)
```

返回值是 `std::vector<std::string>`，可直接通过 `add_during_lifetime()` / `add_to_f_body()` / `add_at()` 追加。

#### 各函数修改要点

**`_generate_unused_mem_heap`**：
1. 全局声明：`memref.global @target_address : memref<1xi64>` + `@exit`
2. `deallocation_pos` 后插入：extract → index_cast → store i64
3. 加载：load i64 → `llvm.inttoptr` → `generate_llvm_ptr`
4. 最后 `func.call @exit(%test_success)`

**`_generate_unused_mem_stack`**：
- 同理，保留 `f()` / `main()` 跨函数结构，但只保存/加载 `@target_address`（i64）

**`_generate_reused_mem_heap`（Simple 变体）**：
1. 去掉 `@target_ptr`
2. dealloc 前保存 target 地址（i64）
3. 单次 alloc `%reallocated_out`
4. **删除** `arith.cmpi eq` + `scf.if`
5. 直接加载 `@target_address` → `inttoptr` → `generate_llvm_ptr` → `func.call @exit(%test_success)`
6. 最后 `memref.dealloc %reallocated_out`

**`_generate_reused_mem_heap`（Repeated 变体）**：
- **删除地址比较逻辑**：`scf.while` 不再以"地址匹配"为目的

**`_generate_reused_mem_stack`（Simple 变体）**：
1. `f()` 中：保存 target 地址（i64）到 `@target_address`
2. `other_f()` 中：alloca reallocated → **删除**地址比较
3. 直接加载 `@target_address` → `inttoptr` → `generate_llvm_ptr` → `func.call @exit(%test_success)`

**`_generate_reused_mem_stack`（Repeated 变体）**：
- 同 heap repeated，删除地址匹配逻辑
- `last_address` 检查也可删除（其目的是避免连续相同地址）

**`_generate_reused_mem_stack`（Array 变体）**：
- `@target_arr` 可删除，仅保留 `@target_addresses`（改为 `memref<16xi64>`）
- `f()` 中：循环保存16个 `i64` 地址（`arith.index_cast` 转换后 store）
- `other_f()` 中：
  - 加载 `@target_addresses[0]` → `inttoptr` → `generate_llvm_ptr`

#### Validation 版本同步

所有 `*_validation()` 方法应用同样的修改：
- 去掉 `@target_ptr`
- `@target_address` 改为 `memref<1xi64>`
- 保存/加载逻辑使用 `index_cast` / `inttoptr`
- 删除所有前置条件地址比较
- access 调用 `generate_llvm_ptr`

#### 外部函数声明

在 `use_after_star.cpp` 中，当 `access_location` 为 `StdlibLocation` 时，需在 `CodeCanvas` 的 globals 中添加：
```cpp
code.add_global("func.func private @memset(!llvm.ptr, i32, i64) -> !llvm.ptr");
code.add_global("func.func private @memcpy(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr");
```

若直接通过 `is_a<StdlibLocation>(access_location)` 判断不方便（因为 `access_location` 是基类指针），可改为**无条件声明**，或仅在 generate 方法开头统一添加（重复声明无害，MLIR 允许重复的 `func.func private` 声明）。

### 原因

- **去掉 `@target_ptr` 是 test.mlir 的核心简化**：不再保存 memref SSA value 本身，仅保存 raw address（`i64`），通过 `llvm.inttoptr` 在需要时恢复 `!llvm.ptr`。
- **`!llvm.ptr` 访问更贴近底层 C 语义**：C 代码中 `free(target); memset(target, 0xFF, 8);` 直接操作的是裸指针，而非 memref。`generate_llvm_ptr` 让 MLIR 生成代码更接近原始 C 语义，有利于检测器（如 ASan）识别。
- **删除前置条件**：test.mlir 明确删除地址比较。原 C 代码中 `if (target_address != reallocated) _exit(PRECONDITIONS_FAILED_VALUE);` 是测试框架的"前置条件保护"，在 MLIR 化后变为 bug 检测的干扰项。删除后，无论 realloc 是否返回相同地址，都执行 access，更符合"测试用例应尽可能触发 bug"的设计目标。
- **扩展 AccessLocation（方案C）而非硬编码（方案B）**：保持 MSET 的通用架构。`DirectLocation` 和 `StdlibLocation` 的 `generate_llvm_ptr` 实现可被其他 bug type 复用（如 `misuse_of_free` 已使用 `llvm.inttoptr`）。

---

## 修改文件清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `src/generator/primitives/bug_types/temporal/double_free.cpp` | 字符串替换 | C malloc/free → MLIR alloc/dealloc；`_use` 改为 `func.call @use`；保留 tmp 辅助分配 |
| `src/generator/primitives/bug_types/temporal/misuse_of_free.cpp` | 字符串替换 | 引入 memref.subview + memref.dealloc；magic value 改为 memref.store |
| `src/generator/primitives/bug_types/temporal/use_after_star.cpp` | 字符串替换 | 双全局变量、extract_aligned_pointer_as_index、`scf.while` + iter_args 保留循环内 realloc |
| `src/generator/primitives/access_types/access_location.h` | 接口修改 | `generate` / `generate_split_const_vars` / `generate_split_aux_vars` 新增 `array_size` 和 `index_var` 默认参数 |
| `src/generator/primitives/access_types/direct_location.h/.cpp` | 实现修改 | 实现新增参数，支持 2D memref 数组元素访问 |
| `src/generator/primitives/access_types/stdlib_location.h/.cpp` | 实现修改 | 实现新增参数，支持 2D memref 数组元素访问 |
| `src/generator/primitives/regions/heap_region.h` | 接口修改 | `generate_deallocation` 增加 `size_t size` 参数 |
| `src/generator/primitives/regions/heap_region.cpp` | 实现修改 | `generate_deallocation` 使用传入的 `size` 生成正确类型 |

---

## 验证步骤

1. `cd build && make`
2. `./mset --output-dir test_cases_mlir/`
3. 检查生成文件数量应与 C 版本一致
4. `mlir-opt --verify-diagnostics test_cases_mlir/xxx.mlir`
