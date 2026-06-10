## evaluate_mlir.py 使用说明

### 最基本用法

```bash
python evaluate_mlir.py
```

默认行为：
- 读取 `test_cases_mlir/` 目录下的所有 `.mlir` 文件
- 并行数 `CPU数`
- 每个样例超时 30 秒

### 可用命令示例

```bash
# 详细输出（输出表格总结每个bug type的overall 结果）
python evaluate_mlir.py -v


# 8 线程并行，单样例超时 60 秒
python evaluate_mlir.py -j 8 --timeout 60

# 保留编译生成的中间文件（默认会清理 temp 目录）
python evaluate_mlir.py --keep

# 使用 asan0 配置，优化级别 O2
python evaluate_mlir.py --config asan0 --opt 2

# 只测 Linear OOBA 和 Type Confusion OOBA 两类 spatial bug
python evaluate_mlir.py --bug linear_ooba type_confusion_ooba

# 只测 Double-free 这一类 temporal bug
python evaluate_mlir.py --bug double_free
```

**`--bug` 说明**

- 可指定一个或多个 bug type，只运行匹配的测试样例。
- 支持的标准名及别名：
  - `Linear OOBA` / `linear_ooba` / `linearooba` / `linear`
  - `Non-Linear OOBA` / `non_linear_ooba` / `nonlinearooba` / `nonlinear`
  - `Type Confusion OOBA` / `type_confusion_ooba` / `typeconfusion` / `typeconfusionooba`
  - `Misuse-of-free` / `misuse_of_free` / `misuseoffree`
  - `Double-free` / `double_free` / `doublefree`
  - `Use-after-*` / `use_after_star` / `useafterstar`
- 如果指定了不存在的 bug type，脚本会报错并列出所有支持的类型。


## 输出格式

### 默认输出

```
Found 123 test cases, 456 variants.
Total work items: 456
Evaluated 123 test cases, 456 variants.
Overall results:
Detection rate: 85.37% (105 out of 123 test cases)
Results for test cases:
- Preconditions failed: 12.20% (15)
- Detected: 73.17% (90)
- Undetected: 14.63% (18)
```

### 详细输出 (`-v`)

按 temporal / spatial 两大类，分别输出 bug type、region、memory state、access location、access action 等维度的统计。
```
Temporal Bugs Distribution
Category                 | Detection Rate | Precond Failed |   Detected | Undetected
------------------------------------------------------------------------------------
Overall                  |    50.00% (20) |      0.00% (0) | 50.00% (20) | 50.00% (20)
Misuse-of-free           |   100.00% (20) |      0.00% (0) | 100.00% (20) |  0.00% (0)
Double-free              |      0.00% (0) |      0.00% (0) |  0.00% (0) | 100.00% (4)
....
```

---



## crisp_phase2.py 使用说明

`crisp_phase2.py` 是 MSET-MLIR 的编译运行脚本，负责将 **已 bufferized** 的 MLIR 文件（使用 `memref` 而非 `tensor`）经 CRISP/ASan pipeline 编译为可执行文件并运行。


### 基本用法

```bash
# 使用 crisp 配置（默认）
python crisp_phase2.py test_cases_mlir/linear_ooba_heap_heap_inter_object_overflow_direct_read_0.mlir

# 使用 asan0 配置
python crisp_phase2.py test.mlir --config asan0

# 使用 asan-outline 配置，优化级别 O2，保留中间文件
python crisp_phase2.py test.mlir --config asan-outline --opt 2 --keep

# 指定输出目录并保留中间文件
python crisp_phase2.py test.mlir ./my_output --config crisp --keep
```

### 完整编译流程

单个 `.mlir` 文件的编译执行流程如下：

```
input.mlir
    ↓  mlir-opt (pass pipeline)
_llvm.mlir  (LLVM dialect)
    ↓  mlir-translate
.ll         (LLVM IR)
    ↓  llc
.o          (object file)
    ↓  clang + link
executable  (可执行文件)
    ↓  run
exit code   (42=成功, 43=预条件失败, 0=未检测, 其他=检测到错误)
```

默认情况下，中间文件（`_llvm.mlir`, `.ll`, `.o`, `executable`）会在执行完毕后自动清理。使用 `--keep` 可保留。

---

### 命令行参数

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `input_mlir` | 位置参数 | 必填 | 输入的 bufferized MLIR 文件路径 |
| `output_dir` | 位置参数 | `./<stem>_output` | 输出目录，存放中间文件和可执行文件 |
| `--config` | 可选 | `crisp` | 编译配置：`crisp`、`asan0`、`asan-outline`、`asan-opt`、`base` |
| `--keep` | 开关 | `False` | 保留中间文件（LLVM dialect、LLVM IR、object、可执行文件） |
| `--opt` | 可选 | `0` | 优化级别（`0/1/2/3`），传给 llc / clang |
| `--debug-ir` | 开关 | 隐藏 | 打印 `--mlir-print-ir-after-all`（调试用，参数隐藏） |


## 二者关系

`evaluate_mlir.py` 本身不直接编译 MLIR，而是作为调度器调用 `crisp_phase2.py`：

```bash
python crisp_phase2.py <mlir_file> <output_dir> --config <config>
```

因此：
- `crisp_phase2.py` 必须能独立完成单个 `.mlir` → 可执行文件 → 运行的全流程
- `evaluate_mlir.py` 只负责批量调度、结果解析和统计

---
