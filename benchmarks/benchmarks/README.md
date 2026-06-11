# Benchmarks

`benchmarks/synthetic/` 当前保留 11 组核心 probe family。每组都包含一个 `safe_*` 控制组、一个错误样例，以及能被 `manifest.json` 和 `scripts/run_probes.py` 统一调度的命令行入口。

## Case Inventory

| 文件 | 控制组 | 错误样例 | 触发方式 | native | `compute-sanitizer` | 来源 |
|---|---|---|---|---|---|---|
| `shared_memory.cu` | `shared_safe_copy` | `shared_negative_index_read` | thread 0 读取 `scratch[-1]` | 正常退出，输出 hash 与控制组不同 | 常见为 `missed` | synthetic |
| `allocator_order.cu` | `safe_alloc_decode` | `late_bounds_check_oob` | kernel 先读 `freePages[requestStart]`，host 再做容量检查 | 正常退出 | `detected` | SGLang `#22035` |
| `chunk_cat_pattern.cu` | `safe_chunk_cat_read` | `logical_chunk_cat_oob_read` | 同一 allocation 内，logical payload 末尾之后继续读 8B | 正常退出，输出 hash 与控制组不同 | 常见为 `missed` | PyTorch `#122026` |
| `llama_kv_pattern.cu` | `safe_llama_f16_read` | `llama_f16_padding_oob_read` | F16 buffer 做 one-past-end 边界读 | 正常退出 | `detected` | llama.cpp `#8798` |
| `scaled_mm_rowwise_pattern.cu` | `safe_rowwise_scale_read` | `rowwise_scale_tail_oob_read` | row-wise scale metadata 按 `256` 行 tile 对齐后继续读 padding 尾部 | 正常退出 | `detected` | PyTorch `#133334` |
| `bgmv_tile_pattern.cu` | `safe_bgmv_tile_read` | `bgmv_tile_overflow_read` | tile/vectorized 输入读取在 `feat_in = 512` 时越过 `X` 的最后一行尾部 | native 在同步点报错 | `detected` | vLLM `#6902` / `#4756` / PR `#5169` |
| `llama_copy_override_pattern.cu` | `safe_llama_override_copy` | `llama_override_one_past_end_read` | F16 copy/override 路径从 32B 缓冲末尾之后继续做 8B 读取 | 正常退出 | `detected` | llama.cpp `#12798` |
| `mla_concat_threshold_pattern.cu` | `safe_mla_concat_read` | `mla_concat_threshold_oob_read` | 长输入从 `29120` 跨到 `29130` 后，rope-side concat 读越过尾部 | 正常退出 | `detected` | SGLang `#12250` |
| `multimargin_target_pattern.cu` | `safe_multimargin_target_read` | `negative_multimargin_target_read` | 非法负 target 让 kernel 读取 `rowBase[-1]` | 正常退出 | `detected` | PyTorch `#88724` / PR `#89008` |
| `qo_indptr_overflow_pattern.cu` | `safe_qo_indptr_read` | `qo_indptr_int32_overflow_read` | `qo_indptr * stride_qbs` 截断到 `int32` 后变成负偏移，再读到可见子视图之前 | 正常退出 | `detected` | SGLang `#20389` |
| `varlen_metadata_pattern.cu` | `safe_varlen_metadata_read` | `varlen_cu_seqlens_mismatch_read` | `cu_seqlens` 末值声称有 `768` 个 token，但实际 `q` 只有 `256` 个 token | 同步点报错 | `detected` | FlashAttention `#2381` |

## Family Notes

### `shared_memory.cu`

- `shared_safe_copy`：把前 64 个输入元素搬进 shared scratchpad，再原样写回，作为 shared-memory 控制组。
- `shared_negative_index_read`：只有 `threadIdx.x == 0` 访问 `scratch[-1]`，其余线程走正常路径；这个 case 的价值不是“真实来源”，而是稳定制造一个 shared-memory 静默污染候选。
- 来源：仓库内 synthetic probe，没有外部 issue；文档中必须明确它不是从真实框架 bug 直接收缩得到的样例。

### `allocator_order.cu`

- `safe_alloc_decode`：先做 early reject，不启动越界 kernel，用来模拟修复后的安全路径。
- `late_bounds_check_oob`：保留“先 launch kernel，再判断 free pages 是否够用”的错误顺序，最小化复刻 allocator bounds-check-order 失效。
- 来源：SGLang paged decode allocator 问题，见 `sgl-project/sglang#22035`。该 PR 明确说明 root cause 是 kernel 已经读取 `free_pages`，host 侧 OOM guard 才执行。

### `chunk_cat_pattern.cu`

- `safe_chunk_cat_read`：在 80B allocation 内，从 logical payload 内部偏移 `8` 读取 8B。
- `logical_chunk_cat_oob_read`：仍留在同一 allocation 内，但从 logical payload 边界偏移 `16` 读取 8B；这类错误更像框架内部 offset / index 计算失误，而不是经典 allocation 外越界。
- 来源：PyTorch `_chunk_cat` CUDA fast path 问题，见 `pytorch/pytorch#122026`；`compute-sanitizer` 报告显示地址“超出逻辑对象边界，但仍位于最近 allocation 内”。

### `llama_kv_pattern.cu`

- `safe_llama_f16_read`：读取 `KV_HALF_ELEMENTS - 1` 位置的最后一个合法 `__half`。
- `llama_f16_padding_oob_read`：读取 `KV_HALF_ELEMENTS`，即 one-past-end 的 `__half` 边界。
- 来源：llama.cpp / GPT4All 路径中的 F16 DMMV / KV padding 问题，见 `ggml-org/llama.cpp#8798`；`compute-sanitizer` 首条错误是 size 2 的 invalid global read。

### `scaled_mm_rowwise_pattern.cu`

- `safe_rowwise_scale_read`：`M = 512`，row-wise scale metadata 与 `256` 行 tile 对齐，kernel 只读取合法 `scale[row]`。
- `rowwise_scale_tail_oob_read`：`M = 257`，kernel 仍按 `align_up(M, 256)` 推进第二个 tile，从而继续读取 `scale` 尾部之后的 padding 区域。
- 来源：PyTorch `torch._scaled_mm` row-wise FP8 路径问题，见 `pytorch/pytorch#133334`；原始 issue 的关键点是 `scale_a` metadata buffer 会被 tile padding 逻辑读穿。

### `bgmv_tile_pattern.cu`

- `safe_bgmv_tile_read`：保留 `seq_len = 32768` 和 tile/vectorized 读取骨架，但把 `feat_in` 设成与 tile 完整对齐的 `640`，让 control 组稳定 `clean`。
- `bgmv_tile_overflow_read`：把 `feat_in` 改成 `512`，保持 `vec_size = 4`、`160` 线程的 tile 步幅，让最后一行尾部被继续读取。
- 来源：vLLM Punica / `dispatch_bgmv` 路径问题，见 `vllm-project/vllm#6902`、更早的 `#4756` 和修复 PR `#5169`。

### `llama_copy_override_pattern.cu`

- `safe_llama_override_copy`：从 32B F16 缓冲中最后一个合法 8B 段读取 packed 值。
- `llama_override_one_past_end_read`：从 half offset `16` 开始直接做 8B 读取，对应 “最近 allocation 之后 1 字节” 的 one-past-end 模式。
- 来源：llama.cpp tensor-buffer override 路径问题，见 `ggml-org/llama.cpp#12798`；原始 `compute-sanitizer` 报告也是 `size 8`、最近 allocation 为 `32` 字节。

### `mla_concat_threshold_pattern.cu`

- `safe_mla_concat_read`：保留 `D_NOPE = 512`、`D_ROPE = 64`、`H = 128` 的长输入 concat 骨架，`S = 29120` 时只覆盖合法行。
- `mla_concat_threshold_oob_read`：把 `S` 提到 `29130`，再按 `32` 行 tile 向上对齐到 `29152`，模拟阈值跨过后继续读取 rope-side 尾部。
- 来源：SGLang `concat_mla_absorb_q` 长输入失败问题，见 `sgl-project/sglang#12250` 和修复 PR `#12453`。

### `multimargin_target_pattern.cu`

- `safe_multimargin_target_read`：读取合法 target `0/1/2` 对应的类分数。
- `negative_multimargin_target_read`：把 target 设成 `-1`，直接复现 `rowBase[target]` 这种 pre-allocation 读。
- 来源：PyTorch `MultiMarginLoss` 问题，见 `pytorch/pytorch#88724`；维护者把根因定位到 `input_k[target_k]`，随后在 PR `#89008` 中补了 range check。

### `qo_indptr_overflow_pattern.cu`

- `safe_qo_indptr_read`：保留 `stride_qbs = 8192` 的 Qwen3-Next 风格布局，但只用小 `qo_indptr`，避免乘法溢出。
- `qo_indptr_int32_overflow_read`：把 `qo_indptr` 设到高位，先按 `int64` 形成完整乘积，再手动截断成 `int32` wrapped offset；probe 用一个 shifted visible view 把负 wrapped offset 压成稳定的 pre-allocation 读取。
- 来源：SGLang `extend_attention` / `qo_indptr` 问题，见 `sgl-project/sglang#20389`。

### `varlen_metadata_pattern.cu`

- `safe_varlen_metadata_read`：`cu_seqlens = [0, 256]`，metadata 与实际 `q` token 数匹配。
- `varlen_cu_seqlens_mismatch_read`：`cu_seqlens = [0, 256, 512, 768]`，最后一个 sequence 会把 `last_token` 算到真实 buffer 之外。
- 来源：FlashAttention `flash_attention_2` + Qwen3.5 的 packed-sequence 误判问题，见 `Dao-AILab/flash-attention#2381`。

## Source Links

- SGLang allocator bug: <https://github.com/sgl-project/sglang/pull/22035>
- PyTorch `_chunk_cat` bug: <https://github.com/pytorch/pytorch/issues/122026>
- llama.cpp F16 / KV padding bug: <https://github.com/ggml-org/llama.cpp/issues/8798>
- PyTorch `scaled_mm` row-wise scale bug: <https://github.com/pytorch/pytorch/issues/133334>
- vLLM Punica / BGMV bug: <https://github.com/vllm-project/vllm/issues/6902>
- vLLM related issue: <https://github.com/vllm-project/vllm/issues/4756>
- vLLM fix PR: <https://github.com/vllm-project/vllm/pull/5169>
- llama.cpp override copy bug: <https://github.com/ggml-org/llama.cpp/issues/12798>
- SGLang MLA concat threshold bug: <https://github.com/sgl-project/sglang/issues/12250>
- SGLang MLA concat fix PR: <https://github.com/sgl-project/sglang/pull/12453>
- PyTorch `MultiMarginLoss` bug: <https://github.com/pytorch/pytorch/issues/88724>
- PyTorch fix PR: <https://github.com/pytorch/pytorch/pull/89008>
- SGLang `qo_indptr` overflow bug: <https://github.com/sgl-project/sglang/issues/20389>
- FlashAttention varlen metadata mismatch bug: <https://github.com/Dao-AILab/flash-attention/issues/2381>

## Build

```bash
cmake -S benchmarks/synthetic -B build/synthetic
cmake --build build/synthetic
```

## Run

```bash
./build/synthetic/shared_memory shared_negative_index_read
./build/synthetic/allocator_order late_bounds_check_oob
./build/synthetic/chunk_cat_pattern logical_chunk_cat_oob_read
./build/synthetic/llama_kv_pattern llama_f16_padding_oob_read
./build/synthetic/scaled_mm_rowwise_pattern rowwise_scale_tail_oob_read
./build/synthetic/bgmv_tile_pattern bgmv_tile_overflow_read
./build/synthetic/llama_copy_override_pattern llama_override_one_past_end_read
./build/synthetic/mla_concat_threshold_pattern mla_concat_threshold_oob_read
./build/synthetic/multimargin_target_pattern negative_multimargin_target_read
./build/synthetic/qo_indptr_overflow_pattern qo_indptr_int32_overflow_read
./build/synthetic/varlen_metadata_pattern varlen_cu_seqlens_mismatch_read
```

## Run Through The Probe Runner

```bash
python scripts/run_probes.py --case late_bounds_check_oob --modes native,compute-sanitizer
```
