# xLLM DCU 接入 flash-mla 实施计划

> 状态：**Phase 1 完成 —— BF16 MLA 已接入并通过正确性验证**
> 编写日期：2026-06-14（Phase 1 实施完成 2026-06-14）
> 关联参考：`/wksp/d0612/dcu_flash_mla_adaptation_note.md`（另一份代码库上已验证的复现记录）
> 适用范围：gfx938 DCU，8 卡；DeepSeek-V2-Lite（BF16）、V3-0324（W8A8 FP8）、R1（INT8 / FP8 W8A8）

---

## 1. 任务背景与目标

在 xLLM 的 DCU 设备后端中接入 flash-mla，完成 DeepSeek-V2 / V3 / R1 系列模型的支持并跑通冒烟测试。

- **Phase 1（本阶段核心）**：接入 flash-mla，先支持 **BF16**，跑通 **DeepSeek-V2-Lite** 单卡冒烟测试（精度与语义正确）。
- **Phase 2**：在 BF16 通路稳定后，参考 vLLM-HCU / SGLang 的 DCU 实现，支持 **W8A8（FP8 / INT8 权重量化）**，跑通 V3-0324-Channel-FP8-w8a8 与 R1。
- **Phase 3**：所有冒烟测试通过后，回填本文档为最终设计（实际接入的 API、功能、接入方式、与 vllm/sglang 差异、优化方向）。

### 1.1 硬约束（来自任务要求）

1. 高性能算子优先用 **aiter**，但接入方式必须用 **`.so` 薄 adapter**（不在 C++ 运行时 import Python）。
2. 保持 xLLM 对多设备的**统一抽象**：设备专有逻辑收敛在 `layers/dcu`、`kernels/dcu`，公共主干改动最小且有 device guard。
3. **不改动 DCU `fused_moe`**（当前 torch-based 方案保持原设计），除非为让模型测试通过必须修改，并需说明理由。
4. FP8 / INT8 方案尽量与 vllm-hcu / sglang DCU 的 API 一致，**对 xllm 改动最小**。

---

## 2. 现状分析（gap 清单）

调研 `/wksp/d0612/xllm` 当前主干，确认以下现状：

### 2.1 已经具备 / 可直接复用（不要动）

| 能力 | 位置 | 说明 |
| --- | --- | --- |
| 通用 DCU attention 走 FlashAttentionImpl | `xllm/core/layers/dcu/attention.cpp:36` | 已是 `FlashAttentionImpl`，无需改回 torch fallback |
| DCU MoE（torch-based） | `xllm/core/layers/dcu/fused_moe.cpp` | 保持原设计（要求 2） |
| `moe_active_topk` 支持 softmax/sigmoid | `xllm/core/kernels/param.h:494` | V2 用 softmax，V3 用 sigmoid，均支持 |
| `reshape_paged_cache` 通用 latent 写入 | `xllm/core/kernels/ops_api.h:31` | num_heads=1 + head_dim=kv_lora_rank 时即可存 MLA latent k |
| `enable_mla` 自动设置 | `xllm/core/util/utils.h:202`、`kMlaModelTypeSet` 含 deepseek_v2/v3/v32 | 由 model_type 白名单推导，非 config 键 |
| `deepseek_v3` 模型 args 完整 | `xllm/models/llm/deepseek_v3.h` | 已含 scoring_func=sigmoid、topk_method=noaux_tc、hidden_act=silu |
| 构建系统已配置 | `build/cmake.linux-x86_64-cpython-310`（`USE_DCU=ON`） | 二进制已能构建 |
| flash-mla `.so` 与符号 | `/usr/local/lib/python3.10/dist-packages/flash_mla/cuda.cpython-310-x86_64-linux-gnu.so` | 见 §3.1 |
| aiter 算子库已装 | `/usr/local/lib/python3.10/dist-packages/aiter` | Phase2 W8A8 用 |

### 2.2 缺口（本阶段需要补）

| # | 缺口 | 影响 |
| --- | --- | --- |
| G1 | `models.h` DCU 分支未注册 deepseek | 模型无法加载 |
| G2 | `deepseek_v2.h` 硬编码 `#include "core/layers/mlu/deepseek_v2_decoder_layer_impl.h"`；V2 args 缺 `hidden_act`/`scoring_func` | DCU 编译失败 / MoE scoring 空串报错 |
| G3 | 无 DCU flash-mla adapter | 无法调用 `.so` |
| G4 | 无 DCU `DeepseekV2Attention` / `DeepseekV2DecoderLayer` | 无 MLA 前向 |
| G5 | `kv_cache_shape.cpp::apply_device_layout` 仅 MLU 做 MLA latent 合并（key-only）；DCU 缺失 | k_cache shape 不符合 flash-mla 输入 |
| G6 | DCU kernels/layers CMake 未链接 flash_mla `.so`、未加新源文件 | 链接失败 |

---

## 3. flash-mla `.so` 调研（已用 `nm -D` 实测）

库：`/usr/local/lib/python3.10/dist-packages/flash_mla/cuda.cpython-310-x86_64-linux-gnu.so`（flash-mla 1.2.0）

**本阶段（BF16 split-q decode）只用两个符号：**

```cpp
// 生成 decode 所需的 tile_scheduler_metadata / num_splits
std::vector<at::Tensor> get_mla_decoding_metadata_dense_fp8(
    at::Tensor& seqlens_k, int num_heads_per_head_k, int num_heads_k);

// BF16/FP16 split-q decode
std::vector<at::Tensor> mha_fwd_kvcache_mla_nope_pe(
    at::Tensor& q_nope,          // [B, S_q, H_q, kv_lora_rank]
    at::Tensor& q_pe,            // [B, S_q, H_q, qk_rope_head_dim]
    const at::Tensor& kcache,    // [blocks, 64, 1, kv_lora_rank + qk_rope_head_dim]
    const std::optional<const at::Tensor>& vcache,  // nullopt
    int head_size_v,             // kv_lora_rank
    const at::Tensor& seqlens_k, // [B]
    const at::Tensor& block_table,
    float softmax_scale,
    bool is_causal,
    const at::Tensor& tile_scheduler_metadata,
    const at::Tensor& num_splits);
```

**Phase2 备用符号（FP8 / quantization KV cache，调研已确认存在）：**
`mha_fwd_kvcache_mla_fp8`、`mha_fwd_kvcache_mla_fp8_with_cat`、`mha_fwd_kvcache_quantization_mla`、`mha_fwd_kvcache_quantization_q_nope_pe_mla`（见 §8）。

**Shape 约束**：`page_size/block_size = 64`；`k_cache` 单 KV head；`k_cache.size(-1) == q_nope.size(-1) + q_pe.size(-1)`。

---

## 4. 整体方案设计

### 4.1 分层与边界（遵循 xllm-device-backend-extension 准则）

```
models/llm/deepseek_v2.h          ← 设备条件 include（公共，最小改动 + device guard）
        │
        ▼
core/layers/dcu/
   deepseek_v2_decoder_layer_impl.{h,cpp}   ← DCU 设备层（结构镜像 qwen3_moe）
   deepseek_v2_attention.{h,cpp}            ← DCU MLA attention（镜像 MLU 简化版）
        │  decode: 调 flash-mla adapter
        │  prefill: torch sdpa fallback + reshape_paged_cache 存 latent
        ▼
core/kernels/dcu/
   flash_mla_adapter.{h,cpp}     ← 薄 adapter，唯一处声明 flash-mla .so ABI
```

- flash-mla ABI 细节**只**出现在 `flash_mla_adapter.*`，不泄漏到 attention / model。
- 公共文件改动仅 `models.h`、`deepseek_v2.h`、`kv_cache_shape.cpp`，全部用 `#if defined(USE_DCU)` 保护。
- 复用 common 模块：`ReplicatedLinear`/`ColumnParallelLinear`/`RowParallelLinear`、`RMSNorm`、`DenseMLP`、`FusedMoE`（DCU 版）、`create_mla_rotary_embedding`。

### 4.2 MLA 前向数据流（DCU 简化版）

**Decode（单 token，命中 flash-mla）：**
```
hidden_states
 → q_a_proj/q_a_layernorm/q_b_proj  (q_lora_rank>0) 或 q_proj  (V2-Lite q_lora_rank=0)
 → q = [tokens, H_q, qk_nope + qk_rope]
 → split: q_nope[H_q, qk_nope], q_pe[H_q, qk_rope]
 → q_nope_proj = bmm(q_nope, w_kc_)         # 投影到 kv_lora_rank
 → rotary(q_pe)                              # q_pe 已旋转
 → flash_mla::dense_decode_q_nope_pe(q_nope_proj, q_pe, k_cache, ...)
     → out = [B, S_q=1, H_q, kv_lora_rank]
 → bmm(out, w_vc_)  → [tokens, H_q, v_head_dim]
 → o_proj
```

**Prefill（prompt，torch sdpa fallback）：**
```
hidden_states
 → 同上得 q_nope_proj[H, kv_lora], q_pe（已旋转）→ q_input = cat(q_nope_proj, q_pe)
 → kv_a_proj_with_mqa → latent_cache = [tokens, kv_lora + qk_rope]
 → kv_a_layernorm(latent_cache[:, :kv_lora]); rotary(latent_cache[:, kv_lora:] as k_pe)
 → reshape_paged_cache(latent_cache → paged k_cache, slot_mapping)   # 存 latent 供后续 decode
 → sdpa(q_input[H, kv_lora+rope], latent_k/v[1, kv_lora+rope] MQA 广播)
 → bmm(attn_out, w_vc_) → o_proj
```

> 说明：V3 已在构造时 CHECK 关闭 prefix_cache / chunked_prefill；V2-Lite 冒烟也关闭（`--enable_prefix_cache=false --enable_chunked_prefill=false`），故 prefill 无需读取历史 paged cache。

### 4.3 KV cache shape（DCU MLA latent key-only）

`kv_cache_shape.cpp::apply_device_layout` 增加 `USE_DCU` 分支（与 MLU 对称）：
```cpp
#if defined(USE_MLU)
  ...（既有）
#elif defined(USE_DCU)
  if (model_args.enable_mla()) {
    CHECK(key_cache_shape_.has_value());
    CHECK_GE(key_cache_shape_->size(), 4);
    (*key_cache_shape_)[3] = model_args.kv_lora_rank() + model_args.qk_rope_head_dim();
    value_cache_shape_.reset();   // key-only latent cache
  }
#endif
```
结果：`k_cache = [num_blocks, block_size, 1, kv_lora_rank + qk_rope_head_dim]`，`v_cache` 为空。V2-Lite + `block_size=64` → `[blocks, 64, 1, 576]`。

> 注意：DCU 不做 MLU/ILU 的 dim1/dim2 转置，保持 `[blocks, block_size, 1, D]`，正是 flash-mla 期望的 layout。

---

## 5. 详细实现步骤（按依赖顺序）

### Step 1 — flash-mla adapter（G3）

**新增** `xllm/core/kernels/dcu/flash_mla_adapter.h`：
- `enum class DenseDecodeKind { kQNopePe };`（Phase2 扩 `kQNopePeFp8WithCat` / `kQNopePeQuantized`）
- `struct DenseDecodeParams { torch::Tensor q_nope, q_pe, k_cache, seqlens_k, block_table; int64_t head_size_v; float softmax_scale; DenseDecodeKind kind; ... }`
- `torch::Tensor dense_decode_q_nope_pe(const DenseDecodeParams&);`
- 匿名 namespace 声明 `.so` 导出的两个 C++ 符号（extern 签名严格匹配 §3.1 的 `nm` 输出）。

**新增** `xllm/core/kernels/dcu/flash_mla_adapter.cpp`：
- 早失败检查：`q_nope/q_pe/k_cache/seqlens_k/block_table` defined + DCU device；`q_nope.dim()==4 && q_pe.dim()==4`；`k_cache.dim()==4 && k_cache.size(2)==1`；`k_cache.size(-1)==q_nope.size(-1)+q_pe.size(-1)`；`block_table.dim()==2`；`seqlens_k.dim()==1`。
- 调 `get_mla_decoding_metadata_dense_fp8(seqlens_k, H_q_per_kv, H_q)` 取 `{tile_scheduler_metadata, num_splits}`。
- 调 `mha_fwd_kvcache_mla_nope_pe(...)` 取返回 vector 第 0 个（attention output）。
- `LOG(INFO) << "DCU MLA dense decode enabled: kind=q_nope_pe ..."`（一次性，用 glog `LOG_FIRST_N` 或自管 bool）。

**CMake**（`xllm/core/kernels/dcu/CMakeLists.txt`）：
- `DCU_LOCAL_HIP_SOURCES` 加 `flash_mla_adapter.cpp`。
- 查找 flash_mla `.so`（优先 dist-packages，可配置 `FLASH_MLA_LIB` cache 变量），建 `IMPORTED` target，加入 rpath；找不到则 `FATAL_ERROR`（镜像环境默认具备）。
- `dcu_kernels` DEPS 加 flash_mla target。

### Step 2 — DCU DeepseekV2Attention（G4）

**新增** `xllm/core/layers/dcu/deepseek_v2_attention.h/.cpp`。结构镜像 MLU 版但**剔除** SP / indexer / fused MLA qkv。成员（参考 `mlu/deepseek_v2_attention.h`）：

- `q_lora_rank_>0`：`ReplicatedLinear q_a_proj_`、`RMSNorm q_a_layernorm_`、`ColumnParallelLinear q_b_proj_`；
  否则（V2-Lite）：`ColumnParallelLinear q_proj_`。
- `ReplicatedLinear kv_a_proj_with_mqa_`、`RMSNorm kv_a_layernorm_`、`ColumnParallelLinear kv_b_proj_`（load 后提取 `w_kc_`/`w_vc_` 视图）。
- `std::shared_ptr<RotaryEmbeddingBase> rotary_emb_ = create_mla_rotary_embedding(...)`。
- `RowParallelLinear o_proj_`。
- 预填 fallback：`std::shared_ptr<TorchAttentionImpl>`（构造 `num_heads=tp_heads`, `head_size=kv_lora+rope`, `num_kv_heads=1`, `scale`）。**或**复用一个一次性 sdpa 调用而非持有 module（见 §7 风险 R3）。

方法：`prep_query`、`fill_q_input`（`bmm(q_nope, w_kc_)`）、`decode_kv_pre_base`（norm + rotary k_pe）、`store_latent_cache`（`reshape_paged_cache`）、`project_output`（`bmm(attn_out, w_vc_)` + `o_proj_`）、`forward(positions, hidden_states, attn_metadata, kv_cache)`。

> 与 MLU 关键差异：decode 时**保持 q_nope / q_pe 分离**直接喂 flash-mla（MLU 是拼成 q_input 喂自家 attn kernel）。

### Step 3 — DCU DeepseekV2DecoderLayer（G4）

**新增** `xllm/core/layers/dcu/deepseek_v2_decoder_layer_impl.h/.cpp`。结构镜像 `qwen3_moe_decoder_layer.{h,cpp}`：
- 成员：`DeepseekV2Attention attention_`、`DenseMLP mlp_`、`FusedMoE moe_mlp_`（DCU 版）、`RMSNorm input_norm_`/`post_norm_`、`ParallelArgs parallel_args_`、`bool is_moe_layer_`（由 `layer_id >= first_k_dense_replace` 决定）。
- `forward(x, residual, positions, attn_metadata, kv_cache, input_params)`：input_norm → attention → post_norm → (moe_mlp_ if is_moe else mlp_)，RMSNorm 用 `[out, residual]` 元组形式（参考 `qwen3_moe_decoder_layer.cpp:140-151`）。
- `load_state_dict` / `verify_loaded_weights` 转发子模块。

**CMake**（`xllm/core/layers/dcu/CMakeLists.txt`）：HDRS/SRCS 加两个新文件。

### Step 4 — 模型注册与参数（G1、G2）

- `models.h` DCU 分支（`#elif defined(USE_DCU)`）加 `#include "llm/deepseek_v3.h"`。
  - **关键**：V3-0324 / R1 的 `model_type=deepseek_v3`，需注册 deepseek_v3；而 `deepseek_v3.h` 顶部 `#include "deepseek_v2.h"`，故加 deepseek_v3.h 即同时注册 deepseek_v2 + deepseek_v3 两个模型类型。
  - deepseek_v32.h（V3.2）本阶段不在目标内，不接入。
- `deepseek_v2.h`：
  ```cpp
  #if defined(USE_DCU)
  #include "core/layers/dcu/deepseek_v2_decoder_layer_impl.h"
  #elif defined(USE_MLU)
  #include "core/layers/mlu/deepseek_v2_decoder_layer_impl.h"
  #endif
  ```
  args 加：`LOAD_ARG_OR(hidden_act, "hidden_act", "silu");` `LOAD_ARG_OR(scoring_func, "scoring_func", "softmax");`

### Step 5 — KV cache shape（G5）

见 §4.3。

### Step 6 — 构建 + BF16 冒烟（G6 验证）

```bash
cmake --build build/cmake.linux-x86_64-cpython-310 --target xllm -j8
# 单卡 V2-Lite
./build/xllm/core/server/xllm --model /mnt/deepseek-v2/DeepSeek-V2-Lite \
  --served_model_name DeepSeek-V2-Lite --devices=dcu:6 --block_size=64 \
  --enable_prefix_cache=false --enable_chunked_prefill=false --enable_graph=0
```
验证：`curl /v1/models` 返回模型；`/v1/chat/completions` 中文回答正常 + HTTP 200；日志出现一次 `DCU MLA dense decode enabled`。`pkill -9 xllm` 清理。

---

## 6. 与 vLLM-HCU / SGLang DCU 的对比

| 维度 | vLLM-HCU | SGLang DCU | xLLM DCU（本方案） |
| --- | --- | --- | --- |
| 接入语言 | Python backend | Python backend | **C++ 薄 adapter 直链 `.so`** |
| q 形态 | concat q（`flash_mla_with_kvcache`） | split q（`flash_mla_with_kvcache_q_nope_pe`） | **split q**（对齐 SGLang） |
| page_size | 可配置 | 强制 64 | 强制 64（冒烟 `--block_size=64`） |
| BF16 KV cache | `flash_mla_with_kvcache` | `flash_mla_with_kvcache_q_nope_pe` | `mha_fwd_kvcache_mla_nope_pe`（同符号 C++ 侧） |
| FP8 KV cache | `flash_mla_with_kvcache_fp8`（gfx938 + e4m3 传 q/k scale） | `flash_mla_with_kvcache_quantization_q_nope_pe` | Phase2 扩 `kQNopePeFp8WithCat` / `kQNopePeQuantized` |
| MLA q/k/v 投影 | Python torch | Python torch | C++ common Linear + bmm 吸收 w_kc_/w_vc_（镜像 MLU） |
| prefill | flash-mla varlen | flash-mla varlen | **torch sdpa fallback**（先求稳） |

> 本质：xLLM DCU = “SGLang DCU 的 split-q 策略 + MLU MLA 权重吸收 + C++ 直链 `.so`”。

---

## 7. 风险与待确认问题（自审）

| # | 风险 / 问题 | 处置 |
| --- | --- | --- |
| R1 | flash-mla `.so` 符号 ABI 已 `nm -D` 实测匹配，但 `get_mla_decoding_metadata_dense_fp8` 返回 vector 顺序需确认（tile_scheduler_metadata / num_splits） | adapter 内取返回 vector 时按 Python `flash_mla_interface.py` 顺序映射；首跑看日志 shape 校验 |
| R2 | RoPE interleaved 约定需与 flash-mla 一致（q_pe 已旋转、k_pe 存旋转后） | 镜像 SGLang `dcu_mla_backend.py` 的 rope；首跑对比 vllm/sglang 输出。MLU 设 `interleaved_=true`，DCU 默认对齐 |
| R3 | prefill torch sdpa fallback 是否复用现有 `TorchAttentionImpl`，还是 attention 内直接 `aten::scaled_dot_product_attention` | 倾向直接 sdpa（更少 module 耦合），但需确认 MLA 的 MQA 广播与 is_causal。若 `TorchAttentionImpl` 的 `expand_kv_for_mqa` 已够用则复用 |
| R4 | **MoE 不改（要求 2）**：V3/R1 W8A8 的 expert 权重是 FP8/INT8，但 DCU `fused_moe` 当前只 torch BF16/INT8 w13/w2。BF16 V2-Lite 不受影响；Phase2 V3/R1 必须处理 | Phase2 评估：是否用 aiter `fused_moe_asm` / `blaslt_scale_mm` 做 W8A8 expert GEMM；若不改 MoE 则 V3/R1 无法过，需给出理由并最小改动 |
| R5 | V3-0324 / R1 量化元数据键是 **`compression_config`**（非 `quantization_config`），compressed-tensors W8A8 FP8（channel weight / dynamic token act） | loader 识别 `compression_config`；权重键 `weight`+`weight_scale`。Phase2 处理 |
| R6 | R1-INT8：flash-mla **无原生 INT8 KV cache** API（仅 fp8_e4m3/e5m2） | 若 R1-INT8 仅权重/激活 INT8 而 KV cache 走 BF16/FP8，可复用 flash-mla；否则需另寻 kernel。Phase2 确认 R1-INT8 的实际 KV cache dtype |
| R7 | `block_size=64` 与 qwen 冒烟的 128 不同，需 deepseek 单独启动参数 | 冒烟脚本显式 `--block_size=64` |
| R8 | DCU `kv_b_proj` 权重吸收 `w_kc_`/`w_vc_` 在 TP 下需正确分片（`full_heads` vs `tp_heads`） | 镜像 MLU：`full_heads().proj_width(...)` + load 后 transpose；单卡 V2-Lite 先验证，多卡 TP 在 V3 验证 |
| R9 | `get_mla_decoding_metadata_dense_fp8` 第二/三参语义（`num_heads_per_head_k`、`num_heads_k`）需对齐 Python wrapper | 查 `flash_mla_interface.py::get_mla_decoding_metadata_dense_fp8` 确认传值（一般为 1、H_q） |
| R10 | **softmax_scale 推导** | **已解决（与参考方案 0.114721 吻合）**：SGLang 源码确认 `scaling = 1/sqrt(qk_nope_head_dim+qk_rope_head_dim) [* yarn mscale**2]`。V2-Lite 基础 1/√192=0.0722，含 yarn(factor=40, mscale=0.707) → mscale=0.1·0.707·ln40+1=1.2608 → 0.0722×1.2608²=**0.1148 ≈ 参考日志 0.114721**。即 MLU 的 `pow(qk_head_dim,-0.5)×mscale²` 公式（吸收不改 dot 秩 192）。attention 层算此值并显式传 adapter；adapter 强制要求 softmax_scale>0（拒绝 flash-mla 错误的 1/√576 默认） |

> **自审结论**：本计划覆盖 Phase 1 BF16 全部 gap（G1–G6），公共改动 3 处均有 device guard，flash-mla ABI 隔离在 adapter。R1–R3、R9 为接入期需在 adapter 首跑确认的 ABI/约定细节；R4–R6 明确归 Phase2，不阻塞 BF16 冒烟。

---

## 8. Phase 2（W8A8 / int8 / fp8）方向（后续）

### 8.1 先厘清“权重 W8A8”与“KV cache FP8/INT8”是两层问题

- **W8A8 权重**影响 Linear / MoE / gate / projection 的 GEMM。V3-0324/R1 的 `compression_config` 是 compressed-tensors FP8（channel weight + dynamic token act）。
- **flash-mla decode** 只关心 q / q_nope / q_pe、KV cache dtype、scale、block_table、seqlens。MLA 一个 W8A8 模型可暂用 **BF16 KV cache** 调 flash-mla（长上下文才需 FP8 KV cache 省显存）。

### 8.2 推荐接入（对齐 vllm-hcu/sglang，改动最小）

1. **W8A8 Linear**：参考 vllm-hcu DCU，优先用 **aiter** `blaslt_scale_mm` / awq_gemm_asm 走 `.so` adapter；checkpoint 兼容 compressed-tensors `weight`+`weight_scale`。
2. **MoE W8A8**（要求 2 的例外论证）：当前 DCU torch MoE 无法跑 FP8 expert 权重 → V3/R1 不过。**必须**接入 W8A8 expert GEMM（aiter `fused_moe_asm` / `fused_moe_asm_wna16`）。理由：不改则 V3/R1 冒烟无法通过，符合“如果一定要改 moe 部分，请给出理由”。
3. **flash-mla adapter 扩 `DenseDecodeKind`**：
   - `kQNopePeFp8WithCat → mha_fwd_kvcache_mla_fp8_with_cat`（k_cache fp8_e4m3 + split q）
   - `kQNopePeQuantized → mha_fwd_kvcache_quantization_q_nope_pe_mla`（k_cache fp8_e5m2 + k_scale）
   - `DenseDecodeParams` 加 `descale_q/descale_k/k_scale/kv_cache_dtype`。
4. **DCU FP8 KV cache**：补 `--kv_cache_dtype=fp8/fp8_e4m3/fp8_e5m2` 解析 + DCU FP8 cache tensor 创建 + scale 生命周期 + `store_latent_cache` 量化写入（独立于 W8A8 权重开关）。

### 8.3 R1-INT8 的 INT8 KV cache

flash-mla 当前仅支持 fp8_e4m3/e5m2 KV cache，无原生 INT8。若 R1-INT8 检查点实际 KV cache 仍 BF16/FP8（仅权重激活 INT8），可直接复用；否则需另寻 INT8 MLA kernel。Phase2 第一步先确认 R1-INT8 实际 KV cache dtype。

### 8.4 Phase 2 调研结论（2026-06-14，实施前阻塞）

深入核实后发现 Phase 2 有三个硬约束，与初始"优先 aiter + .so 导入"指令冲突，需决策后才能高效推进：

**约束 1 — aiter 无法 `.so` 直链**：每个 aiter 模块 `.so` 仅导出 `PyInit_<module>` 一个 `T` 符号，pybind host 函数（`dynamic_per_token_scaled_quant`、`asm_fmoe_a8`、`moe_c_moe_gemm_marlin_w8a8` 等）因 `-fvisibility=hidden` 未导出；也无 `torch.ops.aiter` 注册。→ aiter **只能 Python import 调用**（vllm-hcu/sglang 正是如此），与"不嵌 Python"要求冲突。此外 `module_gemm_a8w8` / `module_gemm_a8w8_asm`（INT8 dense GEMM）在已安装包中**未预编译**。

**约束 2 — R1-INT8 模型缺失**：`/mlsdeepseek0519/DeepSeek-R1-Channel-INT8` 不存在（目录为空）。实际可用 W8A8 模型只有 **V3-0324-Channel-FP8-w8a8** 和 **R1-Channel-FP8-w8a8**（均为 FP8 W8A8，671B 级，需多卡 TP）。

**约束 3 — DCU 已有 INT8 W8A8 Linear，缺 FP8 + MoE**：
- INT8 W8A8 Linear ✅ 已可用（`dcu::scaled_quantize` + `dcu::scaled_matmul` hipBLASLt，`is_compressed_tensors_w8a8_dynamic` gate 已 DCU 启用；与 vllm-hcu/lmslim 同库）。INT8 下 aiter 无现成优势（其 dense INT8 模块未预编译，hipBLASLt 即高性能正解）。
- FP8 W8A8 Linear ❌ 缺失（`fp8_scaled_matmul` 仅 CUDA/CUTLASS）。
- W8A8 MoE expert GEMM ❌ 缺失（DCU fused_moe 仅 torch BF16）。

**推荐路径（待确认）**：
- **A. 提供 INT8 模型 + 补 INT8 W8A8 MoE（hipBLASLt 复用，纯 C++）**：Linear 已通，MoE 用 per-expert `dcu::scaled_quantize`+`dcu::scaled_matmul` 循环（镜像现有 torch expert_gemm 结构）。改动最小、最快、可单卡测。需 INT8 权重。
- **B. FP8 + aiter-via-Python**：可测 V3-0324/R1-FP8，但需 (1) xLLM C++→Python 桥接 aiter（架构改动，放宽 .so 约束）；(2) FP8 Linear 走 aiter scale_mm；(3) FP8 MoE 走 aiter asm_moe；(4) 671B 多卡 TP。工作量大。
- **C. FP8 + 纯 C++（hipBLASLt FP8 / 移植 CUTLASS）**：不碰 aiter，但需新写 FP8 kernel，gfx938 hipBLASLt FP8 支持待验证，工作量大。

⚠️ **实施阻塞**：无 INT8 模型则 INT8 MoE 代码无法运行验证（Phase 1 经验：即便"显然"的代码也有 ~6 个运行期 bug）。不建议盲写大量未验证代码。

### 8.5 INT8 W8A8 MoE 已实现（2026-06-14，编译通过，待运行期验证）

用户选定 **option 1（放弃 aiter，纯 C++ hipBLASLt）** 后，已补齐 DCU INT8 W8A8 MoE（Linear 路径此前已通）：

- `xllm/core/layers/dcu/fused_moe.{h,cpp}`：
  - `is_w8a8_ = quant_args.is_compressed_tensors_w8a8_dynamic()`；构造时按 W8A8 分配 int8 `w13_`/`w2_` + fp32 per-channel `w13_scale_`/`w2_scale_`。
  - `load_experts`：`LOAD_MOE_FUSED_WEIGHT("weight_scale", w1_scale, w3_scale, w13_scale)`（与 weight 同融合顺序）+ `LOAD_MOE_WEIGHT("down_proj.", "weight_scale", w2_scale, 1)`。
  - 新增 `expert_gemm_w8a8`：per-expert 循环 → `dcu::scaled_quantize`（per-token int8）→ `dcu::scaled_matmul`（hipBLASLt INT8×INT8，per-channel scale），复用与 DCU W8A8 Linear 相同的 kernel。`forward_experts` 在 `is_w8a8_` 时走该路径。
- 编译 + 全量链接通过（BF16 路径无回归，`is_w8a8_` 默认 false）。

**仍待**：运行期验证需 INT8 W8A8 `compressed-tensors` 模型（R1-INT8 缺失）。**FP8（V3-0324/R1-FP8）**：需额外 FP8 Linear（DCU 现 scaled_matmul 仅 INT8）+ FP8 MoE，是更大独立工作，未做。

---

## 9. 任务进度追踪

| Task | 内容 | 状态 |
| --- | --- | --- |
| 1 | 编写本计划文档（自审） | ✅ 完成 |
| 2 | flash-mla adapter + CMake | ✅ 完成 |
| 3 | DCU DeepseekV2Attention | ✅ 完成 |
| 4 | DCU DeepseekV2DecoderLayer | ✅ 完成 |
| 5 | 模型注册 + 参数 + 设备 include | ✅ 完成 |
| 6 | KV cache shape DCU MLA | ✅ 完成 |
| 7 | 构建 + BF16 V2-Lite 冒烟 | ✅ 完成（见 §11） |
| 8 | [Phase2] W8A8 V3/R1 | ⏳ 待启动 |
| 9 | [Task3] 回填最终设计 | ✅ 完成（本文档） |

---

## 10. Phase 1 实际接入结果（Task 3 回填）

### 10.1 实际接入的 flash-mla API

链接 `/usr/local/lib/python3.10/dist-packages/flash_mla/cuda.cpython-310-x86_64-linux-gnu.so`，C++ 直链两个符号（全局作用域前向声明，由 CMake IMPORTED target 解析）：

- `get_mla_decoding_metadata_dense_fp8(seqlens_k, num_heads_per_head_k=H_q, num_heads_k=1)` → `{tile_scheduler_metadata, num_splits}`
- `mha_fwd_kvcache_mla_nope_pe(q_nope, q_pe, k_cache, vcache=nullopt, head_size_v=kv_lora_rank, seqlens_k, block_table, softmax_scale, is_causal=false, tile_scheduler_metadata, num_splits)` → `{out[B,1,H,kv_lora], lse}`

ldd 确认二进制已嵌入 `.so` 绝对路径（DT_NEEDED），运行期无需 LD_LIBRARY_PATH。

### 10.2 实现的功能（BF16）

- `xllm/core/kernels/dcu/flash_mla_adapter.{h,cpp}`：薄 adapter，唯一处声明 `.so` ABI；`DenseDecodeParams` + `dense_decode()`；早失败 shape/device 检查 + 一次性 LOG。
- `xllm/core/layers/dcu/deepseek_v2_attention.{h,cpp}`：MLA attention，镜像 MLU 权重吸收（`w_kc_`/`w_vc_`），**decode 走 flash-mla split-q，prefill 走 torch sdpa over latent KV**（V=latent 取前 kv_lora 列 = o_latente）。
- `xllm/core/layers/dcu/deepseek_v2_decoder_layer_impl.{h,cpp}`：结构镜像 qwen3_moe（input_norm → MLA attn → post_norm → DenseMLP/FusedMoE by `first_k_dense_replace`）。
- `models.h`：DCU 分支加 `deepseek_v3.h`（同时注册 v2/v3）。
- `deepseek_v2.h`：设备条件 include + 补 `hidden_act`/`scoring_func`。
- `kv_cache_shape.cpp`：`apply_device_layout` 加 `USE_DCU` MLA latent 分支（key-only `[blocks,64,1,kv_lora+rope]`）。

### 10.3 接入方式

- **`.so` 链接**：`kernels/dcu/CMakeLists.txt` 用 `file(GLOB)` 定位 `cuda.*.so`，建 `IMPORTED` target，加入 `dcu_kernels` DEPS，绝对路径进 DT_NEEDED。
- **公共主干**：仅 `models.h`/`deepseek_v2.h`/`kv_cache_shape.cpp`/`ops_api.cpp` 4 处，全部 device guard。
- **构建**：`FLASH_ATTENTION_LIB=... PYTORCH_ROCM_ARCH=gfx938 PYTORCH_INSTALL_PATH=/usr/local/.../torch python setup.py build --device dcu`。

### 10.4 接入期发现并修复的框架 bug（4 个）

| # | bug | 修复 |
| --- | --- | --- |
| B1 | `apply_rotary`（CUDA/MUSA/DCU 分支）把 MLA 单 tensor 路径下**未定义**的 `params.k`（`torch::Tensor`）隐式转成**非空** `std::optional`，kernel 对 undefined tensor 检查 size 崩溃 | `ops_api.cpp`：`undefined → std::nullopt`，仅影响 MLA，2-tensor 路径不变 |
| B2 | `cuda::reshape_paged_cache` 对 MLA key-only（未设 value/v_cache）传入 undefined tensor，访问 `.strides()` 崩溃 | MLA `store_latent_cache` 改用直接 `index_copy_` 写 paged cache（设备本地，不碰公共 kernel） |
| B3 | ROCm/HIP PyTorch 设备类型是 `kCUDA`（非 `kHIP`），adapter device check 误判 | adapter 改用 `k_cache.device()` 作参照 |
| B4 | prefill sdpa 默认 scale `1/√576=0.0417` 与 MLA 正确 scale `0.1147` 不符 → prefill 输出错误 | sdpa 显式传 `scale=softmax_scale_` + `enable_gqa=true` |

### 10.5 BF16 正确性验证（DeepSeek-V2-Lite，单卡 dcu:6，block_size=64）

flash-mla decode 命中日志（与参考方案完全一致）：
```
DCU MLA dense decode enabled: kind=q_nope_pe, q_nope=[1,1,16,512], q_pe=[1,1,16,64],
k_cache=[31406,64,1,576], block_table=[1,1], seqlens_k=[1], head_size_v=512, softmax_scale=0.114721
```
补全正确性（temperature=0）：

| Prompt | 输出 | 正确 |
| --- | --- | --- |
| `中国的首都是` | `北京。` | ✅ |
| `1+1=` | `2` | ✅ |
| `The capital of France is` | ` Paris.` | ✅ |
| `<｜begin▁of▁sentence｜>中国的首都是` | `北京。`（bos 正确 tokenize 为 id 100000） | ✅ |

> **chat completion 说明**：`/mnt/deepseek-v2/DeepSeek-V2-Lite` 是 **Base 版本**（非 `-Chat`，参考方案用的是 `DeepSeek-V2-Lite-Chat`）。Base 模型不遵循 chat 指令 → `/v1/chat/completions` 产出空白/换行；但**补全类请求与 MLA/flash-mla 正确性无关，已充分验证**。需 chat 冒烟请提供 `-Chat` 权重。

### 10.6 与 vLLM-HCU / SGLang 的实际差异

- xLLM 用 **C++ 直链 `.so`**（vllm/sglang 用 Python backend）。
- MLA q/k/v 投影 + 权重吸收在 **C++ common Linear + bmm**（vllm/sglang 在 Python torch）。
- prefill 用 **torch sdpa over latent**（vllm/sglang 用 flash-mla varlen）—— 首版求稳，后续可换 flash-mla prefill。
- decode split-q 策略、page_size=64、softmax_scale 公式（`1/√(qk_nope+qk_rope) × mscale²`）与 SGLang/vLLM-HCU 完全一致。

### 10.7 优化方向

1. **prefill**：sdpa fallback → flash-mla varlen prefill API（`flash_mla_with_kvcache`），长 prompt 性能更优。
2. **W8A8（Phase 2）**：V3-0324/R1 的 `compression_config`（compressed-tensors FP8 channel/dynamic）需 W8A8 Linear + MoE expert GEMM（优先 aiter `.so`），见 §8。当前 DCU torch MoE 只支持 BF16，V3/R1 必须扩 MoE（符合要求 2 的“给出理由”）。
3. **FP8 KV cache**：扩 adapter `DenseDecodeKind`（`kQNopePeFp8WithCat` / `kQNopePeQuantized`）+ DCU FP8 cache 创建/写入，长上下文省显存。
4. **多卡 TP**：V2-Lite 单卡已验证；V3/R1 多卡 TP 需验证 `tp_heads` 权重分片与 reduce。
5. **B1/B2 框架 bug**：可上游反馈（apply_rotary undefined-k、reshape_paged_cache undefined-value），让 MLA 在 CUDA/MUSA 上也可用。
