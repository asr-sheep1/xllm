/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Thin adapter around the flash-mla native extension
// (/usr/local/lib/python3.10/dist-packages/flash_mla/cuda.*.so).
//
// All flash-mla ABI details (symbol names, return types, the page-size==64 /
// single-KV-head constraints, the split-q decode shape contract) live ONLY in
// the corresponding .cpp. Downstream DCU MLA layers call the clean wrapper
// below and never declare `mha_fwd_*` symbols themselves, keeping the rest of
// xLLM free of vendor-ABI assumptions.

#ifndef XLLM_CORE_KERNELS_DCU_FLASH_MLA_ADAPTER_H_
#define XLLM_CORE_KERNELS_DCU_FLASH_MLA_ADAPTER_H_

#include <torch/torch.h>

#include <optional>
#include <vector>

namespace xllm {
namespace kernel {
namespace dcu {
namespace flash_mla {

// Decode variant. BF16/FP16 split-q (q_nope already absorbed to kv_lora_rank,
// q_pe already rotated) is the only one wired up in phase 1; FP8 / quantized
// KV-cache kinds are reserved for phase 2 (see dcu_flash_mla_plan.md §8).
enum class DenseDecodeKind {
  kQNopePe,  // -> mha_fwd_kvcache_mla_nope_pe  (BF16/FP16 split-q)
  // kQNopePeFp8WithCat,    // -> mha_fwd_kvcache_mla_fp8_with_cat      (phase
  // 2) kQNopePeQuantized,     // -> mha_fwd_kvcache_quantization_q_nope_pe_mla
  // (phase 2)
};

// Inputs for a flash-mla dense (non-sparse) MLA decode. All tensors must be on
// the DCU device and contiguous.
//
//   q_nope:      [B, S_q, H_q, kv_lora_rank]  (absorbed, bf16/fp16)
//   q_pe:        [B, S_q, H_q, qk_rope_head_dim]  (already rotated)
//   k_cache:     [num_blocks, 64, 1, kv_lora_rank + qk_rope_head_dim]
//   seqlens_k:   [B] int32  (per-sequence cached kv length)
//   block_table: [B, max_blocks] int32  (-1 padded; >=0 page ids otherwise)
//   head_size_v: kv_lora_rank (flash-mla outputs in latent / v-lora space)
//   softmax_scale: REQUIRED (>0). MLA scale =
//     1/sqrt(qk_nope_head_dim + qk_rope_head_dim) [* yarn mscale**2 if yarn].
//     The caller computes it once so prefill (sdpa) and decode (flash-mla)
//     agree; flash-mla's own default (1/sqrt(kv_lora+rope)) is wrong for the
//     absorbed form, so we refuse to fall back to it.
struct DenseDecodeParams {
  torch::Tensor q_nope;
  torch::Tensor q_pe;
  torch::Tensor k_cache;
  torch::Tensor seqlens_k;
  torch::Tensor block_table;
  int64_t head_size_v = 0;
  float softmax_scale = -1.0F;
  DenseDecodeKind kind = DenseDecodeKind::kQNopePe;
};

// Runs the split-q MLA decode and returns the attention output
// [B, S_q, H_q, head_size_v] on the DCU device. Performs early-fail shape /
// device checks and emits a one-shot LOG(INFO) on first hit. Non-const ref
// because the flash-mla .so symbols take at::Tensor& (handle-level).
torch::Tensor dense_decode(DenseDecodeParams& params);

}  // namespace flash_mla
}  // namespace dcu
}  // namespace kernel
}  // namespace xllm

#endif  // XLLM_CORE_KERNELS_DCU_FLASH_MLA_ADAPTER_H_
