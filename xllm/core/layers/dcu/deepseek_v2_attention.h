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

// DCU DeepSeek-V2 MLA attention.
//
// Simplified port of layers/mlu/deepseek_v2_attention.* : drops the MLU-only
// sequence-parallel / lighting-indexer / fused-MLA-qkv machinery and instead
//
//   * DECODE  -> flash-mla split-q (q_nope absorbed to kv_lora_rank, q_pe
//                rotated) via kernels/dcu/flash_mla_adapter (the .so adapter).
//   * PREFILL -> torch scaled_dot_product_attention over the latent KV.
//
// Weight absorption (q_nope @ w_kc_, attn_out @ w_vc_) is identical to the MLU
// path so the latent key-only paged cache layout is shared.

#ifndef XLLM_CORE_LAYERS_DCU_DEEPSEEK_V2_ATTENTION_H_
#define XLLM_CORE_LAYERS_DCU_DEEPSEEK_V2_ATTENTION_H_

#include <torch/torch.h>

#include <memory>
#include <optional>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/linear.h"
#include "layers/common/rms_norm.h"
#include "layers/common/rotary_embedding.h"

namespace xllm {
namespace layer {

class DeepseekV2AttentionImpl final : public torch::nn::Module {
 public:
  DeepseekV2AttentionImpl(const ModelArgs& args,
                          const QuantArgs& quant_args,
                          const ParallelArgs& parallel_args,
                          const torch::TensorOptions& options);

  // positions:      [num_tokens] token positions
  // hidden_states:  [num_tokens, hidden_size]
  // Returns attention output [num_tokens, hidden_size].
  torch::Tensor forward(const torch::Tensor& positions,
                        const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache);

  void load_state_dict(const StateDict& state_dict);

 private:
  // Builds q [num_tokens, tp_heads, qk_head_dim_] from hidden_states.
  torch::Tensor prepare_query(const torch::Tensor& hidden_states);

  // Stores the latent KV [num_tokens, kv_lora + qk_rope] into the paged
  // key-only cache using slot_mapping.
  void store_latent_cache(const torch::Tensor& latent_cache,
                          const torch::Tensor& slot_mapping,
                          const torch::Tensor& k_cache);

  // Decode via flash-mla split-q. q_nope_absorbed is [num_tokens, tp_heads,
  // kv_lora_rank] (already absorbed); q_pe is [num_tokens, tp_heads,
  // qk_rope_head_dim] (already rotated). Both prepared in forward().
  torch::Tensor decode_flash_mla(const torch::Tensor& q_nope_absorbed,
                                 const torch::Tensor& q_pe,
                                 const AttentionMetadata& attn_metadata,
                                 KVCache& kv_cache);

  // Prefill via torch sdpa over latent KV (V = latent, take first kv_lora
  // cols).
  torch::Tensor prefill_sdpa(const torch::Tensor& q_nope_absorbed,
                             const torch::Tensor& q_pe,
                             const torch::Tensor& latent_cache,
                             const AttentionMetadata& attn_metadata,
                             const torch::Tensor& slot_mapping,
                             const torch::Tensor& k_cache);

  // bmm(attn_out_latent, w_vc_) -> o_proj. attn_latent is [tokens, heads, kv].
  torch::Tensor project_output(const torch::Tensor& attn_latent);

  int64_t q_lora_rank_;
  int64_t kv_lora_rank_;
  int64_t qk_nope_head_dim_;
  int64_t qk_rope_head_dim_;
  int64_t qk_head_dim_;
  int64_t v_head_dim_;
  int64_t tp_heads_;
  double eps_;
  float softmax_scale_;
  bool interleaved_;

  ReplicatedLinear q_a_proj_{nullptr};
  RMSNorm q_a_layernorm_{nullptr};
  ColumnParallelLinear q_b_proj_{nullptr};
  ColumnParallelLinear q_proj_{nullptr};  // used when q_lora_rank_ == 0

  ReplicatedLinear kv_a_proj_with_mqa_{nullptr};
  RMSNorm kv_a_layernorm_{nullptr};
  ColumnParallelLinear kv_b_proj_{nullptr};
  RowParallelLinear o_proj_{nullptr};

  torch::Tensor w_kc_;  // [tp_heads, qk_nope, kv_lora]
  torch::Tensor w_vc_;  // [tp_heads, kv_lora, v_head_dim]
  bool has_trans_ = false;

  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;
};

TORCH_MODULE(DeepseekV2Attention);

}  // namespace layer
}  // namespace xllm

#endif  // XLLM_CORE_LAYERS_DCU_DEEPSEEK_V2_ATTENTION_H_
