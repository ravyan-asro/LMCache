# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Third Party
from torch import nn
import torch

# First Party
from lmcache.v1.compute.attention.flash_attn import LMCFlashAttnBackend
from lmcache.v1.compute.attention.metadata import LMCFlashAttnMetadata
from lmcache.v1.compute.positional_encoding import get_fused_rope

# FIXME(Jiayi): A few things need to be tested/supported:
# PP, Multimodal


class LMCLlamaModel(nn.Module):
    def __init__(
        self,
        vllm_model,
        blender,
    ):
        super().__init__()
        self.vllm_model = vllm_model

        # Compat: vLLM 0.18+ uses embed_input_ids(), older uses get_input_embeddings()
        if hasattr(vllm_model, "embed_input_ids"):
            self._embed = vllm_model.embed_input_ids
        elif hasattr(vllm_model, "get_input_embeddings"):
            self._embed = vllm_model.get_input_embeddings
        else:
            self._embed = vllm_model.model.embed_tokens

        self.num_layers = len(vllm_model.model.layers)

        self.vllm_attn_layers = []
        self.lmc_attn_layers = []
        for i in range(self.num_layers):
            vllm_attn = vllm_model.model.layers[i].self_attn.attn
            self.vllm_attn_layers.append(vllm_attn)
            self.lmc_attn_layers.append(LMCFlashAttnBackend(vllm_attn))

        # NOTE(Jiayi): better not to pass the blender in init
        # if we want to make this LMCModel more general.
        self.blender = blender

        # Detect QK-norm (e.g. Qwen3 MoE, MiniMax M2.5) — must be applied
        # between QKV split and RoPE.
        attn0 = vllm_model.model.layers[0].self_attn
        self.has_qk_norm = hasattr(attn0, "q_norm") and hasattr(attn0, "k_norm")

        # Detect MiniMax-style QK-norm (uses static forward_qk method)
        self.minimax_qk_norm = False
        if self.has_qk_norm:
            try:
                from vllm.model_executor.layers.mamba.linear_attn import (
                    MiniMaxText01RMSNormTP,
                )
                if isinstance(attn0.q_norm, MiniMaxText01RMSNormTP):
                    self.minimax_qk_norm = True
            except ImportError:
                pass

        # Detect MLP attribute name: 'mlp' (Llama) or 'block_sparse_moe' (MiniMax)
        layer0 = vllm_model.model.layers[0]
        if hasattr(layer0, "mlp"):
            self.mlp_attr = "mlp"
        elif hasattr(layer0, "block_sparse_moe"):
            self.mlp_attr = "block_sparse_moe"
        else:
            raise ValueError("Cannot find MLP/MoE attribute on model layer")

        rotary_emb = vllm_model.model.layers[0].self_attn.rotary_emb
        head_dim = rotary_emb.head_size
        max_position_embeddings = rotary_emb.max_position_embeddings
        base = rotary_emb.base
        is_neox_style = rotary_emb.is_neox_style
        dtype = rotary_emb.dtype

        # Detect partial rotary (e.g. MiniMax M2.5: rotary_dim=64, head_dim=128)
        rotary_dim = getattr(vllm_model.config, 'rotary_dim',
                    getattr(vllm_model.config, 'partial_rotary_factor', 1.0) * head_dim)
        rotary_dim = int(rotary_dim) if rotary_dim != head_dim else head_dim

        rope_scaling = getattr(vllm_model.config, 'rope_scaling', None)
        self.fused_rotary_emb = get_fused_rope(
            head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position_embeddings,
            base=base,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )

    def compute_layer(
        self,
        input_ids: torch.Tensor,
    ):
        # Disable vLLM 0.18+ MoE layer tracking during our manual forward
        # to avoid "all_moe_layers" count mismatch.
        try:
            from vllm.forward_context import get_forward_context
            ctx = get_forward_context()
            saved_moe_layers = ctx.all_moe_layers
            saved_moe_idx = ctx.moe_layer_index
            ctx.all_moe_layers = None
            ctx.moe_layer_index = 0
        except Exception:
            ctx = None
            saved_moe_layers = None
            saved_moe_idx = 0

        hidden_states = self._embed(input_ids.cuda())
        residual = None

        # TODO (Jiayi): reduce the number of calls
        attn_output = None

        # TODO(Jiayi): Need to build `attn_metadata` more elegantly.
        attn_metadata = LMCFlashAttnMetadata(
            query_start_loc=torch.tensor(
                [0, input_ids.shape[0]], dtype=torch.int32, device=hidden_states.device
            ),
            seq_lens=torch.tensor([input_ids.shape[0]], device=hidden_states.device),
            cu_seqlens_k=torch.tensor(
                [0, input_ids.shape[0]], dtype=torch.int32, device=hidden_states.device
            ),
            max_query_len=input_ids.shape[0],
            max_seq_len=input_ids.shape[0],
        )

        for idx, layer in enumerate(
            self.vllm_model.model.layers[
                self.vllm_model.model.start_layer : self.vllm_model.model.end_layer
            ]
        ):
            # TODO(Jiayi) The last layer doesn't have to be computed
            # hidden_states, residual = layer(positions, hidden_states, residual)

            # Self Attention
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(hidden_states, residual)
            # hidden_states = self.self_attn(positions=positions,
            #                            hidden_states=hidden_states)

            qkv, _ = layer.self_attn.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                [
                    layer.self_attn.q_size,
                    layer.self_attn.kv_size,
                    layer.self_attn.kv_size,
                ],
                dim=-1,
            )

            # QK-norm (Qwen3 MoE, MiniMax M2.5, and similar models)
            if self.has_qk_norm:
                num_heads = self.vllm_attn_layers[idx].num_heads
                num_kv_heads = self.vllm_attn_layers[idx].num_kv_heads
                head_size = self.vllm_attn_layers[idx].head_size
                if self.minimax_qk_norm:
                    # MiniMax uses static forward_qk for TP-synchronized QK-norm
                    from vllm.model_executor.layers.mamba.linear_attn import (
                        MiniMaxText01RMSNormTP,
                    )
                    q, k = MiniMaxText01RMSNormTP.forward_qk(
                        layer.self_attn.q_norm, layer.self_attn.k_norm, q, k
                    )
                else:
                    q = layer.self_attn.q_norm(
                        q.view(*q.shape[:-1], num_heads, head_size)
                    ).view(q.shape)
                    k = layer.self_attn.k_norm(
                        k.view(*k.shape[:-1], num_kv_heads, head_size)
                    ).view(k.shape)

            q, k, v, residual, attn_output, attn_metadata = self.blender.process_qkv(
                q, k, v, residual, idx, attn_output, attn_metadata
            )

            num_heads = self.vllm_attn_layers[idx].num_heads
            num_kv_heads = self.vllm_attn_layers[idx].num_kv_heads
            head_size = self.vllm_attn_layers[idx].head_size

            q = q.view(-1, num_heads, head_size)
            k = k.view(-1, num_kv_heads, head_size)
            v = v.view(-1, num_kv_heads, head_size)
            attn_output = attn_output.view(-1, num_heads, head_size)

            attn_output = self.lmc_attn_layers[idx].forward_contiguous(
                q, k, v, attn_output, attn_metadata
            )

            attn_output = attn_output.view(-1, num_heads * head_size)
            k = k.view(-1, num_kv_heads * head_size)
            v = v.view(-1, num_kv_heads * head_size)

            hidden_states, _ = layer.self_attn.o_proj(attn_output)

            # Fully Connected
            hidden_states, residual = layer.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = getattr(layer, self.mlp_attr)(hidden_states)

            yield

        # Restore MoE tracking after manual forward completes
        if ctx is not None:
            ctx.all_moe_layers = saved_moe_layers
            ctx.moe_layer_index = saved_moe_idx
