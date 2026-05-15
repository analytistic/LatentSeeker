from dataclasses import dataclass
from typing import List
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, GenerationMixin, GradientCheckpointingLayer
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import ModelOutput
from transformers.utils.generic import TransformersKwargs, is_flash_attention_requested
from transformers.processing_utils import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS                                             
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextAttention,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLTextModel,
    Qwen3VLTextMLP,
    apply_rotary_pos_emb,
    eager_attention_forward,
    BaseModelOutputWithDeepstackFeatures  
)

from .configuration_LatentSeeker import LatentSeekerConfig, LatentEncoderConfig


class LatentSeekerTextModel(Qwen3VLTextModel):
    """Qwen3VLTextModel with an additional longtext deepstack injection path.

    Injects longtext deepstack features at the first N decoder layers,
    parallel to the existing visual deepstack path.
    """

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        # visual deepstack
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        # longtext deepstack
        longtext_pos_masks: torch.Tensor | None = None,
        deepstack_longtext_embeds: list[torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        from transformers.cache_utils import DynamicCache
        from transformers.models.qwen3_vl.modeling_qwen3_vl import create_causal_mask

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        attention_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx],
                )

            if deepstack_longtext_embeds is not None and layer_idx in range(len(deepstack_longtext_embeds)):
                hidden_states = self._longtext_deepstack_process(
                    hidden_states, longtext_pos_masks, deepstack_longtext_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _longtext_deepstack_process(
        self, hidden_states: torch.Tensor, pos_masks: torch.Tensor, embed: torch.Tensor
    ) -> torch.Tensor:
        pos_masks = pos_masks.to(hidden_states.device)
        embed = embed.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        local_this = hidden_states[pos_masks, :] + embed
        hidden_states[pos_masks, :] = local_this
        return hidden_states


@dataclass
class MergerOutput(ModelOutput):
    """
    Output of LongertextMerger.

    Args:
        pooled (`torch.FloatTensor` of shape `(total_num_tokens, hidden_size)`):
            Compressed latent tokens after gated pooling.
        svd_loss (`torch.FloatTensor`, *optional*):
            SVD reconstruction loss, ``None`` if no tokens to compress.
    """
    pooled: torch.FloatTensor
    svd_loss: torch.FloatTensor | None = None


class LongertextMerger(nn.Module):
    """Merger with learned gated pooling + SVD reconstruction loss.

    Instead of uniform average, each token gets a learned weight via a linear gate.
    A SVD auxiliary loss constrains the compression to retain maximal information.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, 1, bias=False)
        self.mlp = LatentSeekerEncoderMLP(config)  # kept for potential future use

    def forward(self, longtext_embeds: torch.Tensor, longtext_cu_seqlens: torch.Tensor,
                longtext_num_tokens: list[int]) -> MergerOutput:
        device = longtext_embeds.device
        num_docs = len(longtext_num_tokens)
        total_output = sum(longtext_num_tokens)
        if total_output == 0:
            return MergerOutput(
                pooled=torch.zeros(0, longtext_embeds.shape[-1], device=device, dtype=longtext_embeds.dtype),
            )

        # Build output cumulative offsets: [0, q0, q0+q1, ...]
        out_cu = [0]
        for n in longtext_num_tokens:
            out_cu.append(out_cu[-1] + n)

        # Map each input token to its target output index, with overlap
        pool_indices = []
        embed_extra_idx = []
        for i in range(num_docs):
            seg_len = longtext_cu_seqlens[i + 1] - longtext_cu_seqlens[i]
            target = longtext_num_tokens[i]
            if seg_len == 0 or target == 0:
                continue
            offset = out_cu[i]

            main = torch.arange(seg_len, device=device) * target // seg_len + offset
            pool_indices.append(main)

            # Overlap: first ~ratio/2 tokens of each bin also map to previous bin
            if target > 1:
                stride = seg_len / target
                overlap_len = max(1, int(stride / 2))
                for b in range(1, target):
                    bin_mask = (main - offset) == b
                    positions = torch.where(bin_mask)[0]
                    if len(positions) == 0:
                        continue
                    overlap_positions = positions[:overlap_len]
                    if len(overlap_positions) > 0:
                        pool_indices.append(torch.full(
                            (len(overlap_positions),), b - 1 + offset, device=device
                        ))
                        embed_extra_idx.append(overlap_positions)

        pool_indices = torch.cat(pool_indices, dim=0)

        # Duplicate embedding entries for overlapped positions
        if embed_extra_idx:
            extra_idx = torch.cat(embed_extra_idx)
            longtext_embeds = torch.cat([longtext_embeds, longtext_embeds[extra_idx]], dim=0)

        # Learned gate: each token gets a score
        gate_scores = self.gate(longtext_embeds).squeeze(-1)  # [N]

        # Softmax weights within each bin
 max_score = gate_scores.max()
        gate_scores_exp = (gate_scores - max_score).exp()  # [N]
        weight_sum = torch.zeros(total_output, device=device, dtype=longtext_embeds.dtype)
        weight_sum.index_add_(0, pool_indices, gate_scores_exp)
        weights = gate_scores_exp / weight_sum[pool_indices].clamp(min=1e-8)  # [N]

        # Weighted sum into target positions
        weighted = weights.unsqueeze(-1) * longtext_embeds  # [N, D]
        summed = torch.zeros(total_output, longtext_embeds.shape[-1], device=device, dtype=longtext_embeds.dtype)
        summed.index_add_(0, pool_indices, weighted)

        # SVD reconstruction loss: c must be able to reconstruct original tokens
        pooled = summed  # weighted sum (weights already normalized per bin)

        # Compute svd_loss: ||x - αc||² where α = (x·c) / ||c||²
 c_expanded = pooled[pool_indices]
        dot = (longtext_embeds * c_expanded).sum(dim=-1, keepdim=True)
 c_norm_sq = (pooled * pooled).sum(dim=-1, keepdim=True)
        alpha = dot / c_norm_sq[pool_indices].clamp(min=1e-8)
        recon = alpha * c_expanded
        svd_loss = F.mse_loss(recon, longtext_embeds)

        return MergerOutput(pooled=pooled, svd_loss=svd_loss)


class LatentSeekerEncoderMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = getattr(config, "intermediate_size", 4 * config.hidden_size)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class LatentSeekerPreTrainedModel(PreTrainedModel):
    config_class = LatentSeekerConfig
    base_model_prefix = "model"
    input_modalities = ["text", "longtext"]
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3VLTextDecoderLayer,
        "attentions": Qwen3VLTextAttention,
    }

    def _init_weights(self, module):
        super()._init_weights(module)


class LongtextAttention(nn.Module):
    def __init__(self, config: LatentEncoderConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3VLTextRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        seq_length = hidden_states.shape[0] 
        hidden_shape = (seq_length, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(0, 1).unsqueeze(0)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(0, 1).unsqueeze(0)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(0, 1).unsqueeze(0)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if is_flash_attention_requested(self.config):
            # Flash Attention: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output



        

class LongtextEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LatentEncoderConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attn = LongtextAttention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LatentSeekerEncoderModel(LatentSeekerPreTrainedModel):
    """
    Encoder: compresses longtext documents into latent embeddings.

    Each doc processed independently: embed tokens -> append learnable query tokens ->
    run through decoder layers -> extract query hidden states as latents.
    All doc latents concatenated into flat output.
    """
    config: LatentEncoderConfig
    input_modalities = ["longtext"]


    def __init__(self, config: LatentEncoderConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # self.query_num = config.query_num

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )
        # self.query_tokens = nn.Embedding(config.query_num, config.hidden_size)
        self.layers = nn.ModuleList([
            LongtextEncoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.merger = LongertextMerger(config)
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def _compute_position_ids(self, cu_seqlens, device):                                                                                                                    
        positions = []                                                                                          
        for i in range(len(cu_seqlens) - 1):                                                                    
            seg_len = cu_seqlens[i + 1] - cu_seqlens[i]                                                         
            positions.append(torch.arange(seg_len, device=device))                                              
        return torch.cat(positions, dim=0)  # (total_seq_len,) 
    

    def forward(
        self,
        longtext_input_ids: torch.LongTensor | List[int],
        longtext_cu_seqlens: list[int] | None = None,
        longtext_num_tokens: list[int] | None = None,
        **kwargs,
    ) -> BaseModelOutputWithDeepstackFeatures:
        if isinstance(longtext_input_ids, (list, tuple)):
            device = next(self.parameters()).device
            longtext_input_ids = torch.tensor(longtext_input_ids, dtype=torch.long, device=device)

        longtext_embeds = self.embed_tokens(longtext_input_ids)
        device = longtext_input_ids.device
        position_ids = self._compute_position_ids(longtext_cu_seqlens, device)
        position_embeddings = self.rotary_emb(longtext_embeds.unsqueeze(0), position_ids.unsqueeze(0))

    
        if longtext_embeds.shape[0] == 0:
            return BaseModelOutputWithDeepstackFeatures()

        deepstack_features = []
        for layer_num, blk in enumerate(self.layers):
            longtext_embeds: torch.FloatTensor = blk(
                longtext_embeds,
                cu_seqlens=longtext_cu_seqlens,
                position_embeddings=position_embeddings,
            )

            if layer_num in self.config.deepstack_latent_indexes:
                merger_out = self.merger(longtext_embeds, longtext_cu_seqlens, longtext_num_tokens)
                deepstack_features.append(merger_out.pooled)

        pooler_output = deepstack_features[-1] if deepstack_features else None

        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=longtext_embeds,
            pooler_output=pooler_output,
            deepstack_features=deepstack_features,
            loss=merger_out.svd_loss if deepstack_features else None,
        )
    



class LatentSeekerModel(LatentSeekerPreTrainedModel):
    """LatentSeeker base model: latent encoder + Qwen3VL text decoder."""
    config: LatentSeekerConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]

    def __init__(self, config: LatentSeekerConfig):
        super().__init__(config)
        self.longtext = LatentSeekerEncoderModel._from_config(config.longtext_config)
        self.language_model = LatentSeekerTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        longtext_input_ids: list[int] | torch.LongTensor | None = None,
        longtext_cu_seqlens: list[int] | torch.LongTensor | None = None,
        longtext_num_tokens: list[int] | torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of `input_ids` or `inputs_embeds`.")

        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 1. Encode longtext into latents and replace placeholders
        svd_loss = None
        deepstack_features = None
        longtext_mask = None
        if longtext_input_ids is not None:
            longtext_outputs = self.longtext(
                longtext_input_ids,
                longtext_cu_seqlens=longtext_cu_seqlens,
                longtext_num_tokens=longtext_num_tokens,
            )
            longtext_embeds = longtext_outputs.pooler_output
            svd_loss = getattr(longtext_outputs, "loss", None)
            deepstack_features = getattr(longtext_outputs, "deepstack_features", None)

            # get_placeholder_mask
            longtext_token_id = self.config.longtext_token_id
            longtext_mask = input_ids == longtext_token_id
            n_placeholders = longtext_mask.sum().item()
            n_longtext = longtext_embeds.shape[0]
            if n_placeholders != n_longtext:
                raise ValueError(
                    f"Longtext placeholder count ({n_placeholders}) != longtext count ({n_longtext})."
                )

            longtext_mask_3d = longtext_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(longtext_mask_3d, longtext_embeds)

        # 2. Text model forward (with deepstack injection if applicable)
        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            longtext_pos_masks=longtext_mask,
            deepstack_longtext_embeds=deepstack_features,
            **kwargs,
        )

        if svd_loss is not None:
            outputs.loss = svd_loss

        return outputs


class LatentSeekerForConditionalGeneration(LatentSeekerPreTrainedModel, GenerationMixin):
    """LatentSeeker with LM head for text generation."""
    config: LatentSeekerConfig
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: LatentSeekerConfig):
        super().__init__(config)
        self.model = LatentSeekerModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False,
        )
        self.post_init()

    @classmethod
    def init_from_pretrained(cls, pretrained_path: str, config=None, **kwargs):
        if config is None:
            config = LatentSeekerConfig.from_pretrained(pretrained_path)

        model = cls.from_pretrained(
            pretrained_path,
            config=config,
            ignore_mismatched_sizes=True,
            **kwargs,
        )

        model.model.longtext.embed_tokens.weight.data.copy_(
            model.model.language_model.embed_tokens.weight.data
        )

        # Copy encoder layer weights from LM pretrained layers
        num_encoder_layers = len(model.model.longtext.layers)
        for i in range(num_encoder_layers):
            src = model.model.language_model.layers[i]
            dst = model.model.longtext.layers[i]

            # Attention projections (identical shapes, different class name)
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                getattr(dst.attn, proj).weight.data.copy_(
                    getattr(src.self_attn, proj).weight.data
                )
                bias_src = getattr(src.self_attn, proj).bias
                if bias_src is not None:
                    getattr(dst.attn, proj).bias.data.copy_(bias_src.data)

            # QK RMSNorm
            dst.attn.q_norm.weight.data.copy_(src.self_attn.q_norm.weight.data)
            dst.attn.k_norm.weight.data.copy_(src.self_attn.k_norm.weight.data)

            # MLP (same class)
            dst.mlp.load_state_dict(src.mlp.state_dict())

            # Layer norms (same class)
            dst.input_layernorm.weight.data.copy_(src.input_layernorm.weight.data)
            dst.post_attention_layernorm.weight.data.copy_(
                src.post_attention_layernorm.weight.data
            )

        # Copy final norm
        model.model.longtext.norm.weight.data.copy_(
            model.model.language_model.norm.weight.data
        )

        return model

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        longtext_input_ids: list[int] | torch.LongTensor | None = None,
        longtext_cu_seqlens: list[int]| torch.LongTensor | None = None,
        longtext_num_tokens: list[int] | torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            longtext_input_ids=longtext_input_ids,
            longtext_cu_seqlens=longtext_cu_seqlens,
            longtext_num_tokens=longtext_num_tokens,
            **kwargs,
        )

        hidden_states = outputs[0]

        # LM head (partial logits if logits_to_keep is set)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        # Add SVD auxiliary loss from the encoder merger
        svd_loss = getattr(outputs, "loss", None)
        if loss is not None and svd_loss is not None:
            weight = getattr(self.config.longtext_config, "svd_loss_weight", 0.01)
            loss = loss + weight * svd_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        longtext_input_ids=None,
        longtext_cu_seqlens=None,
        longtext_num_tokens=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            longtext_input_ids=longtext_input_ids,
            longtext_cu_seqlens=longtext_cu_seqlens,
            longtext_num_tokens=longtext_num_tokens,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if not is_first_iteration and use_cache:
            model_inputs["longtext_input_ids"] = None
            model_inputs["longtext_cu_seqlens"] = None
            model_inputs["longtext_num_tokens"] = None
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["image_grid_thw"] = None
            model_inputs["video_grid_thw"] = None

        return model_inputs

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        text_positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)

        # Early exit in case we are continuing generation from past kv
        past_length = 0
        if (cache := model_kwargs.get("past_key_values")) is not None:
            past_length = cache.get_seq_length()
        if past_length != 0 and self.model.rope_deltas is not None:
            position_ids = text_positions[None, ...] + self.model.rope_deltas
            return position_ids

        # Otherwise compute 3d position ids for vision tokens and concat with text position ids
        if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]

        is_input_ids = len(inputs_tensor.shape) == 2 and inputs_tensor.dtype in [torch.int, torch.long]
        if (
            is_input_ids
            and model_kwargs.get("mm_token_type_ids") is not None
            and (model_kwargs.get("image_grid_thw") is not None or model_kwargs.get("video_grid_thw") is not None)
        ):
            model_kwargs = {k: v for k, v in model_kwargs.items() if k != "input_ids"}
            vision_positions, rope_deltas = self.model.get_rope_index(inputs_tensor, **model_kwargs)
            self.model.rope_deltas = rope_deltas
        else:
            vision_positions = text_positions.unsqueeze(0).expand(3, -1, -1)
            self.model.rope_deltas = torch.zeros(
                inputs_tensor.shape[0], 1, dtype=torch.long, device=inputs_tensor.device
            )

        # Concatenate "text + vision" positions into [4, bs, seq-len]
        text_positions = text_positions[None, ...]
        position_ids = torch.cat([text_positions, vision_positions], dim=0)

        return position_ids
    


__all__ = ["LatentSeekerTextModel", "LatentSeekerEncoderModel", "LatentSeekerModel", "LatentSeekerForConditionalGeneration"]
