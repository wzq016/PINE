"""Adapted from transformers.models.qwen2.modeling_qwen2"""
from typing import Optional, Tuple, List, Union

import warnings
import math

import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    Qwen2Config,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen2.modeling_qwen2 import (
    logger,
    apply_rotary_pos_emb,
    repeat_kv,
    Qwen2MLP,
    Qwen2RMSNorm,
    BaseModelOutputWithPast,
    _prepare_4d_causal_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask,
    add_start_docstrings_to_model_forward,
    QWEN2_INPUTS_DOCSTRING,
    replace_return_docstrings,
    CausalLMOutputWithPast,
    _CONFIG_FOR_DOC,
    CrossEntropyLoss,
    Qwen2RotaryEmbedding,
    rotate_half
)

class Qwen2AttentionWithPS(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


        ############### PS Content ####################
        doc_range = kwargs.get('doc_range')
        doc_mask = kwargs.get('doc_mask')
        doc_select = kwargs.get('doc_select')
        all_position_ids = kwargs.get('all_position_ids')
        self_region_mask = kwargs.get('self_region_mask')
        num_docs, max_doc_len = doc_range.shape
        bs, all_input_len = all_position_ids.shape
        doc_st = doc_range[:, 0]
        doc_ed = torch.max(doc_range, dim=-1)[0] + 1
        doc_len = doc_mask.sum(dim=-1) # [num_docs]
        doc_len_tot = doc_select.shape[0]
        attn_outputs_list = []
        attn_weights_list = []
        is_forward_pass = False
        doc_range_region = (doc_range - doc_range[0, 0]) * doc_mask
        n_heads = query_states.shape[1]
        if past_key_value.get_seq_length(self.layer_idx) == 0:
            is_forward_pass = True
            if num_docs == 2:
                position_shift = torch.zeros([bs, n_heads, num_docs, num_docs], dtype=torch.long).to(query_states.device)
                position_shift[:, :, 0, 1] = doc_len[0]
                position_shift[:, :, 1, 0] = doc_len[1]
            else:
                # compute attn weight w/o position encoding
                key_states_no_rope = repeat_kv(key_states, self.num_key_value_groups)
                query_states_region = query_states[:, :, doc_st[0]: doc_ed[-1], :]
                key_states_region = key_states_no_rope[:, :, doc_st[0]: doc_ed[-1], :]
                attn_weights = torch.matmul(query_states_region, key_states_region.transpose(2, 3)) / math.sqrt(self.head_dim)
                region_mask = self_region_mask.to(attn_weights.dtype)
                region_mask = region_mask.masked_fill(self_region_mask == 1, torch.finfo(region_mask.dtype).min)
                attn_weights += region_mask
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

                # Compute per doc attn
                attn_weights_per_doc = attn_weights[:, :, :, doc_range_region] # [bs, heads, q_len, num_docs, max_doc_len]
                attn_weights_per_doc_mask = doc_mask[None, None, None, :, :].expand(1, 1, 1, num_docs, max_doc_len)
                attn_weights_per_doc *= attn_weights_per_doc_mask
                attn_weights_per_doc = attn_weights_per_doc.sum(dim=-1) # [bs, heads, q_len, num_docs]
                attn_weights_per_doc /= doc_len[None, None, None, :]

                # Compute doc2doc attn
                attn_weights_doc2doc = attn_weights_per_doc[:, :, doc_range_region, :] # [bs, heads, num_docs, max_doc_len, num_docs]
                attn_weights_doc2doc_mask = doc_mask[None, None, :, :, None].expand(1, 1, num_docs, max_doc_len, 1)
                attn_weights_doc2doc *= attn_weights_doc2doc_mask
                attn_weights_doc2doc = attn_weights_doc2doc.sum(dim=-2) # [bs, heads, num_docs, num_docs]
                # attn_weights_doc2doc = torch.randn(bs, query_states.shape[1], num_docs, num_docs, dtype=query_states.dtype).to(query_states.device) # OOD test

                # Sorting
                diag_mask = torch.ones([num_docs], dtype=torch.long).to(attn_weights_doc2doc.device)
                diag_mask = torch.diag(diag_mask, diagonal=0)[None, None, :, :].expand(1, 1, num_docs, num_docs)
                attn_weights_doc2doc = attn_weights_doc2doc.masked_fill(diag_mask == 1, torch.finfo(attn_weights_doc2doc.dtype).max)
                attn_weights_doc2doc_sort = torch.argsort(attn_weights_doc2doc, dim=-1, stable=True, descending=True) # [bs, heads, num_docs, num_docs]
                # For the above: value of [:, :, i, j] is the doc id that has jth largest attention score to doc i
                # Therefore, v[:, :, i, 0] = i
                attn_weights_doc2doc_sort_inverse = torch.argsort(attn_weights_doc2doc_sort, dim=-1)
                # For the above: value of [:, :, i, j] is the sorting index of attention score from doc j to i

                # Position shift
                position_shift = doc_len[attn_weights_doc2doc_sort]# [bs, heads, num_docs, num_docs]
                position_shift = torch.cumsum(position_shift, dim=-1) - position_shift
                position_shift = torch.gather(position_shift, dim=-1, index=attn_weights_doc2doc_sort_inverse)
                # Now position_shift[:, :, i, j] means the position shift of doc j when computing attention between i and j

            for doc_id in range(num_docs):
                doc_position_shift = position_shift[:, :, doc_id, :].clone() # [bs, heads, num_docs]
                doc_position_shift = doc_position_shift[:, :, doc_select] # [bs, heads, doc_len_total]
                doc_position = all_position_ids[:, None, :].expand(bs, doc_position_shift.shape[1], all_input_len).clone() 
                doc_position[:, :, doc_st[0]: doc_ed[-1]] -= doc_position_shift
                doc_position = doc_position[:, :, :doc_ed[-1]]

                posi_cos, posi_sin = self.rotary_emb(value_states, all_input_len) # [bs, heads, all_input_len - suffix, hidden_dim]
                doc_cos, doc_sin = posi_cos[doc_position], posi_sin[doc_position]
                doc_q = query_states[:, :, doc_st[doc_id]: doc_ed[doc_id],:]
                doc_q = ((doc_q * doc_cos[:, :, doc_st[doc_id]: doc_ed[doc_id], :]) 
                         + (rotate_half(doc_q) * doc_sin[:, :, doc_st[doc_id]: doc_ed[doc_id], :]))
                doc_k = key_states[:, :, :doc_ed[-1], :]
                doc_k = repeat_kv(doc_k, self.num_key_value_groups) # TODO: head num consistency?
                doc_k = (doc_k * doc_cos) + (rotate_half(doc_k) * doc_sin)
                doc_v = value_states[:, :, :doc_ed[-1], :]
                doc_v = repeat_kv(doc_v, self.num_key_value_groups)
                
                doc_attn_weights = torch.matmul(doc_q, doc_k.transpose(2, 3)) / math.sqrt(self.head_dim)
                doc_mask_ = doc_position[:, :, doc_st[doc_id]: doc_ed[doc_id]].unsqueeze(-1) < doc_position.unsqueeze(-2)
                _doc_mask = torch.zeros(doc_mask_.shape, dtype=doc_attn_weights.dtype).to(doc_attn_weights.device)
                _doc_mask = _doc_mask.masked_fill(doc_mask_, torch.finfo(_doc_mask.dtype).min)
                doc_attn_weights += _doc_mask
                doc_attn_weights = nn.functional.softmax(doc_attn_weights, dim=-1, dtype=torch.float32).to(doc_q.dtype)
                doc_attn_outputs = torch.matmul(doc_attn_weights, doc_v) # [bs, heads, one_doc_length, hidden dim]

                attn_outputs_list.append(doc_attn_outputs)
                attn_weights_list.append(doc_attn_weights)

        ############### PS Content Ends ####################


        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # # repeat k/v heads if n_kv_heads < n_heads
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)

        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
        #         f" {attn_weights.size()}"
        #     )

        # if attention_mask is not None:
        #     if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
        #         raise ValueError(
        #             f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
        #         )

        #     attn_weights = attn_weights + attention_mask

        # # upcast attention to fp32
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # attn_output = torch.matmul(attn_weights, value_states)

        ############### PS Content ####################
        # Now we need to handle the suffix tokens, or generated tokens (len=1)
        
        # Change: Unify the query states of the two cases
        suffix_query_states = query_states[:, :, doc_ed[-1]:, :] if is_forward_pass else query_states
        suffix_q_len = suffix_query_states.shape[-2]

        # compute attn weight w/o position encoding
        key_states_no_rope = repeat_kv(key_states, self.num_key_value_groups)
        key_states_region = key_states_no_rope[:, :, doc_st[0]:doc_ed[-1], :]
        attn_weights = torch.matmul(suffix_query_states, key_states_region.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(suffix_query_states.dtype)

        # Compute per doc attn
        attn_weights_per_doc = attn_weights[:, :, :, doc_range_region] # [bs, heads, suffix_q_len, num_docs, max_doc_len]
        attn_weights_per_doc_mask = doc_mask[None, None, None, :, :].expand(1, 1, 1, num_docs, max_doc_len)
        attn_weights_per_doc *= attn_weights_per_doc_mask
        attn_weights_per_doc = attn_weights_per_doc.sum(dim=-1) # [bs, heads, suffix_q_len, num_docs]
        attn_weights_per_doc /= doc_len[None, None, None, :]

        # Sorting. Change: No masks are needed here.
        attn_weights_per_doc_sort = torch.argsort(attn_weights_per_doc, dim=-1, stable=True, descending=True) # [bs, heads, suffix_q_len, num_docs]
        # For the above: value of [:, :, i, j] is the doc id that has jth largest attention score to token i
        attn_weights_per_doc_sort_inverse = torch.argsort(attn_weights_per_doc_sort, dim=-1)
        # For the above: value of [:, :, i, j] is the sorting index of attention score from doc j to token i

        # Position shift
        position_shift = doc_len[attn_weights_per_doc_sort] # [bs, heads, suffix_q_len, num_docs]
        position_shift = torch.cumsum(position_shift, dim=-1) - position_shift
        position_shift = torch.gather(position_shift, dim=-1, index=attn_weights_per_doc_sort_inverse)
        # Now position_shift[:, :, i, j] means the position shift of doc j when computing attention between i and j


        # Post-process. Change: no for-loop now
        position_shift = position_shift[:, :, :, doc_select] # [bs, heads, suffix_q_len, doc_len_tot]
        dynamic_position = all_position_ids[:, None, None, :].expand(bs, position_shift.shape[1], 
                                                                     position_shift.shape[2], all_input_len).clone() # [bs, 1, 1, all input length]
        dynamic_position[:, :, :, doc_st[0]: doc_ed[-1]] -= position_shift

        posi_cos, posi_sin = self.rotary_emb(value_states, all_input_len)
        dynamic_cos, dynamic_sin = posi_cos[dynamic_position], posi_sin[dynamic_position] # [bs, heads, suffix_q_len, all_input_length, hidden_dim]
        
        dynamic_q = ((suffix_query_states * dynamic_cos[:, :, 0, -suffix_q_len:, :]) 
                    + (rotate_half(suffix_query_states) * dynamic_sin[:, :, 0, -suffix_q_len:, :])) # [bs, heads, suffix_q_len, hidden_dim]
        dynamic_q = dynamic_q.unsqueeze(-2) # [bs, heads, suffix_q_len, 1, hidden_sdim]
        dynamic_k = repeat_kv(key_states, self.num_key_value_groups) # TODO: head num consistency?
        dynamic_k = dynamic_k.unsqueeze(2) #[bs, heads, 1, all_input_len, dims]
        dynamic_k = (dynamic_k * dynamic_cos) + (rotate_half(dynamic_k) * dynamic_sin)
        dynamic_v = repeat_kv(value_states, self.num_key_value_groups) # [bs, heads, all_input_len, dims]

        dynamic_attn_weights = torch.matmul(dynamic_q, dynamic_k.transpose(3, 4)) / math.sqrt(self.head_dim)
        dynamic_attn_weights = dynamic_attn_weights.squeeze(-2)
        dynamic_attn_weights += attention_mask[:, :, -suffix_q_len:, :]
        dynamic_attn_weights = nn.functional.softmax(dynamic_attn_weights, dim=-1, dtype=torch.float32).to(dynamic_q.dtype) # [bs, heads, suffix_q_len, 1, all_input_len]
        dynamic_attn_outputs = torch.matmul(dynamic_attn_weights, dynamic_v) # [bs, heads, suffix_q_len, dims]

        attn_outputs_list.append(dynamic_attn_outputs)
        attn_weights_list.append(dynamic_attn_weights)

        ############### PS Content Ends ####################

        ############### PS Content ####################
        if is_forward_pass:
            prefix_token_q = query_states[:, :, :doc_st[0], :]
            prefix_token_k = key_states[:, :, :doc_st[0], :]
            prefix_token_v = value_states[:, :, :doc_st[0], :]
            prefix_token_q, prefix_token_k = apply_rotary_pos_emb(prefix_token_q, prefix_token_k, posi_cos, posi_sin, all_position_ids[:, :doc_st[0]])
            prefix_token_k = repeat_kv(prefix_token_k, self.num_key_value_groups)
            prefix_token_v = repeat_kv(prefix_token_v, self.num_key_value_groups)
            attn_weights = torch.matmul(prefix_token_q, prefix_token_k.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_weights += attention_mask[:, :, :doc_st[0], :doc_st[0]]
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_outputs = torch.matmul(attn_weights, prefix_token_v)
            attn_outputs_list = [attn_outputs] + attn_outputs_list # [bs, heads, ?, dim] 
            attn_weights_list = [attn_weights] + attn_weights_list # TODO: cannot be concat.
        
        attn_output = torch.cat(attn_outputs_list, dim=2)
        attn_weights = None

        ############### PS Content Ends ####################

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


QWEN2_ATTENTION_CLASSES_PS = {
    "eager": Qwen2AttentionWithPS
}

class Qwen2DecoderLayerWithPS(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_ATTENTION_CLASSES_PS[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class Qwen2PreTrainedModelWithPS(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayerWithPS"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen2ModelWithPS(Qwen2PreTrainedModelWithPS):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayerWithPS(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **ps_kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **ps_kwargs
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class Qwen2ForCausalLMWithPS(Qwen2PreTrainedModelWithPS):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2ModelWithPS(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        all_position_ids = None,
        doc_range = None,
        doc_mask = None,
        doc_select = None,
        self_region_mask = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        ps_kwargs = {
            'all_position_ids' : all_position_ids,
            'doc_range': doc_range,
            'doc_mask': doc_mask,
            'doc_select': doc_select,
            'self_region_mask': self_region_mask
        }
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **ps_kwargs
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        all_position_ids = None
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            ############### PS Content ####################
            all_position_ids = position_ids.clone()
            _all_position_ids = kwargs.get("all_position_ids")
            all_position_ids[:, :_all_position_ids.shape[-1]] = _all_position_ids
            ############### PS Content Ends ####################
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                ############### PS Content ####################
                "all_position_ids": all_position_ids,
                "doc_range": kwargs.get("doc_range"),
                "doc_mask": kwargs.get("doc_mask"),
                "doc_select": kwargs.get("doc_select"),
                "self_region_mask": kwargs.get("self_region_mask"),
                ############### PS Content Ends ####################
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
