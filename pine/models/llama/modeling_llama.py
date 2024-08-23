"""Adapted from transformers.models.llama.modeling_llama."""

from transformers.models.llama.modeling_llama import *


_CONFIG_FOR_DOC = "LlamaConfig"

class LlamaRotaryEmbeddingExtension(LlamaRotaryEmbedding):
    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        dim_extend = len(position_ids.shape)-1
        inv_freq_expanded = self.inv_freq[(None,)*dim_extend].float().expand(*position_ids.shape[:-1], -1).unsqueeze(-1)
        position_ids_expanded = position_ids.unsqueeze(-2).float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(-1, -2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaAttentionWithPS(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbeddingExtension(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        ############### PS Content ####################

        # drop the original codes.
        # cos, sin = self.rotary_emb(value_states, position_ids)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


        # For the forward pass, we need to handle the repr of parallel docs.
        # Several arguments (bs must be 1):
        # doc_range: [num_docs, max_doc_len].
        # doc_mask: [num_docs, max_doc_len], 1 and 0 mask.
        # doc_select: [doc_len_total]
        # all_position_ids: are stacked ids
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
                position_shift = torch.zeros([bs, n_heads, num_docs, num_docs], dtype=torch.long, device=query_states.device)
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

                # Sorting
                diag_mask = torch.ones([num_docs], dtype=torch.long, device=attn_weights_doc2doc.device)
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

                doc_cos, doc_sin = self.rotary_emb(value_states, doc_position) # [bs, heads, all_input_len - suffix, hidden_dim]
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
                _doc_mask = torch.zeros(doc_mask_.shape, dtype=doc_attn_weights.dtype, device=doc_attn_weights.device)
                _doc_mask = _doc_mask.masked_fill(doc_mask_, torch.finfo(_doc_mask.dtype).min)
                doc_attn_weights += _doc_mask
                doc_attn_weights = nn.functional.softmax(doc_attn_weights, dim=-1, dtype=torch.float32).to(doc_q.dtype)
                doc_attn_outputs = torch.matmul(doc_attn_weights, doc_v) # [bs, heads, one_doc_length, hidden dim]

                attn_outputs_list.append(doc_attn_outputs)
                attn_weights_list.append(doc_attn_weights)
                
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {} # {"sin": sin, "cos": cos, "cache_position": cache_position} PS Content: Drop this line
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
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

        dynamic_cos, dynamic_sin = self.rotary_emb(value_states, dynamic_position) # [bs, heads, suffix_q_len, all_input_length, hidden_dim]
        
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
        # DROP INITAL CODES
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)

        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # if attention_mask is not None:  # no matter the length, we just slice it
        #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        #     attn_weights = attn_weights + causal_mask

        # # upcast attention to fp32
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # attn_output = torch.matmul(attn_weights, value_states)

        if is_forward_pass:
            prefix_token_q = query_states[:, :, :doc_st[0], :]
            prefix_token_k = key_states[:, :, :doc_st[0], :]
            prefix_token_v = value_states[:, :, :doc_st[0], :]
            cos, sin = self.rotary_emb(value_states, all_position_ids[:, :doc_st[0]])
            prefix_token_q, prefix_token_k = apply_rotary_pos_emb(prefix_token_q, prefix_token_k, cos, sin)
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

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# TODO: Only basic attention class is supported since I am lazy.
LLAMA_ATTENTION_CLASSES_PS = {
    "eager": LlamaAttentionWithPS,
}

class LlamaDecoderLayerWithPS(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES_PS[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

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
            cache_position=cache_position,
            **kwargs,
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


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top. Position stacking ver.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModelWithPS(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayerWithPS"]
    _skip_keys_device_placement = ["past_key_values"]
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

    def _setup_cache(self, cache_cls, max_batch_size, max_cache_len: Optional[int] = None):
        if self.config._attn_implementation == "flash_attention_2" and cache_cls == StaticCache:
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        for layer in self.model.layers:
            device = layer.input_layernorm.weight.device
            if hasattr(self.config, "_pre_quantization_dtype"):
                dtype = self.config._pre_quantization_dtype
            else:
                dtype = layer.self_attn.o_proj.weight.dtype
            layer.self_attn.past_key_value = cache_cls(
                self.config, max_batch_size, max_cache_len, device=device, dtype=dtype
            )

    def _reset_cache(self):
        for layer in self.model.layers:
            layer.self_attn.past_key_value = None


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top. Position stacking ver.",
    LLAMA_START_DOCSTRING,
)
class LlamaModelWithPS(LlamaPreTrainedModelWithPS):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayerWithPS(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
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
        cache_position: Optional[torch.LongTensor] = None,
        **ps_kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)

        # embed positions
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
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
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
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_seen_tokens: int,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        if self.config._attn_implementation == "sdpa":
            # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument,
            # in order to dispatch on Flash Attention 2.
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask, inputs_embeds=input_tensor, past_key_values_length=past_seen_tokens
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if hasattr(getattr(self.layers[0], "self_attn", {}), "past_key_value"):  # static cache
            target_length = self.config.max_position_embeddings
        else:  # dynamic cache
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class LlamaForCausalLMWithPS(LlamaPreTrainedModelWithPS):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModelWithPS(config)
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

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
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
        cache_position: Optional[torch.LongTensor] = None,
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
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        # assert all_position_ids is not None
        # assert doc_range is not None
        # assert doc_mask is not None
        # assert doc_select is not None
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
            cache_position=cache_position,
            **ps_kwargs
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
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
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        # With static cache, the `past_key_values` is None
        # TODO joao: standardize interface for the different Cache classes and remove of this if
        has_static_cache = False
        if past_key_values is None:
            past_key_values = getattr(getattr(self.model.layers[0], "self_attn", {}), "past_key_value", None)
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
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
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
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

